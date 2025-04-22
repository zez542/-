from einops import rearrange
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class DCNet(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, stride=1, first=False):
        super(DCNet, self).__init__()
        self.stride = stride

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, kernel_size=1),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(True),
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch//2, in_ch//2, 3, padding=1, stride=stride),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(True),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_ch // 2, in_ch // 2, 3, padding=1),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(True),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_ch // 2, in_ch // 2, 3, padding=1),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(True),
        )

        self.conv_final = nn.Sequential(
            nn.Conv2d(in_ch//2, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            eca_layer(),
        )

        if in_ch != out_ch or first == True:
            self.identity = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )
        if in_ch == out_ch and first == False:
            self.identity = nn.Identity()

        if self.stride == 1:
            self.shortcut = nn.Identity()
        if self.stride == 2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch//2, in_ch//2, kernel_size=1, stride=2),
                nn.BatchNorm2d(in_ch//2),
                nn.ReLU(True),
            )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        input = x
        x_ = self.conv(x)
        x1 = self.conv_1(x_)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        x4 = self.shortcut(x_) + x1 + x2 + x3
        x5 = self.conv_final(x4)

        identity = self.identity(input)
        out = self.relu(x5 + identity)
        return out


class MSAM(nn.Module):
    def __init__(self, in_channels):
        super(MSAM, self).__init__()
        # 不同尺度的池化层
        self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pool_5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.max_pool_7 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)  # 修改padding值为3，以适配7x7池化核

        # 增加额外的卷积层用于初步处理不同尺度特征
        self.conv_1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

        # 融合多尺度信息的卷积层，这里可以使用多个卷积层来深度融合
        self.conv_fusion_1 = nn.Conv2d(in_channels * 2, in_channels * 3 // 4, kernel_size=1)  # 修改通道数，适配4个尺度特征
        self.bn_fusion_1 = nn.BatchNorm2d(in_channels * 3 // 4)
        self.relu_fusion_1 = nn.ReLU(inplace=True)

        self.conv_fusion_2 = nn.Conv2d(in_channels * 3 // 4, in_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_size = x.size()
        avg_pooled = self.avg_pool_1(x)
        max_pooled_3 = self.max_pool_3(x)
        max_pooled_5 = self.max_pool_5(x)
        max_pooled_7 = self.max_pool_7(x)

        # 对不同尺度特征进行初步卷积处理
        avg_pooled_conv = self.conv_1(avg_pooled)
        avg_pooled_conv = F.interpolate(avg_pooled_conv, x_size[2:], mode='bilinear', align_corners=True)
        max_pooled_3_conv = self.conv_1(max_pooled_3)
        max_pooled_5_conv = self.conv_1(max_pooled_5)
        max_pooled_7_conv = self.conv_1(max_pooled_7)

        # 拼接多尺度特征
        multi_scale_features = torch.cat((avg_pooled_conv, max_pooled_3_conv, max_pooled_5_conv, max_pooled_7_conv), dim=1)

        # 深度融合多尺度特征
        fused_features_1 = self.conv_fusion_1(multi_scale_features)
        fused_features_1 = self.bn_fusion_1(fused_features_1)
        fused_features_1 = self.relu_fusion_1(fused_features_1)

        fused_features_2 = self.conv_fusion_2(fused_features_1)

        # 生成注意力权重
        attention_weights = self.sigmoid(fused_features_2)

        # 将注意力权重与原始特征相乘
        output = x * attention_weights


        return output + x


class Block(nn.Module):
    def __init__(self, inc, outc, stride):
        super(Block, self).__init__()
        self.Conv = nn.Sequential(
            DCNet(inc, outc, stride=stride),
            DCNet(outc, outc, stride=1)
        )

    def forward(self, x):
        out = self.Conv(x)
        return out


class net(nn.Module):
    def __init__(self, ):
        super(net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.block1 = Block(32, 64, 2)
        self.block2 = Block(64, 128, 2)
        self.block3 = Block(128, 256, 2)
        self.block4 = Block(256, 512, 2)

        self.msam1 = MSAM(64)
        self.msam2 = MSAM(128)
        self.msam3 = MSAM(256)
        self.msam4 = MSAM(512)



    def forward(self, x):
        x = self.conv(x) #32x256x256
        layer1 = self.block1(x) #64x128x128
        layer1 = self.msam1(layer1)
        layer2 = self.block2(layer1) #128x64x64
        layer2 = self.msam2(layer2)
        layer3 = self.block3(layer2) #256x32x32
        layer3 = self.msam3(layer3)
        layer4 = self.block4(layer3) #512x16x16
        layer4 = self.msam4(layer4)

        return layer1, layer2, layer3, layer4

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class FEM(nn.Module):
    def __init__(self, in_c):
        super(FEM, self).__init__()
        self.conv = ConvBlock(in_c, in_c, kernel_size=1, padding=0)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels=in_c * 2, out_channels=in_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_c),
            nn.ReLU(True),

        )
        self.conv3x3 = ConvBlock(in_c, in_c)

    def forward(self, x1, x2):
        abs_x = torch.abs(x1 - x2)
        add_x = x1 + x2
        x_ = self.GAP(abs_x)
        x__ = torch.mul(abs_x, x_)
        abs_x_out = self.sigmoid(x__) + abs_x

        add_x_out = self.conv3x3(add_x)

        out = torch.cat([abs_x_out, add_x_out], dim=1)
        out = self.doubleconv(out)
        return out



class ABC(nn.Module):
    def __init__(self, num_classes):
        super(ABC, self).__init__()

        self.net = net()

        self.fem1 = FEM(64)
        self.fem2 = FEM(128)
        self.fem3 = FEM(256)
        self.fem4 = FEM(512)


        """上采样"""
        self.up4 = nn.Sequential(
            MSAM(512),
            Block(512, 256, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )

        self.up3 = nn.Sequential(
            MSAM(256),
            Block(256, 128, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )

        self.up2 = nn.Sequential(
            MSAM(128),
            Block(128, 64, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )

        self.up1 = nn.Sequential(
            MSAM(64),
            Block(64, 32, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )



        self.output =nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x1, x2):
        x1_layer1, x1_layer2, x1_layer3, x1_layer4 = self.net(x1)
        x2_layer1, x2_layer2, x2_layer3, x2_layer4 = self.net(x2)


        dx_layer1 = self.fem1(x1_layer1, x2_layer1)
        dx_layer2 = self.fem2(x1_layer2, x2_layer2)
        dx_layer3 = self.fem3(x1_layer3, x2_layer3)
        dx_layer4 = self.fem4(x1_layer4, x2_layer4)

        up4 = self.up4(dx_layer4)  # 512X16X16

        out3 = dx_layer3 + up4  # 512X16X16
        up3 = self.up3(out3)  # 256X32X32

        out2 = dx_layer2 + up3  # 256X32X32
        up2 = self.up2(out2)  # 128X64X64

        out1 = dx_layer1 + up2  # 128X64X64
        up1 = self.up1(out1)  # 64x256x256

        output = self.output(up1)
        result = torch.nn.functional.interpolate(output, scale_factor=2, mode='bilinear')
        return result


if __name__ == '__main__':
    x1 = torch.rand(4, 3, 256, 256)
    x2 = torch.rand(4, 3, 256, 256)
    model = ABC(2)
    a = model(x1, x2)
    print(a.shape)

    from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    out = model_eval(x1, x2)
    print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + get_model_parameters_number(model))
    print('Output shape: {}'.format(list(out.shape)))
    total_paramters = sum(p.numel() for p in model.parameters())
    print('Total paramters: {}'.format(total_paramters))

    import time

    if torch.cuda.is_available():
        model = model.cuda()  # .half()  #HALF seems to be doing slower for some reason
        x1 = x1.cuda()  # .half()
        x2 = x2.cuda()

    time_train = []
    i = 0
    # model.load_state_dict(torch.load("../Testmodel_List/KR94187_Portrait_98/result/Dnc_C3Portrait/model_266.pth",
    #                       map_location=torch.device(device='cpu')))
    # 0.273
    while (i < 20):
        # for step, (images, labels, filename, filenameGt) in enumerate(loader):

        start_time = time.time()

        inputs1 = torch.autograd.Variable(x1)
        inputs2 = torch.autograd.Variable(x2)
        with torch.no_grad():
            outputs = model(x1, x2)

        # preds = outputs.cpu()
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)

        if i != 0:  # first run always takes some time for setup
            fwt = time.time() - start_time
            time_train.append(fwt)
            print("Forward time per img (b=%d): %.3f (Mean: %.3f)" % (
                1, fwt / 1, sum(time_train) / len(time_train) / 1))

        time.sleep(1)  # to avoid overheating the GPU too much
        i += 1

