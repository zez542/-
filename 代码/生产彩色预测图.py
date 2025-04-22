import os
os.chdir('../')
import math
import pdb
import argparse, torch, cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from utils2 import *
import warnings
warnings.filterwarnings('ignore')
# import glob
# img_list = glob.glob('D:/python/pycharm/daima/change_detection/data1/test/img1'+'/*.jpg')
# print(img_list)

# from MY_NET import MY_NET
# from our_model6 import our_model6
# from PSPnet.pspnet import PSPNet
# from our_model2 import our_model2
# from segnet.segnet import SegNet
# from FCN.fcn_v2 import FCN8s
# from UNet.UNet import UNet
# from DeepLabV3.DeepLabv3_plus import DeepLabv3_plus
# from TRI.model import BaseNet
# from ENet.enet import ENet
# from FC_EF.FC_EF import Unet
# from FC_CONC.FC_Conc import SiamUnet_conc
# from FC_DIFF.FC_Diff import FC_Siam_diff
# from ChangeNet.ChangNet import ChangNet
# from xiaorong.MY_model5 import our_model5
# from MFGAN.our_model2 import our_model
# from TCDNet.our_model import our_model
# from BiseNet.build_BiSeNet import BiSeNet
# from HRNet.HRnet import HighResolutionNet
# from DABNet.DABNet import DABNet
# from My_model import My_model
# from YHY.MY_NET_G4_P import MY_NET
# from ABMFNet import ABMFNet
# from SNUNet.Models import Siam_NestedUNet_Conc
# from BIT.ChangeFormer import ChangeFormerV2
# from ABC2MSAMFEM import ABC
from ABC2SAMFEM import ABC
# test

# 定义双时态图片文件夹路径
file_location = r'C:\Users\Liyan\Desktop\CD\Public_Dataset\BCDD\test\img1'
file_location1 = r'C:\Users\Liyan\Desktop\CD\Public_Dataset\BCDD\test\img2'
file_location2 = r'C:\Users\Liyan\Desktop\CD\Public_Dataset\BCDD\test\label'

# 批量读取文件名称
file_list = []
file_list1 = []
file_list2 = []
file_name = []

for root, dirs, files in os.walk(file_location):
    file_name.append(files)
    for filename in files:
        if filename.endswith('.png'):
            file_path = os.path.join(root, filename)
            file_path = os.path.abspath(file_path)
            file_list.append(file_path)

file_name = file_name[0]
for root, dirs, files in os.walk(file_location1):
    for filename in files:
        if filename.endswith('.png'):
            file_path = os.path.join(root, filename)
            file_path = os.path.abspath(file_path)
            file_list1.append(file_path)

for root, dirs, files in os.walk(file_location2):
    for filename in files:
        if filename.endswith('.png'):
            file_path = os.path.join(root, filename)
            file_path = os.path.abspath(file_path)
            file_list2.append(file_path)

# 预测函数
def predict_on_image(model):
    to_tensor = transforms.ToTensor()
    # for i, (img1, img2, label) in enumerate(dataloader_train):
    #     img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
    for i in range(len(file_list)):
        img1 = Image.open(file_list[i])
        img1 = to_tensor(img1).float().unsqueeze(0)#(1,3,256,256)
        img2 = Image.open(file_list1[i])
        img2 = to_tensor(img2).float().unsqueeze(0)
        label_info = get_label_info()
        model.eval()
        # y1, dx1 = model(img1, img2)#将需要的哪一层热力图输出定义
        # imshowAtt(x=dx1, y=file_list[i])
        predict = model(img1, img2)[0].squeeze()
        predict = reverse_one_hot(predict)
        predict = colour_code_segmentation(np.array(predict), label_info)
        predict = Image.fromarray(np.uint8(predict))

        label = Image.open(file_list2[i])

        # 定义要应用的彩色（以RGB值表示）
        miss_color = (255, 0, 0)  # 红色
        wrong_color = (0, 0, 255)  # 蓝色
        # 将预测图像和标签图像转换为张量
        predict_tensor = transforms.ToTensor()(predict)
        label_tensor = transforms.ToTensor()(label)

        # 将预测图像与标签图像对比，将预测不正确的地方改变颜色
        for j in range(label.size[0]):
            for k in range(label.size[1]):
                if (predict_tensor[0][j][k] != label_tensor[0][j][k]) and (predict_tensor[0][j][k] == 0):
                    predict.putpixel((k, j), miss_color)
                elif (predict_tensor[0][j][k] != label_tensor[0][j][k]) and (predict_tensor[0][j][k] != 0):
                    predict.putpixel((k, j), wrong_color)

        predict = np.array(predict)
        # 将张量转换回PIL图像
        colorized_predict = transforms.ToPILImage()(predict)

        # 黑白图存放位置
        save_folder = r'D:\LY\SNUNet\BCDD_result'
        # 检查保存文件夹是否存在，如果不存在则创建
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        path = os.path.join(save_folder, str((file_name[i])[:-4]) + '.png')
        colorized_predict.save(path)

if __name__ == '__main__':
    print('load model  ...' )
    "导入模型权重"
    # pdb.set_trace()

    # model = ABC(2)
    # model_path = r'D:\LY\ABC2MSAMFEM\BCDD\BCDD\max_epoch_111_miou_0.810523.pth'

    # model = PSPNet(psp_size=2048, deep_features_size=1024, backend='resnet34')
    # model_path = r'D:\LY\PSPnet\PSPnet.checkpoint\PSPnet_LEVIR\max_epoch_32_miou_0.847909.pth'

    # model = SiamUnet_conc()
    # model_path = r'D:\LY\FC_CONC\LEVIR\LEVIR\max_epoch_139_miou_0.845675.pth'

    # model = FC_Siam_diff()
    # model_path = r'D:\LY\FC_DIFF\LEVIR\LEVIR\max_epoch_175_miou_0.842441.pth'

    # model = ChangNet()
    # model_path = r'D:\LY\ChangeNet\LEVIRlast.pth'

    # model = MY_NET()
    # model_path = r'D:\LY\YHY\checkpoint\MY_NET_G4_P\max_epoch_118_miou_0.842873.pth'

    # model = ABMFNet(2, 'resnet34')
    # model_path = r'D:\LY\ABMFNet\LEVIR\LEVIR\max_epoch_18_miou_0.858383.pth'

    # model = My_model(2, 'resnet34')
    # model_path = r'D:\LY\My_mode2.checkpointlast.pth'

    model = Siam_NestedUNet_Conc(in_ch=3, out_ch=2)
    model_path = r'D:\LY\SNUNet\BCDD\BCDD\max_epoch_194_miou_0.794790.pth'

    # model = our_model(2, 'resnet34')
    # model_path = r'D:\LY\TCDNet\TCDNet.checkpoint\LEVIR\max_epoch_34_miou_0.829367.pth'

    # model = BaseNet(input_nc=3, output_nc=2)
    # model_path = r'D:\LY\TRI\TRI-GR\TRI-GR.checkpoint\TRI-GR_LEVIR\max_epoch_14_miou_0.851496.pth'

    # model = DABNet()
    # model_path = r'D:\LY\DASNet\DASNet_LEVIR\max_epoch_39_miou_0.833382.pth'

    # model = HighResolutionNet()
    # model_path = r'D:\LY\HRNet\HRNet.checkpoint\HRNet_LEVIR\max_epoch_29_miou_0.847845.pth'
    #
    # model = BiSeNet(2, 'resnet50')
    # model_path = r'D:\LY\BiseNet\BiseNet.checkpoint\BiseNet_LEVIR\max_epoch_53_miou_0.833868.pth'

    # model = SegNet()
    # model_path = r'D:\LY\segnet\SegNet.checkpoint\SegNet_LEVIR\max_epoch_38_miou_0.835365.pth'

    # model = our_model(2, 'resnet34')
    # model_path = r'D:\LY\MFGAN\MFGAN.checkpoint\LEVIR\max_epoch_48_miou_0.835981.pth'

    # model = ChangeFormerV2()
    # model_path = r'D:\LY\BIT\checkpointlast.pth'

    # model = ENet(2)
    # model_path = r'D:\LY\ENet\ENet.checkpoint\ENet_LEVIR\max_epoch_63_miou_0.839098.pth'

    # model = Unet(input_nbr=6, label_nbr=2)
    # model_path = r'D:\LY\FC_EF\FC_EF.checkpoint\FC_EF_LEVIR\max_epoch_197_miou_0.825458.pth'

    # model = DeepLabv3_plus(nInputChannels=6, n_classes=2, os=16, pretrained=False, _print=True)
    # model_path = r'D:\LY\DeepLabV3\DeepLabV3+.checkpoint\DeepLabV3+_LEVIR\max_epoch_48_miou_0.850600.pth'

    # model =UNet()
    # model_path = r'D:\LY\UNet\UNet.checkpointlast.pth'

    try:
        pretrained_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(pretrained_dict)
        print("single GPU!")
    except:
        def load_GPUS(model, model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`kua
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)
            return model
        model = load_GPUS(model, model_path=model_path)
        print("multi GPUs!")

    print('Done!')
    predict_on_image(model)