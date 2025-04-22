import os, argparse
import torch
from LoadData import Change_Detect
from train import train
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')

# from TRI.model import BaseNet
# from DABNet.DABNet import DABNet
# from BiseNet.build_BiSeNet import BiSeNet
# from MFGAN.our_model2 import our_model
# from TCDNet.our_model import our_model
# from MY_NET import MY_NET
# from segnet.segnet import SegNet
# from UNet.UNet import UNet
# from ICNet.icnet import ICNet
# from SNUNet.Models import Siam_NestedUNet_Conc
# from HRNet.HRnet import HighResolutionNet
# from BiseNet.build_BiSeNet import BiSeNet
# from ENet.enet import ENet
# from FC_EF.FC_EF import Unet
# from FC_CONC.FC_Conc import SiamUnet_conc
# from FC_DIFF.FC_Diff import FC_Siam_diff
# from ChangeNet.ChangNet import ChangNet
# from ICIFNet.ICIFNet import ICIFNet
# from BIT.ChangeFormer import ChangeFormerV2
# from FCN.fcn_v2 import FCN8s
# from PSPnet.pspnet import PSPNet
# from DeepLabV3.DeepLabv3_plus import DeepLabv3_plus
# from YHY.MY_NET_G4_P import MY_NET
# from ABMFNet import ABMFNet
# from My_model import My_model
# from xiaorong.MY_model import our_model1
# from ABC import ABC
# from ABC2 import ABC
# from ABC2MSAMFEM import ABC
# from ABC2SAMFEM import ABC
# from ABC3 import ABC
# from ABC3MSAM import ABC
# from ABC4MSAMFEM import ABC

# from ABC2MSAMGSAM import ABC


def main(params):
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs',          type=int,       default=200,)
    parser.add_argument('--checkpoint_step',     type=int,       default=5, )
    parser.add_argument('--validation_step',     type=int,       default=1, )
    parser.add_argument('--batch_size',          type=int,       default=4, )
    parser.add_argument('--num_workers',         type=int,       default=4, )
    parser.add_argument('--lr',                  type=float,     default=0.0001, )
    parser.add_argument('--lr_scheduler',        type=int,       default=3, )
    parser.add_argument('--lr_scheduler_gamma',  type=float,     default=0.99,       help='learning rate attenuation coefficient')
    parser.add_argument('--warmup',              type=int,       default=1,          help='warm up')
    parser.add_argument('--cuda',                type=str,       default='0')
    parser.add_argument('--DataParallel',        type=int,       default=1,          help='train in multi GPU')
    parser.add_argument('--beta1',               type=float,     default=0.5)
    parser.add_argument('--beta2',               type=float,     default=0.999)
    parser.add_argument('--miou_max',            type=float,     default=0.80)
    parser.add_argument('--pretrained_model_path',  type=str,       default=None,       help='None')
    parser.add_argument('--save_model_path',     type=str,       default="./ABC2/BCDD")
    parser.add_argument('--data',                type=str,       default="C:/Users/Liyan/Desktop/CD/Public_Dataset/BCDD")
    parser.add_argument('--log_path',            type=str,       default="./log")
    parser.add_argument('--result',              type=str,       default="./")
    parser.add_argument('--model_name',          type=str,       default='dfn',      help='')
    parser.add_argument('--dir_name',            type=str,       default='dfn',     help='')


    args=parser.parse_args(params)

    # 打印 params 的信息
    tb = PrettyTable(['Num', 'Key', 'Value'])
    args_str = str(args)[10:-1].split(', ')
    for i, key_value in enumerate(args_str):
        key, value = key_value.split('=')[0], key_value.split('=')[1]
        tb.add_row([i + 1, key, value])
    print(tb)

    # 检查文件夹是否存在，不存在就创建
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    # if not os.path.exists(args.result):
    #     os.makedirs(args.result)

    if not os.path.exists(f'{args.save_model_path}/{args.model_name}'):
        os.makedirs(f'{args.save_model_path}/{args.model_name}')
    # if not os.path.exists(args.summary_path+args.dir_name+'/checkpoints'):
    #     os.makedirs(args.summary_path+args.dir_name+'/checkpoints')

    # 创建数据集和数据加载器
    # 训练集
    train_path_img1  = os.path.join(args.data, 'train/img1')
    train_path_img2  = os.path.join(args.data, 'train/img2')
    train_path_label = os.path.join(args.data, 'train/label')# os.path.join()函数：连接两个或更多的路径名组件
    # 验证集
    val_path_img1    = os.path.join(args.data, 'test/img1')
    val_path_img2    = os.path.join(args.data, 'test/img2')
    val_path_label   = os.path.join(args.data, 'test/label')
    #
    csv_path         = os.path.join(args.data, 'class_dict.csv')

    # 训练集数据加载
    dataset_train    = Change_Detect(
                                    train_path_img1,
                                    train_path_img2,
                                    train_path_label,
                                    csv_path,
                                    mode='train',
                                    )
    dataloader_train = DataLoader(
                                    dataset_train,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    # drop_last=True,
                                    num_workers=args.num_workers,
                                  )
    # 验证集数据加载
    dataset_val      =Change_Detect(
                                    val_path_img1,
                                    val_path_img2,
                                    val_path_label,
                                    csv_path,
                                    mode='val'
                                    )
    dataloader_val = DataLoader(
                                    dataset_val,
                                    batch_size=1,  # must 1
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                )

    #
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda #s.environ[‘环境变量名称’]=‘环境变量值’
    torch.backends.cudnn.benchmark = True
    # 选择模型

    # model = our_model1(2, 'resnet34')
    # model = ABC(2)
    # model = SiamUnet_conc()
    # model = FC_Siam_diff()
    # model = ChangNet()
    # model = ChangeFormerV2()
    # model = ICIFNet()
    # model = BaseNet(input_nc=3, output_nc=2)
    # model = DABNet()
    # model = BiSeNet(2, 'resnet50')
    # model = our_model(2, 'resnet34')
    # model = MY_NET()
    # model = FCN8s()
    # model = PSPNet(psp_size=2048, deep_features_size=1024, backend='resnet34')
    # model = DeepLabv3_plus(nInputChannels=6, n_classes=2, os=16, pretrained=False, _print=True)
    # model = ICNet(nclass=2, backbone='resnet50', pretrained_base=True)
    # model = Siam_NestedUNet_Conc(in_ch=3, out_ch=2)
    # model = HighResolutionNet()
    # model = ABMFNet(2, 'resnet34')
    # model = My_model(2, 'resnet18')
    # model = BiSeNet(2, 'resnet34')
    # model = ENet(2)
    # model = Unet(input_nbr=6, label_nbr=2)
    # model = UNet()
    # model = our_model1(2, 'resnet34')


    # 设置GPU
    # if args.DataParallel == 1:
    #     print('mulit Cuda! cuda:{:}'.format(args.cuda))
    #     model = torch.nn.DataParallel(model)
    #     model = model.cuda()
    # else:
    #     print('Single cuda!')
    #     model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Adam梯度下降
    optimizer = torch.optim.Adam(model.parameters(),args.lr,(args.beta1,args.beta1))
    lr_scheduler = StepLR(optimizer, step_size=args.lr_scheduler, gamma=args.lr_scheduler_gamma)

    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)

        # loading the part of network params
        pretrained_dict = torch.load(args.pretrained_model_path)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        print('Done!')

    # 开始训练
    train(args, model, optimizer, dataloader_train, dataloader_val, lr_scheduler)

if __name__ == '__main__':
    params = [
        '--num_epochs', '200',
        '--batch_size', '8',
        '--lr', '0.001',
        '--warmup', '0',
        '--lr_scheduler_gamma', '0.9',
        '--lr_scheduler', '4',
        '--miou_max', '0.70',
        '--DataParallel', '1',  # 1: True  0:False
        '--cuda', '0',  # model put in the cuda[0]
        '--checkpoint_step', '20',
        '--result', 'BCDD_result',
        '--model_name', 'BCDD',
        #         # '--dir_name', 'FC_DIFF',
        #         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
    ]

    main(params)


# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.75',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '20',
#         '--result', 'MY_NET_G4_16',
#         '--model_name','MY_NET_G4_16',
        # '--dir_name', 'FC_DIFF',
        # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
        #     ]

    # main(params)


# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.80',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_G4_18',
#         '--model_name','MY_NET_G4_18',
#         '--dir_name', 'FC_DIFF',
        # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
        #     ]

    # main(params)

# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.6',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_G4_11',
#         '--model_name','MY_NET_G4_11',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)

# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.6',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_G4_12',
#         '--model_name','MY_NET_G4_12',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)

# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.6',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_G4_13',
#         '--model_name','MY_NET_G4_13',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)

# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.6',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_G4_14',
#         '--model_name','MY_NET_G4_14',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)
#
# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.6',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_G4_15',
#         '--model_name','MY_NET_G4_15',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)

# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.6',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_G4_13',
#         '--model_name','MY_NET_G4_13',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)

# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.82',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_G2',
#         '--model_name','MY_NET_G2',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)

# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.82',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '20',
#         '--result', 'MY_NET_result',
#         '--model_name', 'MY_NET',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)

# if __name__ == '__main__':
#     params = [
#          '--num_epochs', '200',
#          '--batch_size', '6',
#          '--lr', '0.0015',
#          '--warmup', '0',
#          '--lr_scheduler_gamma', '0.9',
#          '--lr_scheduler', '4',
#          '--miou_max', '0.83',
#          '--DataParallel', '1',  # 1: True  0:False
#          '--cuda', '0',  # model put in the cuda[0]
#          '--checkpoint_step', '40',
#          '--result', 'MY_NET_G4_P',
#          '--model_name','MY_NET_G4_P',
# #         # '--dir_name', 'FC_DIFF',
# #         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#              ]
#
#     main(params)


# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.82',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_G4_1',
#         '--model_name','MY_NET_G4_1',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)

# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.82',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_G4_2',
#         '--model_name','MY_NET_G4_2',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)

# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.82',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_G4_3',
#         '--model_name','MY_NET_G4_3',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)

# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.830',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_G4_4',
#         '--model_name','MY_NET_G4_4',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)


# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.830',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_1',
#         '--model_name','MY_NET_1',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)

# if __name__ == '__main__':
#     params = [
#         '--num_epochs', '200',
#         '--batch_size', '6',
#         '--lr', '0.0015',
#         '--warmup', '0',
#         '--lr_scheduler_gamma', '0.9',
#         '--lr_scheduler', '4',
#         '--miou_max', '0.70',
#         '--DataParallel', '1',  # 1: True  0:False
#         '--cuda', '0',  # model put in the cuda[0]
#         '--checkpoint_step', '40',
#         '--result', 'MY_NET_2',
#         '--model_name','MY_NET_2',
#         # '--dir_name', 'FC_DIFF',
#         # '--pretrained_model_path', 'C:\YHY\CDYHY\MY_NET\checkpoint\MY_NET_V3\max_epoch_106_miou_0.811596.pth'
#             ]
#
#     main(params)




