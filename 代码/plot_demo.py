import os
os.chdir('../')
import glob
import argparse, torch,cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from utils2 import *
import warnings
warnings.filterwarnings('ignore')

# from CDYHY.MY_NET.MY_NET_V5 import MY_NET_V5 as our_model
# from CDYHY.MY_NET_VGG.MY_NET_resnet18 import MY_NET_resnet18_nopre as our_model
# from CDYHY.MY_NET_VGG.MY_NET_resnet18_nopre_APP import MY_NET_resnet18_APP as our_model
# from CDYHY.MY_NET.MY_NET_G4_3 import MY_NET as our_model

# from CDYHY.MY_NET.MY_NET_G4_P import MY_NET as our_model
# from CDYHY.Comparsion_Experiment.UNet import UNet as our_model
# from CDYHY.Comparsion_Experiment.FCN_8s import FCN8s as our_model
# from CDYHY.Comparsion_Experiment.FC_CONC import SiamUnet_conc as our_model
# from CDYHY.Comparsion_Experiment.FC_DIFF import FC_Siam_diff as our_model
# from CDYHY.Comparsion_Experiment.FC_EF import FC_EF as our_model
# from CDYHY.Comparsion_Experiment.TCDnet import our_model as our_model
# from CDYHY.Comparsion_Experiment.MFGANnet import MFGANnet as our_model
# from networks import BASE_Transformer as our_model
from CDYHY.Comparsion_Experiment.TFI_GR import BaseNet as our_model

def predict_on_image(model):
    # pre-processing on image
    to_tensor = transforms.ToTensor()

    file_path = r'C:/YHY/CDYHY/Comparsion_Experiment/pre_img'
    # file_path = r'C:/YHY/CDYHY/MY_NET/pre_img'
    # file_path = r'C:/YHY/CDYHY/BIT/pre_img'

    dir = str("TFI_GR_data6")

    if not os.path.exists(f'{file_path}/{dir}'):
        os.makedirs(f'{file_path}/{dir}')

    val_path_img1 = os.path.join(r'C:\YHY\CDYHY\data_6', 'pre_1\img1')
    val_path_img2 = os.path.join(r'C:\YHY\CDYHY\data_6', 'pre_1\img2')

    img1_list = glob.glob(os.path.join(val_path_img1, '*.jpg'))  # glob.glob()返回一个某一种文件夹下面的某一类型文件路径列表
    img2_list = glob.glob(os.path.join(val_path_img2, '*.jpg'))

    # val_path_img1 = os.path.join(r'C:\YHY\CDYHY\DATA\LEVIR_YYtest', 'pre\img1')
    # val_path_img2 = os.path.join(r'C:\YHY\CDYHY\DATA\LEVIR_YYtest', 'pre\img2')

    # img1_list = glob.glob(os.path.join(val_path_img1, '*.png'))  # glob.glob()返回一个某一种文件夹下面的某一类型文件路径列表
    # img2_list = glob.glob(os.path.join(val_path_img2, '*.png'))


    for i in range(len(img1_list)):

        img1 = Image.open(img1_list[i])
        img2 = Image.open(img2_list[i])

        img1 = to_tensor(img1).float().unsqueeze(0)
        img2 = to_tensor(img2).float().unsqueeze(0)

        # read csv label path
        label_info = get_label_info()
        # predict
        model.eval()
        predict = model(img1, img2).squeeze()
        predict = reverse_one_hot(predict)
        predict = colour_code_segmentation(np.array(predict), label_info)
        # predict = Image.fromarray(np.uint8(predict))
        # predict.save("123.png")
        predict = cv2.resize(np.uint8(predict), (512, 512))
        cv2.imwrite(f'{file_path}/{dir}/'+"{:}.png".format(img1_list[i].split('\\')[-1]), cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))

        # cv2.imwrite(f'{save_root}/'+'{:}'.format(i)+'*.png', cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    print('load model  ...')

    model = our_model()
    model_path = r'C:\YHY\CDYHY\max_epoch_48_miou_0.831404.pth'

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
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)
            return model
        model = load_GPUS(model, model_path=model_path)
        print("multi GPUs!")

    predict_on_image(model)
    print('Done!')
