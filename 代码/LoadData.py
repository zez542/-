import glob,os,torch
from PIL import Image
from utils import *
import numpy as np
from torchvision import transforms
import cv2

class Change_Detect(torch.utils.data.Dataset):
    """
    对img1,img2,label进行处理
    img1和img2->tensor,
    label->one-hot->（n_classes,H,W）
    """
    def __init__(self,img1_path,img2_path,label_path,csv_path,mode='train'):
        super().__init__()
        self.mode=mode

        """BICD"""
        # self.img1_list  = glob.glob(os.path.join(img1_path,'*.jpg'))#glob.glob()返回一个某一种文件夹下面的某一类型文件路径列表
        # self.img2_list  = glob.glob(os.path.join(img2_path,'*.jpg'))
        # self.label_list = glob.glob(os.path.join(label_path,'*.png'))
        """"LEVIR-CD"" """"BCDD"""
        self.img1_list = glob.glob(os.path.join(img1_path,   '*.png'))  # glob.glob()返回一个某一种文件夹下面的某一类型文件路径列表
        self.img2_list = glob.glob(os.path.join(img2_path,   '*.png'))
        self.label_list = glob.glob(os.path.join(label_path, '*.png'))

        self.label_info = get_label_info(csv_path)
        self.to_tensor  = transforms.ToTensor()

    def __getitem__(self, index):
        img1  = Image.open(self.img1_list[index])
        img1  = self.to_tensor(img1).float() #torch.Size([3, 512, 512])
        # print(img1.shape)
        #----------------------------------------------------------------#
        # lenna1=cv2.imread(self.img1_list[index],0)
        # lenna1= cv2.GaussianBlur(lenna1, (3, 3), 0)
        # lenna1 = cv2.Canny(lenna1, 120, 150)
        # lenna1=self.to_tensor(lenna1).float()
        # img1  =torch.cat((img1,lenna1),0)
        #----------------------------------------------------------------#
        img2  = Image.open(self.img2_list[index])
        img2  = self.to_tensor(img2).float()
        # print(img2.shape)
        #----------------------------------------------------------------#
        # lenna2 = cv2.imread(self.img1_list[index], 0)
        # lenna2 = cv2.GaussianBlur(lenna2, (3, 3), 0)
        # lenna2 = cv2.Canny(lenna2, 120, 150)
        # lenna2 = self.to_tensor(lenna2).float()
        # img2 = torch.cat((img2, lenna2), 0)
        #----------------------------------------------------------------#
        label = Image.open(self.label_list[index])
        label = np.array(label)
        if len(label.shape) == 2:  # 如果是单通道灰度图像，添加一个维度表示通道数
              label = np.expand_dims(label, axis=2)
        # label = torch.from_numpy(label).long()  # 转换为 PyTorch 张量
        # shape = label.shape
        # print('shape',shape)
        # print('y',np.array(label).shape)#(512,512,3)
        label = one_hot_it(label,self.label_info).astype(np.uint8)#torch.Size([512, 512, 2]
        # print('x1',label.shape)
        label = np.transpose(label,[2,0,1]).astype(np.float32)
        # print('x2', label.shape)
        label = torch.from_numpy(label)#torch.from_numpy()用来将数组array转换为张量Tensor(2,512,512)
        # print('x3',label.shape)

        return img1,img2,label

    def __len__(self):
        # print(len(self.img1_list))
        return len(self.img1_list)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data = Change_Detect(img1_path =r"C:\Users\Liyan\Desktop\CD\Public_Dataset\augmented_BCDD\train\img1",
                         img2_path =r"C:\Users\Liyan\Desktop\CD\Public_Dataset\augmented_BCDD\train\img2",
                         label_path=r"C:\Users\Liyan\Desktop\CD\Public_Dataset\augmented_BCDD\train\label",
                         csv_path  =r"C:\Users\Liyan\Desktop\CD\Public_Dataset\augmented_BCDD\class_dict.csv",
                        )

    # data = Change_Detect(img1_path="C:/YHY/CDYHY/data_6/train/img1",
    #                      img2_path="C:/YHY/CDYHY/data_6/train/img2",
    #                      label_path="C:/YHY/CDYHY/data_6/train/label",
    #                      csv_path="C:/YHY/CDYHY/data_6/class_dict.csv",
    #                      )

    # data = Change_Detect(img1_path="C:/YHY/CDYHY/CDD/train/img1",
    #                      img2_path="C:/YHY/CDYHY/CDD/train/img2",
    #                      label_path="C:/YHY/CDYHY/CDD/train/label",
    #                      csv_path="C:/YHY/CDYHY/CDD/class_dict.csv",
    #                      )

    dataloader_test = DataLoader(data,
                                 batch_size=2,
                                 shuffle=True,
                                 num_workers=0,
                                )

    for _,(img1,img2,label) in enumerate(dataloader_test):
        print('y')
        print('img1',img1)
        print(img1.shape)
        print('img2',img2)
        print('label',label)
        print('y')
        if _ == 0:
            break


























