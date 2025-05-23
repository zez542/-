import pandas as pd
import numpy as np
import torch


def get_label_info(csv_path=r'C:\Users\Liyan\Desktop\CD\Public_Dataset\data1\class_dict.csv'):
    data = pd.read_csv(csv_path)
    label = {}
    for _, row in data.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r), int(g), int(b)]
    return label


def one_hot_it(label, label_info=get_label_info()):
    # return semantic_map -> [H, W, num_classes]
    semantic_map = []
    for info in label_info:
        color = label_info[info]
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def reverse_one_hot(image):
	image = image.permute(1, 2, 0)  # [2, 512, 512] ==> [512, 512, 2]
	x = torch.argmax(image, dim=-1) # [512, 512, 2] ==> [512, 512]
	return x


def colour_code_segmentation(image, label_values):
	label_values = [label_values[key] for key in label_values]
	colour_codes = np.array(label_values)
	x = colour_codes[image.astype(int)]
	return x


if __name__ == '__main__':
    # from PIL import Image
    # img1 = Image.open('ceshi_black_white.png')
    # img2 = Image.open('ceshi_black.png')
    # out = one_hot_it(img2).astype(np.uint8)
    # out = np.transpose(out, [2, 0, 1]).astype(np.float32)
    # print(out)

    x = torch.rand([2, 5, 5])
    out = reverse_one_hot(x)
    print(out.flatten())

