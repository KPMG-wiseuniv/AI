import os
import numpy as np
import torch
from xml.etree.ElementTree import parse
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import matplotlib.pyplot as plt

train_list = ['background', 'bottle', 'chair', 'table', 'desk', 'shelves', 'curtain', 'bed', 'lamp', 'rug', 'potted-plant', 'sofa', 'tv']
n_list = len(train_list)
list_name_to_n = dict(zip(train_list,range(len(train_list))))


def read_file(root):
    f = open(root, 'r')
    data_list = f.readlines()
    for i in range(len(data_list)):
        data_list[i] = data_list[i][:-1]
    return data_list

# def Augmentation(image, flip, flop, scale, rotate) #Transforms as T 이거 쓰던데... 걍 imgaug 쓸까...
#     T.Compose


class Furniture_Segmentation(Dataset):
    def __init__(self, root, resize=None, siwoo = True): #root = VOC2012, list_root = VOC2012/Furniture/train.txt
        self. root = root # /home/siwoo/Desktop/kpmg_image/Furniture/
        self.img_root = os.path.join(root, 'IMAGE')
        self.mask_root = os.path.join(root, 'MASK')
        self.data_list = os.listdir(self.img_root)

        self.siwoo = siwoo
        self.resize = resize


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_root, self.data_list[idx]))
        h, w, c = np.array(img).shape
        if self.resize != None:
            img = img.resize(self.resize)
        img = np.array(img)/255
        img = np.transpose(img, (2, 0, 1))
        img = torch.as_tensor(img, dtype=torch.float32)


        mask = Image.open(os.path.join(self.mask_root, self.data_list[idx][:-4]+'.png'))

        if self.resize != None:
            mask = mask.resize(self.resize, Image.NEAREST)

        mask = np.array(mask)
        # print(np.unique(mask))
        # print(self.data_list[idx])
        # plt.imshow(mask)
        # plt.show()

        if self.siwoo != True:
            mask = np.expand_dims(mask, 2)
            mask = np.transpose(mask, (2, 0, 1))

        mask = torch.as_tensor(mask, dtype=torch.long)

        return img, mask





class Furniture_Detection(Dataset):
    def __init__(self, root, list_root, resize=None): #root = VOC2012, list_root = VOC2012/Furniture/train.txt
        self. root = root
        self.data_list = read_file(list_root)
        self.img_root = os.path.join(root, 'JPEGImages')
        self.annotation_root = os.path.join(root, 'Annotations')

        self.resize = resize


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_root, self.data_list[idx]+'.jpg'))
        h, w, c = np.array(img).shape
        if self.resize != None:
            img = img.resize(self.resize)
        img = np.array(img)/255
        img = np.transpose(img, (2, 0, 1))
        img = torch.as_tensor(img, dtype=torch.float32)

        label = parse(os.path.join(self.annotation_root, self.data_list[idx]+'.xml'))
        label = label.getroot()
        object = label.findall('object')

        labels = []
        boxes = []
        for obj in object:
            name = obj.findtext('name')
            if name in train_list:
                if self.resize != None:
                    xmin = int(obj.find('bndbox').findtext('xmin')) * self.resize[0]/w
                    ymin = int(obj.find('bndbox').findtext('ymin')) * self.resize[1]/h
                    xmax = int(obj.find('bndbox').findtext('xmax')) * self.resize[0]/w
                    ymax = int(obj.find('bndbox').findtext('ymax')) * self.resize[1]/h
                else:
                    xmin = int(obj.find('bndbox').findtext('xmin'))
                    ymin = int(obj.find('bndbox').findtext('ymin'))
                    xmax = int(obj.find('bndbox').findtext('xmax'))
                    ymax = int(obj.find('bndbox').findtext('ymax'))

                labels.append(list_name_to_n[name])
                boxes.append(np.round(np.array([xmin, ymin, xmax, ymax])))


        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {'labels': labels, 'boxes': boxes, 'area': area}

        return img, target

if __name__ == '__main__':
    None