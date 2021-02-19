import os
import numpy as np
import torch
import csv
import random
import torchvision
from PIL import Image
from torch.utils.data import Dataset

def find_index(data, target):
  res = []
  lis = data
  while True:
    try:
      res.append(lis.index(target) + (res[-1]+1 if len(res)!=0 else 0))
      lis = data[res[-1]+1:]
    except:
      break
  return res

class FR_Dataset(Dataset):
    def __init__(self, img_root, label_root, type, resize=None): #root = VOC2012, list_root = VOC2012/Furniture/train.txt

        with open(label_root) as f:
            reader = csv.reader(f)
            labels = []
            interiors = []
            img_name = os.listdir(img_root)
            for txt in reader:
                if txt[1] in img_name:
                    labels.append(txt)
                    interiors.append(txt[6])

        num_modern = interiors.count('modern')
        num_natural = interiors.count('natural')

        if num_modern>num_natural:
            major = num_modern
            minor = num_natural
            minor_interior = 'natural'
        else:
            major = num_natural
            minor = num_modern
            minor_interior = 'modern'

        diff = major-minor
        minor_index = find_index(interiors, minor_interior)

        for i in range(diff):
            add_index = random.choice(minor_index)
            labels.append(labels[add_index])
            interiors.append(interiors[add_index])

        self.img_root = img_root
        self.label_root = label_root
        self.labels = labels
        self.resize = resize
        self.type = type


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_name = label[1]
        interior = label[6]
        color = int(float(label[3]))
        category = label[4]

        if interior == 'modern':
            interior = 0
        else:
            interior = 1

        if self.type == 'chair':
            if category == 'bar chair':
                FR_design = 1
                FR_function = 0
            elif category == 'cafe chair':
                FR_design = 2
                FR_function = 0
            elif category == 'desk chair':
                FR_design = 0
                FR_function = 1
            elif category == 'arm chair' or category == 'armchair':
                FR_design = 0
                FR_function = 2
            elif category == 'stem stool':
                FR_design = 0
                FR_function = 3
        elif self.type == 'table':
            if category == 'bar table':
                FR_design = 1
                FR_function = 0
            elif category == 'cafe table':
                FR_design = 2
                FR_function = 0
            elif category == 'console table':
                FR_design = 0
                FR_function = 1
            elif category == 'beside table':
                FR_design = 0
                FR_function = 2
        total_label = [interior, color, FR_design, FR_function]
        total_label = torch.as_tensor(total_label)

        img = Image.open(os.path.join(self.img_root, img_name)).convert('RGB')
        if self.resize != None:
            img = img.resize(self.resize)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(15),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.RandomVerticalFlip(0.5),
            torchvision.transforms.ToTensor()
        ])

        img = transforms(img)
        return img, total_label

if __name__ == '__main__':
    img_root = '/home/siwoo/Desktop/kpmg_image/Kaggle/table_inpainting_image/output'
    label_root = '/home/siwoo/Desktop/kpmg_image/Kaggle/table_labels.csv'
    dataset = FR_Dataset(img_root, label_root, 'table')