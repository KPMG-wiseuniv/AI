import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset


class Inpainting_Dataset(Dataset):
    def __init__(self, image_root, mask_root, label_root, using_label = True):
        image_name = os.listdir(image_root)
        image_name.sort()

        self.image_root = image_root
        self.mask_root = mask_root
        self.label_root = label_root
        self.image_name = image_name
        self.using_label = using_label

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_root, self.image_name[idx])).convert('RGB')
        img = img.resize((256, 256))
        img = np.array(img)/255
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        if self.using_label == True:
            mask = Image.open(os.path.join(self.mask_root, self.image_name[idx]))
            mask = mask.resize((256, 256), Image.NEAREST)
            mask = np.array(mask)/255
            mask = np.transpose(mask, (2, 0, 1))

            mask = torch.from_numpy(mask).float()

            label = Image.open(os.path.join(self.label_root, self.image_name[idx]))
            label = label.resize((256, 256))
            label = np.array(label) / 255
            label = np.transpose(label, (2, 0, 1))

            label = torch.from_numpy(label).float()

            return label, mask, img

        return img

def make_mask(dataloader, model, label_root, out_dir):
    labels = os.listdir(label_root)
    labels.sort()
    for iter, (img, label) in enumerate(dataloader):
        pred = model(img.to('cuda'))
        pred = pred.detach().cpu().numpy()
        pred = np.squeeze(pred)
        pred = np.transpose(pred,(1, 2, 0))
        pred = pred.astype(np.uint8)

        mask = Image.fromarray(pred)
        mask.save(os.path.join(out_dir, labels[iter]))

def make_label(image_name,image_dir, mask_dir, label_dir, w=None, h=None):
    img = Image.open(os.path.join(image_dir, image_name)).convert('RGB')
    img = np.array(img)
    if w == None and h == None:
        h, w, c = np.array(img).shape

    x = []
    x.append(random.randrange(0, w))
    x.append(random.randrange(0, w))

    y = []
    y.append(random.randrange(0,h))
    y.append(random.randrange(0,h))

    x.sort()
    y.sort()

    mask = np.zeros((h, w, c), dtype=np.uint8)+1
    mask[x[0]:x[1], y[0]:y[1], :]=0
    label=img*mask

    mask = Image.fromarray(mask*255)
    mask.save(os.path.join(mask_dir, image_name))

    label = Image.fromarray(label)
    label.save(os.path.join(label_dir, image_name))

def make_traindataset(image_path, mask_path, label_path):
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    image_name = os.listdir(image_path)
    for i in range(len(image_name)):
        make_label(image_name[i], image_path, mask_path, label_path)

if __name__ == '__main__':
    image_path = '/home/siwoo/Desktop/kpmg_image/empty_room/image'
    mask_path = '/home/siwoo/Desktop/kpmg_image/empty_room/mask'
    label_path = '/home/siwoo/Desktop/kpmg_image/empty_room/label'
    make_traindataset(image_path, mask_path, label_path)