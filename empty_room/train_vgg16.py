import os
import torch
import torch.nn as nn
from tqdm import tqdm
from VGG16 import pdvgg16
from inpainting_DataLoader import *
from torch.utils.data import DataLoader

def train_vgg16(args, device):
    model = pdvgg16(args, pretrained=False, num_classes=args.vgg16_num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.vgg16_lr, weight_decay = args.vgg16_weight_decay)
    criterion = nn.CrossEntropyLoss() # chage -> cross entropy or paper's loss

    dataset = Inpainting_Dataset(args.img_root, args.mask_root, args.label_root, using_label=False)
    dataloader = DataLoader(dataset, batch_size=args.vgg16_batch_size, shuffle=True, drop_last=True)

    for epoch in tqdm(range(args.vgg16_num_epoch)):
        for pack, img in enumerate(dataloader):
            img = img.to(device)
            label = torch.ones(args.vgg16_batch_size, dtype=torch.long)
            label = label.to(device)

            pred = model(img)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), args.vgg16_root)