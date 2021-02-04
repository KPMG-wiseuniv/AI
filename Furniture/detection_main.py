import torch
import os
import argparse
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from VOC_DataLoader import Furniture_Detection
from torch.utils.data import DataLoader


def collate_fn(batch):
    return tuple(zip(*batch))

parser = argparse.ArgumentParser()

parser.add_argument("--root", default='/home/siwoo/Desktop/kpmg_image/Furniture_VOC/VOC2012', type=str)
parser.add_argument("--train_list_root", default='/home/siwoo/Desktop/kpmg_image/Furniture_VOC/VOC2012/Furniture/train.txt', type=str)
parser.add_argument("--val_list_root", default='/home/siwoo/Desktop/kpmg_image/Furniture_VOC/VOC2012/Furniture/val.txt', type=str)
parser.add_argument("--num_epochs", default='10', type=int)
parser.add_argument("--resize_w", default='320', type=int)
parser.add_argument("--resize_h", default='320', type=int)



parser.add_argument("--num_classes", default='2', type=int)
parser.add_argument("--batch_size", default='8', type=int)
parser.add_argument("--k", default='10', type=int, help='cross validation')
parser.add_argument("--resize", default=(320, 320), type=tuple, help='cross validation')

# parser.add_argument("--k", default='10', type=int, help='cross validation')

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=6)
# model.load_state_dict(torch.load('/home/siwoo/Desktop/Furiture_VOC/VOC2012/detection_model10.pth'))

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)

train_dataset = Furniture_Detection(args.root, args.train_list_root, (args.resize_w, args.resize_h))
val_dataset = Furniture_Detection(args.root, args.val_list_root)

train_dataloader = DataLoader(train_dataset, batch_size= args.batch_size, drop_last=True, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, drop_last=False, batch_size=1, shuffle=False, collate_fn=collate_fn)

# for epoch in range(args.num_epochs):
#     # for iter, pack in enumerate(train_dataloader):
#     #     print(pack)
#     #     img = pack['img']
#     #     target = pack['target']
#     #     output = model(img, target)
#
#     step = 0
#     global_step = int(len(train_dataset)/args.batch_size)+1
#     for img, target in train_dataloader:
#         images = list(image.to(device) for image in img)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in target]
#
#         loss_dict=model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#
#         step+=1
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()
#
#         print('epoch: {}/{}, step: {}/{}, loss: {}'.format(epoch, args.num_epochs, step, global_step, losses))
torch.save(model.state_dict(), '/home/siwoo/Desktop/kpmg_image/Furniture_VOC/detection_model20.pth')

with torch.no_grad():
    model.eval()

    model.load_state_dict(torch.load('/home/siwoo/Desktop/kpmg_image/Furniture_VOC/detection_model20.pth'))
    for img, target in val_dataloader:
        images = list(image.to(device) for image in img)
        targets = [{k: v.to(device) for k, v in t.items()} for t in target]

        output=model(images)
        print(output)
        print(targets)
        plt.imshow(np.squeeze(images[0].detach().cpu().numpy()).transpose(1, 2, 0))
        plt.show()