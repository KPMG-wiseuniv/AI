import argparse
import os
import torch
import torch.nn as nn
from model import PConvUNet
from loss import *
from train import Trainer
from train_vgg16 import train_vgg16
from inpainting_DataLoader import *
from torch.utils.data import DataLoader

from unet import UNet

parser = argparse.ArgumentParser()

parser.add_argument("--img_root", default='/home/siwoo/Desktop/kpmg_image/empty_room/image', type=str)
parser.add_argument("--mask_root", default='/home/siwoo/Desktop/kpmg_image/empty_room/mask', type=str)
parser.add_argument("--label_root", default='/home/siwoo/Desktop/kpmg_image/empty_room/label', type=str)
parser.add_argument("--vgg16_root", default='/home/siwoo/Desktop/kpmg_image/empty_room/model/vgg16.pth', type=str)

parser.add_argument("--vgg16_num_epoch", default=50, type=int)
parser.add_argument("--vgg16_batch_size", default=20, type=int)
parser.add_argument("--vgg16_lr", default=0.001, type=float)
parser.add_argument("--vgg16_weight_decay", default = 1e-2, type= int)

parser.add_argument("--num_classes", default=1, type=int)
parser.add_argument("--batch_size", default=20, type=int)
parser.add_argument("--num_epoch", default=50, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--weight_decay", default = 1e-2, type= int)

parser.add_argument("--start_iter", default=0, type=int)


parser.add_argument("--mask_out_dir", default = '/home/siwoo/Desktop/kpmg_image/empty_room/mask', type= str)
parser.add_argument("--model_out_dir", default= '/home/siwoo/Desktop/kpmg_image/empty_room/model', type= str)
parser.add_argument("--start_number", default= 0, type=int)

parser.add_argument("--train_vgg16", default= True, type=bool)
parser.add_argument("--make_mask", default= True, type=bool)
parser.add_argument("--train", default= True, type=bool)



args = parser.parse_args()

if args.train_vgg16 == True:
    print('start training vgg16...')
    train_vgg16(args)
    print('done!')

if args.make_mask == True:
    print('make mask...')
    make_traindataset(args.img_root, args.mask_root, args.label_root)
    print('done!')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = Inpainting_Dataset(args.img_root, args.mask_root, args.label_root)
dataloader = DataLoader(dataset)

# model = UNet(3,3)
model = PConvUNet(finetune=False, layer_size=7)
model.to(device)

criterion = InpaintingLoss(VGG16FeatureExtractor())
optimizer = torch.optim.Adam(filter(lambda  p: p.required_grad, model.parameters()),
                                    lr = args.lr, weight_decay = args.weight_decay)

trainer = Trainer(args.strat_iter, args, device, model, dataloader, dataloader, criterion, optimizer)
trainer.iterate()

# if args.train == True:
#     optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
#     criterion = nn.BCEWithLogitsLoss() # chage -> cross entropy or paper's loss
#     for epoch in range(args.num_epoch):
#         total_loss = 0
#         for pack, (img, mask, label) in enumerate(dataloader):
#             img = img.to('cuda')
#             label = label.to('cuda')
#
#             pred = model(img)
#             loss = criterion(pred, label)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss +=loss.item()
#
#         print('epoch: {} loss: {}'.format(epoch, total_loss/len(dataloader)))
#         torch.save(model.state_dict(), os.path.join(args.model_out_dir,str(epoch+1+args.start_number)+'.pth'))
#
#
# if args.train == False:
#     with torch.no_grad():
#         model.eval()
#         model.load_state_dict(torch.load(os.path.join(args.model_out_dir,str(50)+'.pth')))
#         make_mask(dataloader, model, args.label_root, args.mask_out_dir)