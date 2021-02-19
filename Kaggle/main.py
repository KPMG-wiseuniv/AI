import torch
import os
import argparse
from PIL import Image
from FR_model import mobilenet_v3_small
from FR_Dataset import FR_Dataset
from torch.utils.data import DataLoader
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--chair_img_root", default='/home/siwoo/Desktop/kpmg_image/Kaggle/chair_inpainting_image/output', type=str)
parser.add_argument("--chair_label_root", default='/home/siwoo/Desktop/kpmg_image/Kaggle/chair_labels.csv', type=str)
parser.add_argument("--table_img_root", default='/home/siwoo/Desktop/kpmg_image/Kaggle/table_inpainting_image/output', type=str)
parser.add_argument("--table_label_root", default='/home/siwoo/Desktop/kpmg_image/Kaggle/table_labels.csv', type=str)
parser.add_argument("--type", default='chair', type=str)


parser.add_argument("--num_interior_classes", default='2', type=int)
parser.add_argument("--num_color_classes", default='6', type=int)
parser.add_argument("--num_design_classes", default='3', type=int)
parser.add_argument("--num_chair_function_classes", default='4', type=int)
parser.add_argument("--num_table_function_classes", default='3', type=int)

parser.add_argument("--num_epochs", default='200', type=int)
parser.add_argument("--lr", default='0.01', type=float)
parser.add_argument("--batch_size", default='4', type=int)

parser.add_argument("--resize_w", default='320', type=int)
parser.add_argument("--resize_h", default='320', type=int)

parser.add_argument("--model_load", default=0, type=int)
parser.add_argument("--train", default=True, type=bool)

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if args.type == 'chair':
    model = mobilenet_v3_small(num_classes = [args.num_interior_classes, args.num_color_classes, args.num_design_classes, args.num_chair_function_classes])
    img_root = args.chair_img_root
    label_root = args.chair_label_root
if args.type == 'table':
    model = mobilenet_v3_small(num_classes = [args.num_interior_classes, args.num_color_classes, args.num_design_classes, args.num_table_function_classes])
    img_root = args.table_img_root
    label_root = args.table_label_root

if args.model_load != 0:
    model.load_state_dict(torch.load('/home/siwoo/Desktop/kpmg_image/Kaggle/model_'+args.type+'/'+str(args.model_load)+'.pth'))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
criterion = torch.nn.CrossEntropyLoss()

train_dataset = FR_Dataset(img_root, label_root, args.type)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)

if args.train == True:
    for epoch in range(args.num_epochs):
        mean_loss = np.array([0, 0, 0, 0])
        for iter, (img, total_label) in enumerate(train_dataloader):
            img = img.to(device)
            total_label = total_label.to(device)


            output = model(img)
            losses =[]
            for i in range(4):
                loss = criterion(output[i], total_label[:, i])
                losses.append(loss)
                mean_loss[i] +=loss.item()

            total_loss = sum(losses)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


        print('epoch: {} / interior, color, design, function loss: {}'.format(epoch, mean_loss/len(train_dataset)))
        torch.save(model.state_dict(), '/home/siwoo/Desktop/kpmg_image/Kaggle/model_'+args.type+'/'+str(epoch+1+args.model_load)+'.pth')
if args.train == False:
    model.load_state_dict(torch.load('/home/siwoo/Desktop/kpmg_image/Kaggle/model_'+args.type+'/'+str(args.model_load)+'.pth'))
    img = Image.open('/home/siwoo/Downloads/test_image.jpg')
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)/255
    img = torch.as_tensor(img, dtype=torch.float32)
    img = img.to(device)

    output = model(img)
    interior = output[0].detach().cpu().numpy()
    color = output[1].detach().cpu().numpy()
    output_design = output[2].detach().cpu().numpy()
    output_function = output[3].detach().cpu().numpy()

    print('interior: {}, color: {}, design: {}, function: {}'.format(interior, color, output_design, output_function))