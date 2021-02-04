import torch
import os
import argparse
import tqdm
import torchvision
import numpy as np
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from VOC_DataLoader import Furniture_Segmentation
from torch.utils.data import DataLoader
import torch.nn.functional as F


def mIOU(label, pred, num_classes=13):
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)

def collate_fn(batch):
    return tuple(zip(*batch))

CLASSES = ['background', 'bottle', 'chair', 'table', 'desk', 'shelves', 'curtain', 'bed', 'lamp', 'rug', 'potted-plant', 'sofa', 'tv']

list_name_to_n = dict(zip(CLASSES ,range(len(CLASSES ))))
print(list_name_to_n)


parser = argparse.ArgumentParser()

parser.add_argument("--root", default='/home/siwoo/Desktop/kpmg_image/Furniture', type=str)
parser.add_argument("--num_epochs", default='50', type=int)
parser.add_argument("--resize_w", default='320', type=int)
parser.add_argument("--resize_h", default='320', type=int)

# parser.add_argument("--lr", default='0.000005', type=float)
parser.add_argument("--lr", default='0.0001', type=float)
parser.add_argument("--num_classes", default='13', type=int)
parser.add_argument("--batch_size", default='16', type=int)
parser.add_argument("--model_load", default=59, type=int)
parser.add_argument("--train", default=False, type=bool)
# parser.add_argument("--k", default='10', type=int, help='cross validation')

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=args.num_classes)
if args.model_load != 0:
    mdoel = model.load_state_dict(torch.load('/home/siwoo/Desktop/kpmg_image/Furniture/seg_model_siwoo/segmentation_model_pre_train'+str(args.model_load)+'.pth'))

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
criterion = torch.nn.CrossEntropyLoss()

train_dataset = Furniture_Segmentation(args.root, (args.resize_w, args.resize_h))
val_dataset = Furniture_Segmentation(args.root, (args.resize_w, args.resize_h))

train_dataloader = DataLoader(train_dataset, batch_size= args.batch_size, drop_last=True, shuffle=True)
val_dataloader = DataLoader(train_dataset, batch_size= 1, drop_last=False, shuffle=True)


if args.train == True:
    for epoch in range(args.num_epochs):
        m_iou = []
        losses = []
        # step = 0
        # global_step = int(len(train_dataset)/args.batch_size)+1
        for img, mask in train_dataloader:
            img = img.to(device)
            mask = mask.to(device)

            output = model(img)['out']

            loss = criterion(output, mask)

            iou = mIOU(mask.detach().cpu(), output.detach().cpu())

            losses.append(loss.item())
            m_iou.append(iou)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: {}/{}, loss: {}, iou: {}'.format(epoch+1, args.num_epochs, sum(losses)/int(len(train_dataset)/args.batch_size), sum(m_iou)/int(len(train_dataset)/args.batch_size)))
        torch.save(model.state_dict(), '/home/siwoo/Desktop/kpmg_image/Furniture/seg_model_siwoo/segmentation_model_pre_train'+str(args.model_load+epoch+1)+'.pth')

if args.train == False:
    with torch.no_grad():
        model.eval()

        model.load_state_dict(torch.load('/home/siwoo/Desktop/kpmg_image/Furniture/seg_model_siwoo/segmentation_model'+str(args.model_load)+'.pth'))
        for img, mask in val_dataloader:
            img = img.to(device)

            output=model(img)['out']

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(np.transpose(np.squeeze(img.detach().cpu().numpy()), (1, 2, 0)))
            ax1.set_title('original_image')
            plt.xticks([])
            plt.yticks([])

            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(np.squeeze(mask.numpy()))
            ax2.set_title('ground_truth')
            plt.xticks([])
            plt.yticks([])

            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(np.squeeze(np.argmax(output.detach().cpu().numpy(), 1)))
            ax3.set_title('model output')
            plt.xticks([])
            plt.yticks([])

            plt.show()

            # plt.imshow(np.squeeze(np.argmax(output.detach().cpu().numpy(), 1)))
            # plt.show()