import torch
import csv
import os
import torchvision
import numpy as np
from PIL import Image


def input_img(img_name):
    modern_path = '/home/siwoo/Desktop/kpmg_image/Kaggle/clf_image'
    img = Image.open(os.path.join(modern_path, img_name))
    img = np.transpose(img, (2, 0, 1)) / 255
    img = np.expand_dims(img, 0)
    img = torch.as_tensor(img, dtype=torch.float32)
    return img

CLASSES = ['background', 'bottle', 'chair', 'table', 'desk', 'shelves', 'curtain', 'bed', 'lamp', 'rug', 'potted-plant', 'sofa', 'tv']

list_name_to_n = dict(zip(CLASSES ,range(len(CLASSES ))))
print(list_name_to_n)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=13)

model.to(device)

modern_path = '/home/siwoo/Desktop/kpmg_image/Kaggle/clf_image'

with torch.no_grad():
    model.eval()
    model.load_state_dict(torch.load('/home/siwoo/Desktop/kpmg_image/Furniture/seg_model_siwoo/segmentation_model_pre_train39.pth'))

    with open('/home/siwoo/Desktop/kpmg_image/Kaggle/라벨링_modern_price.csv') as f:
        reader = csv.reader(f)
        category = ['sofa', 'chair', 'table', 'desk', 'shelves']
        labels = []
        for txt in reader:
            labels.append(txt)

        for label in labels[1:]:
            for i in range(5):
                if label[i*3+2] !='':
                    img = input_img(os.path.join('/home/siwoo/Desktop/kpmg_image/Kaggle/clf_image/', label[1]+'.jpg'))
                    img = img.to(device)
                    output = model(img)['out']
                    mask = np.squeeze(np.argmax(output.detach().cpu().numpy(), 1))
                    mask[mask!=11] = 0
                    mask[mask == 11] = 255
                    mask = mask.astype(np.uint8)
                    mask = Image.fromarray(mask)
                    mask.save('/home/siwoo/Desktop/kpmg_image/Kaggle/mask_modern/'+'modern_'+category[i]+'_'+label[1]+'.png')

    with open('/home/siwoo/Desktop/kpmg_image/Kaggle/라벨링_natural_color.csv') as f:
        reader = csv.reader(f)
        category = ['sofa', 'chair', 'table', 'desk', 'shelves']
        labels = []
        for txt in reader:
            print(txt)
            labels.append(txt)

        for label in labels[1:]:
            for i in range(5):
                if label[i*3+2] !='':
                    img = input_img(os.path.join('/home/siwoo/Desktop/kpmg_image/Kaggle/natural/', '[크기변환]natural ('+str(int(label[1])+1)+').jpg'))
                    img = img.to(device)
                    output = model(img)['out']
                    mask = np.squeeze(np.argmax(output.detach().cpu().numpy(), 1))
                    mask[mask!=11] = 0
                    mask[mask == 11] = 255
                    mask = mask.astype(np.uint8)
                    mask = Image.fromarray(mask)
                    mask.save('/home/siwoo/Desktop/kpmg_image/Kaggle/mask_natural/'+'natural_'+category[i]+'_'+label[1]+'.png')