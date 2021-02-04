import torch
import numpy as np
import segmentation_models_pytorch as smp
import cv2
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from VOC_DataLoader import Furniture_Segmentation
# from segmentation_dataloader import VOCSegmentation
if __name__ == '__main__':
    local_path = './home/siwoo/Desktop/kpmg_image/VOC2012'
    bs = 4  # batch size
    # dst = VOCSegmentation()  #
    # trainloader = data.DataLoader(dst, batch_size=bs)

    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['background', 'bottle', 'chair', 'table', 'desk', 'shelves', 'curtain', 'bed', 'lamp', 'rug', 'potted-plant', 'sofa', 'tv']
    ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'

    # create segmentation model with pretrained encoder
    '''model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION)'''
    model = smp.UNET(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    train_dataset =Furniture_Segmentation(root = '/home/siwoo/Desktop/kpmg_image/Furniture/', resize=(320, 320), siwoo=False)
    # valid_dataset = VOCSegmentation(mode='val')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    # train model for 40 epochs

    max_score = 0

    for i in range(0, 40):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)

        # do something (save model, change lr, etc.)

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
torch.save(model.state_dict(), '/home/siwoo/Desktop/kpmg_image/Furniture/segmentation_model_version2.pth')

