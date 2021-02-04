import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image

def make_img(root):
    os.makedirs('PNGImages', exist_ok=True)
    os.makedirs('labels', exist_ok=True)
    file = h5py.File(root)
    for i in range(1448):
        img = file['images'][i]
        label = file['labels'][i]

        img_ = np.empty([480, 640, 3])
        img_[:,:,0] = img[0,:,:].T
        img_[:,:,1] = img[1,:,:].T
        img_[:,:,2] = img[2,:,:].T

        label = label.T

        img_ = img_.astype(np.uint8)
        label = label.astype(np.uint8)

        image = Image.fromarray(img_)
        label = Image.fromarray(label)

        image.save(os.path.join('PNGImages', str(i)+'.png'))
        label.save(os.path.join('labels', str(i)+'.png'))

def ploting(i, path):
    # path = '/home/siwoo/Desktop/kpmg_image/NYU_V2/labels'
    # path = '/home/siwoo/Desktop/kpmg_image/Furniture_VOC/VOC2012/SegmentationClass'
    img_list = os.listdir(path)
    img_list.sort()
    img = Image.open(os.path.join(path, img_list[i]))
    img = np.array(img)

    print(img_list[i])
    print(img.shape)
    print(np.unique(img))

    plt.imshow(img)
    plt.show()

def make_detection_data_list(label, train): #label: [bottle, chair, diningtable, pottedplant, sofa, tvmonitor], train: train or val or trainval
    data_list = []
    ImageSets = os.listdir('/home/siwoo/Desktop/Furiture_VOC/VOC2012/ImageSets/Main')
    for i in range(len(ImageSets)):
        file_list = ImageSets[i].split('_')
        if file_list[0] in label:
            if train == file_list[1][:-4]:
                f = open(os.path.join('/home/siwoo/Desktop/Furiture_VOC/VOC2012/ImageSets/Main/', ImageSets[i]), 'r')
                lines = f.readlines()
                for j in range(len(lines)):
                    if int(lines[j][-3:]) == 1:
                        if lines[j][:11] not in data_list:
                            data_list.append(lines[j][:11])
                f.close()
    data_list.sort()
    f = open(os.path.join('/home/siwoo/Desktop/Furiture_VOC/VOC2012/Furniture', train+'.txt'), 'w')
    for i in range(len(data_list)):
        f.write(data_list[i])
        f.write('\n')
    f.close()


def make_segmentation_label(names_label, rooTypes_label, names_root, roomTypes_root, mask_root):
    file = h5py.File(os.path.join(mask_root, 'nyu_depth_v2_labeled.mat'))

    with open(names_root, 'r') as f:
        names = f.readlines()
        for i in range(len(names)):
            names[i] = names[i][1:-2]
        names.insert(0, 'background')
    with open(roomTypes_root, 'r') as f:
        roomTypes = f.readlines()
        for i in range(len(roomTypes)):
            roomTypes[i] = roomTypes[i][1:-2]

    names_to_num = dict(zip(names, range(len(names))))
    print(names_to_num)

    for iter in range(1449):
        if roomTypes[iter] in rooTypes_label:
            mask = file['labels'][iter]
            mask = mask.T

            img = file['images'][iter]

            img_ = np.empty([480, 640, 3])
            img_[:, :, 0] = img[0, :, :].T
            img_[:, :, 1] = img[1, :, :].T
            img_[:, :, 2] = img[2, :, :].T
            img_ = img_.astype(np.uint8)
            # print(np.unique(mask))

            new_mask = np.zeros_like(mask, dtype=np.uint8)
            for i in range(len(names_label)):
                new_mask[mask==names_to_num[names_label[i]]] = i+1

            # plt.imshow(img_)
            # plt.show()
            # plt.imshow(new_mask)
            # plt.show()

            img_ = Image.fromarray(img_)
            img_.save(os.path.join(mask_root, 'PNGImages', str(iter)+'.png'))

            seg_mask = Image.fromarray(new_mask)
            seg_mask.save(os.path.join(mask_root, 'mask', str(iter)+'.png'))

def make_segmentation_data_mask(label):
    mask_path = '/home/siwoo/Desktop/kpmg_image/Furniture/VOC2012/SegmentationClass'
    out_dir = '/home/siwoo/Desktop/kpmg_image/Furniture/VOC2012/new_SegmentationClass'

    mask_list = os.listdir(mask_path)

    CLASSES = ("background", "airplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorcycle", "person", "potted-plant", "sheep", "sofa", "train",
               "tv")

    CAT_NAME_TO_NUM = dict(zip(CLASSES, range(len(CLASSES))))

    for mask_name in mask_list:
        mask = Image.open(os.path.join(mask_path, mask_name))
        mask = np.array(mask)

        new_mask = np.zeros_like(mask, dtype=np.uint8)
        for i in range(len(label)):
            new_mask[mask==CAT_NAME_TO_NUM[label[i]]] = i+1

        if np.max(new_mask)!=0:
            seg_mask = Image.fromarray(new_mask)
            seg_mask.save(os.path.join(mask_path, out_dir, mask_name))

def combine_segmentation_mask(VOC_label, NYU_label, total_label):

    # VOC_label = ['bottle', 'chair', 'table', 'potted-plant', 'sofa', 'tv']
    # NYU_label = ['chair', 'table', 'desk', 'shelves', 'sofa', 'curtain', 'bed', 'lamp', 'rug']
    # total_label = ['bottle', 'chair', 'table', 'desk', 'shelves', 'curtain', 'bed', 'lamp', 'rug', 'potted-plant', 'sofa', 'tv']

    VOC_path = '/home/siwoo/Desktop/kpmg_image/Furniture/VOC2012/'
    NYU_path = '/home/siwoo/Desktop/kpmg_image/Furniture/NYU_V2/'
    out_dir = '/home/siwoo/Desktop/kpmg_image/Furniture/'

    VOC_image_name = os.listdir(os.path.join(VOC_path, 'new_SegmentationClass'))
    NYU_image_name = os.listdir(os.path.join(NYU_path, 'mask'))

    names_to_num = dict(zip(total_label, range(len(total_label))))

    for VOC_image in VOC_image_name:
        mask = Image.open(os.path.join(VOC_path, 'new_SegmentationClass', VOC_image))
        mask = np.array(mask)

        new_mask = np.zeros_like(mask, dtype=np.uint8)
        for i in range(len(VOC_label)):
            new_mask[mask == i+1] = names_to_num[VOC_label[i]]

        if np.max(new_mask) !=0:

            new_mask = new_mask.astype(np.uint8)


            new_mask = Image.fromarray(new_mask)
            new_mask.save(os.path.join(out_dir, 'MASK', VOC_image))

            shutil.copyfile(os.path.join(VOC_path, 'JPEGImages', VOC_image[:-4]+'.jpg'),
                            os.path.join(out_dir, 'IMAGE', VOC_image[:-4]+'.jpg'))

    for NYU_image in NYU_image_name:
        mask = Image.open(os.path.join(NYU_path, 'mask', NYU_image))
        mask = np.array(mask)

        new_mask = np.zeros_like(mask, dtype=np.uint8)
        for i in range(len(NYU_label)):
            new_mask[mask == i+1] = names_to_num[NYU_label[i]]

        if np.max(new_mask) != 0:
            new_mask = new_mask.astype(np.uint8)

            new_mask = Image.fromarray(new_mask)
            new_mask.save(os.path.join(out_dir, 'MASK', NYU_image))

            shutil.copyfile(os.path.join(NYU_path, 'PNGImages', NYU_image),
                            os.path.join(out_dir, 'IMAGE', NYU_image))

if __name__ == '__main__':
    # make_img('nyu_depth_v2_labeled.mat')
    # for i in range(1449):
    #     ploting(i)

#VOC segmentation data

    #detection
    # train = 'train'
    # val = 'val'
    # make_detection_data_list(label, train)
    # make_detection_data_list(label, val)

    #seg
    # label = ['bottle', 'chair', 'diningtable', 'potted-plant', 'sofa', 'tv']
    # make_segmentation_data_mask(label)

#NYU segmentation data
    # names_label = ['chair', 'table', 'desk', 'shelves', 'sofa', 'curtain', 'bed', 'lamp', 'rug']
    # roomTypes_label = ['laundry_room', 'dining_room', 'playroom', 'bedroom', 'home_office', 'living_room'] #'basement', 'home_storage', 'furniture_store'
    # names_root = '/home/siwoo/Desktop/kpmg_image/Furniture/NYU_V2/names'
    # roomTypes_root = '/home/siwoo/Desktop/kpmg_image/Furniture/NYU_V2/roomTypes'
    # mask_root = '/home/siwoo/Desktop/kpmg_image/Furniture/NYU_V2/'
    #
    # make_segmentation_label(names_label, roomTypes_label, names_root, roomTypes_root, mask_root)

#total segmentation data
    VOC_label = ['bottle', 'chair', 'table', 'potted-plant', 'sofa', 'tv']
    NYU_label = ['chair', 'table', 'desk', 'shelves', 'sofa', 'curtain', 'bed', 'lamp', 'rug']
    total_label = ['background', 'bottle', 'chair', 'table', 'desk', 'shelves', 'curtain', 'bed', 'lamp', 'rug', 'potted-plant', 'sofa', 'tv']

    combine_segmentation_mask(VOC_label, NYU_label, total_label)

    # NYU_V2_path = '/home/siwoo/Desktop/kpmg_image/Furniture/NYU_V2/mask'
    # VOC2012_path = '/home/siwoo/Desktop/kpmg_image/Furniture/VOC2012/new_SegmentationClass'
    # for i in range(len(os.listdir(VOC2012_path))):
    #     ploting(i, VOC2012_path)
