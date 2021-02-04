# import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import h5py

# data path
path_to_depth = '../NYU_V2/nyu_depth_v2_labeled.mat'
# read mat file
f = h5py.File(path_to_depth)

# # read 0-th image. original format is [3 x 640 x 480], uint8
# img = f['images'][0]
#
# # reshape
# img_ = np.empty([480, 640, 3])
# img_[:,:,0] = img[0,:,:].T
# img_[:,:,1] = img[1,:,:].T
# img_[:,:,2] = img[2,:,:].T
#
# # imshow
# img__ = img_.astype('float32')
# plt.imshow(img__/255.0)
# plt.show()
#
#
# # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
# depth = f['depths'][0]
#
# # reshape for imshow
# depth_ = np.empty([480, 640, 3])
# depth_[:,:,0] = depth[:,:].T
# depth_[:,:,1] = depth[:,:].T
# depth_[:,:,2] = depth[:,:].T
#
# plt.imshow(depth_/4.0)
# plt.show()

label = f['labels'][0]
print(label.shape)
label_ = np.empty([480, 640, 3])
label_[:,:,0] = label[0,:,:].T
label_[:,:,1] = label[1,:,:].T
label_[:,:,2] = label[2,:,:].T

plt.imshow(label_)
plt.show()