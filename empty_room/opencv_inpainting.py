import cv2
import numpy as np
import matplotlib.pyplot as plt
input_path='/home/siwoo/Desktop/kpmg_image//'
output_path='/home/siwoo/Desktop/kpmg_image/inpainting/'

import os
file_list = os.listdir('/home/siwoo/Desktop/kpmg_image/empty_room/')

for i,file in enumerate(file_list):
    img=cv2.imread(input_path+'image/'+file)
    plt.imshow(img[:,:,::-1],'gray')
    plt.axis('off')
    plt.savefig(output_path+'image/'+file,bbox_inches='tight')
    noise=cv2.imread(input_path+'mask/'+file,0)
    noise=cv2.resize(noise,(img.shape[1],img.shape[0]))
    print(noise.shape==img.shape)
    plt.imshow(noise,'gray')
    plt.axis('off')
    plt.savefig(output_path+'mask/'+file,bbox_inches='tight')
    dst=cv2.inpaint(img,noise,3,cv2.INPAINT_TELEA)
    plt.imshow(dst[:,:,::-1])
    plt.axis('off')
    plt.savefig(output_path+'output/'+file,bbox_inches='tight')

