from src import utils 
import numpy as np 
import cv2 
import scipy.io as io 

img = io.loadmat('../OASIS1_T1/test/sample_0/norm.mat')['norm'] 
mask = io.loadmat('../OASIS1_T1/test/sample_0/seg.mat')['seg'] 
idx = 100 

img, mask = img[:, :, idx], mask[:, :, idx] 

# print(np.max(img), np.max(mask))  

img = (img.astype(np.float32) * 255.0).astype(np.uint8)
mask = mask.astype(np.uint8) 

cv2.imwrite('./fig/img.jpg', img) 
cv2.imwrite('./fig/mask.jpg', mask) 

img = cv2.imread('./fig/img.jpg') 
mask = cv2.imread('./fig/mask.jpg', cv2.IMREAD_GRAYSCALE) 

# mask_copy = mask.copy() 
# mask_copy[mask == 1] = 255 
# mask_copy[mask != 1] = 0 
# cv2.imwrite('./fig/mask1.png', mask_copy)


# mask_copy = mask.copy() 
# # mask_copy[mask_copy == int(label)] = 0 
# mask_copy[mask_copy == 9] = 255
# # mask_copy[mask_copy != 9] = 0 
# contours, hierarchy = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
# img1 = cv2.drawContours(img, contours, -1, (255, 0, 0), 1) 

# mask = mask[mask == 1]
# contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
lc_dict = {'1': (255, 0, 0),
               '20': (255, 0, 0), 
               '9': (0, 255, 0), 
               '28': (0, 255, 0), 
               '5': (0, 0, 255),
               '24': (0, 0, 255)}
img1 = utils.draw_contours(img, mask, lc_dict) 
cv2.imwrite('./fig/over.jpg', img1) 