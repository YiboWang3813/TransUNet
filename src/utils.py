import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 


def weights_init_normal(m):
    """ initialize network's weight normally """
    if isinstance(m, nn.Conv3d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def cal_dice_oasis1(dicem):
    """ calculate the dice of oasis1 dataset """
    diceList = np.zeros(16)
    diceList[0] = dicem[12] # Brain-Stem 
    diceList[1] = (dicem[6] + dicem[25]) / 2# Thalamus
    diceList[2] = (dicem[2] + dicem[21]) / 2 # Lateral-Ventricle
    diceList[3] = (dicem[5] + dicem[24]) / 2 # Cerebellum-Cortex
    diceList[4] = (dicem[0] + dicem[19]) / 2 # Cerebral-White-Matter
    diceList[5] = (dicem[8] + dicem[27]) / 2 # Putamen
    diceList[6] = (dicem[7] + dicem[26]) / 2 # Caudate
    diceList[7] = (dicem[4] + dicem[23]) / 2 # Cerebellum-White-Matter
    diceList[8] = (dicem[16] + dicem[32]) / 2 # Ventral-DC
    diceList[9] = (dicem[13] + dicem[29]) / 2 # Hippocampus
    diceList[10] = (dicem[9] + dicem[28]) / 2 # Pallidum
    diceList[11] = dicem[11] # 4th-Ventricle
    diceList[12] = dicem[10] # 3rd-Ventricle
    diceList[13] = (dicem[14] + dicem[30]) / 2 # Amygdala
    diceList[14] = (dicem[1] + dicem[20]) / 2 # Cerebral-Cortex
    diceList[15] = (dicem[18] + dicem[34]) / 2 # Choroid-Plexus
    return diceList
 
def get_oasis1_labels():
    """ get the segment labels of oasis1 T1 dataset """
    return {'Cerebellum-Cortex': [6, 25], 
            'Cerebral-White-Matter': [1, 20],       
            'Brain-Stem': [13], 
            'Cerebral-Cortex': [2, 21], 
            'Cerebellum-White-Matter': [5, 24], 
            'Thalamus': [7, 26], 
            'Putamen': [9, 28], 
            'Ventral-DC': [17, 33], 
            'Hippocampus': [14, 30],
            'Caudate': [8, 27], 
            'Lateral-Ventricle': [3, 22], 
            'Pallidum': [10, 29], 
            'Amygdala': [15, 31], 
            '4th-Ventricle': [12], 
            'Choroid-Plexus': [19, 35],
            '3rd-Ventricle': [11]}

def get_lpba40_labels():
    """ get the segment labels of lpba40 T1 dataset """
    return {'Frontal': [21, 22, 23, 24, 25, 26], 
            'Occipital': [61, 62, 63, 64, 65, 66], 
            'Temporal': [81, 82, 83, 84, 85, 86], 
            'Orbitofrontal': [29, 30, 31, 32], 
            'Percentral': [27, 28], 
            'Postcentral': [41, 42], 
            'Parietal': [43, 44], 
            'Cingulate': [121, 122], 
            'Putamen': [163, 164], 
            'Hippocampus': [165, 166], 
            'Cerebellum': [181], 
            'Brainstem': [182]}

def normal(arr):
     """ normalize array 0~1 """
     max_val, min_val = np.max(arr), np.min(arr)
     return (arr - min_val) / (max_val - min_val)
    
def to_tensor(arr):
    """ convert the numpy ndarray to torch tensor """
    return torch.from_numpy(arr.copy())

def to_array(tensor, shape):
    """ convert torch tensor to numpy ndarray """
    # if tensor.device.find('cuda') != -1:
    #     tensor = tensor.cpu() 
    array = tensor.numpy() 
    array = array.reshape(shape)
    return array

def draw_reg_results(moving, fixed, warped, idx, path):
    """ draw the registration results in one map """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(moving[:, :, idx], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Moving')
    plt.subplot(1, 3, 2)
    plt.imshow(fixed[:, :, idx], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Fixed')
    plt.subplot(1, 3, 3)
    plt.imshow(warped[:, :, idx], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Warped')
    plt.savefig(path)

def draw_contours(img, mask, lc_dict): 
    for label, color in lc_dict.items(): 
        mask_copy = mask.copy() 
        # mask_copy[mask_copy == int(label)] = 0 
        mask_copy[mask == int(label)] = 255
        mask_copy[mask != int(label)] = 0 
        contours, hierarchy = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        img = cv2.drawContours(img, contours, -1, color, 1) 
    return img 

        
    



# predict_path = os.path.join(predict_file_path, name)
# pred_img = cv2.imread(predict_path, flags=cv2.IMREAD_GRAYSCALE)
# mask_path = os.path.join(mask_file_path, name.split('p')[0] + 'bmp')
# mask_img = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
# img_path = os.path.join(images_file_path, name.split('p')[0] + 'bmp')
# img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
# pred_img1 = pred_img.copy()
# mask_img1 = mask_img.copy()
# pred_img2 = pred_img.copy()
# mask_img2 = mask_img.copy()

# pred_img1[pred_img1 == 255] = 0
# mask_img1[mask_img1 == 255] = 0
# contours_p1, hierarchy_p1 = cv2.findContours(pred_img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# overlap_img = cv2.drawContours(img, contours_p1, -1, (0, 0, 255), 1)
# contours1, hierarchy1 = cv2.findContours(mask_img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# overlap_img = cv2.drawContours(overlap_img, contours1, -1, (0, 255, 0), 1)

# pred_img2[pred_img2 == 128] = 0
# mask_img2[mask_img2 == 128] = 0
# contours_p2, hierarchy_p2 = cv2.findContours(pred_img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# overlap_img = cv2.drawContours(img, contours_p2, -1, (0, 0, 255), 1)
# contours2, hierarchy2 = cv2.findContours(mask_img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# overlap_img = cv2.drawContours(overlap_img, contours2, -1, (0, 255, 0), 1)
# cv2.imwrite(save_path1 + '/' + '%d.png' % i, overlap_img)
# i += 1