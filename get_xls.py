import pandas as pd
import numpy as np
from src import utils 
import cv2 

def main(file_name): 

    if file_name == 'lpba': 
        num_samples, num_areas = 9, 12 
    if file_name == 'oasis': 
        num_samples, num_areas = 40, 16

    num_columns = num_areas * 7 
    results = np.zeros((num_samples, num_columns))

    # read files in 
    ants = np.loadtxt('./log/%s_ants.txt' % file_name)
    vtn = np.loadtxt('./log/%s_vtn.txt' % file_name)
    vm = np.loadtxt('./log/%s_vm.txt' % file_name)
    cc = np.loadtxt('./log/%s_cc.txt' % file_name) 
    transunet = np.loadtxt('./log/%s_transunet.txt' % file_name) 
    swinunet = np.loadtxt('./log/%s_swinunet.txt' % file_name) 
    main = np.loadtxt('./log/%s_main.txt' % file_name)

    allArr = [ants, vm, vtn, cc, transunet, swinunet, main]

    for idx in range(7):
        arr = allArr[idx]
        for area_idx in range(num_areas): 
            dataVec = arr[area_idx * num_samples: (area_idx + 1) * num_samples]
            results[:, idx + area_idx * 7] = dataVec

    # column = [str(x) for x in range(num_columns)]
    df = pd.DataFrame(results)
    df.to_excel('./xlsx/%s_out.xlsx' % file_name, index=0)

def ablation(): 
    num_samples, num_areas = 9, 12 
    num_columns = num_areas * 3 
    results = np.zeros((num_samples, num_columns)) 
    
    # read files in  
    main = np.loadtxt('./log/lpba_main.txt') 
    main_notf = np.loadtxt('./log/lpba_main_notf.txt') 
    main_stf = np.loadtxt('./log/lpba_main_stf.txt') 
    
    allArr = [main, main_notf, main_stf] 
    for idx in range(3): 
        arr = allArr[idx] 
        for area_idx in range(num_areas): 
            dataVec = arr[area_idx * num_samples: (area_idx + 1) * num_samples]
            results[:, idx + area_idx * 3] = dataVec 
            
    df = pd.DataFrame(results)
    df.to_excel('./xlsx/lpba_ablation.xlsx', index=0)
    
def overlap(): 
    # load gray img and mask 
    imgs = [] 
    masks = [] 
    names = ['ants', 'cc', 'fixed', 'main', 'moving', 'vm', 'vtn', 'transunet', 'swinunet']  
    # names = ['ants', 'cc', 'fixed', 'main', 'moving', 'vm', 'vtn'] 
    # names = ['fixed', 'moving']
    for name in names: 
        img_name = 'img_' + name + '.png' 
        mask_name = 'mask_' + name + '.png' 
        img = cv2.imread('./fig/' + img_name) 
        mask = cv2.imread('./fig/' + mask_name, cv2.IMREAD_GRAYSCALE) 
        imgs.append(img) 
        masks.append(mask) 
    # draw contours and save 
    lc_dict = {'1': (255, 0, 0),
               '20': (255, 0, 0), 
               '14': (0, 255, 0), 
               '30': (0, 255, 0), 
               '9': (0, 0, 255), 
               '28': (0, 0, 255)}   
    for (name, img, mask) in zip(names, imgs, masks): 
        over = utils.draw_contours(img, mask, lc_dict) 
        cv2.imwrite('./fig/' + 'over_' + name + '.png', over) 

if __name__ == '__main__': 
    # main('oasis') 
    # ablation() 
    overlap()


