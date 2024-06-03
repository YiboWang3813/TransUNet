import torch 
import time 
import os 
from thop import profile 
from networks.main_net import MainNet 
from networks.morph_net import MorphNet 
from networks.swinunet import SwinUnet 
from networks.transunet import TransUnet 

def get_model(name): 
    if name == 'VM': 
        model = MorphNet() 
    elif name == 'TUNet': 
        model = MainNet() 
    elif name == 'TransUnet': 
        model = TransUnet() 
    elif name == 'SwinUnet': 
        model = SwinUnet() 
    return model 

def main(): 
    
    X = torch.zeros((1, 2, 192, 160, 192)).cuda() 
    model_names = ['VM', 'TUNet', 'TransUnet', 'SwinUnet'] 
    for model_name in model_names: 
        model = get_model(model_name) 
        model.cuda() 
        # time 
        start_time = time.time() 
        with torch.no_grad(): 
            Y = model(X) 
        end_time = time.time() 
        pred_time = end_time - start_time 
        # FLOPs 
        flops, params = profile(model, inputs=(X, )) 
        gflops = flops / 1024 ** 3 
        print(f"model name: {model_name}, time cost: {pred_time:.4f} s, GFLOPs: {gflops}") 
        del model 
        
if __name__ == '__main__': 
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()  