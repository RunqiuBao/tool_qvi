from operator import index
import models
# import datas
import configs

import argparse
import torch
import torchvision
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from math import log10
import numpy as np
import datetime
from utils.config import Config
#from tensorboardX import SummaryWriter
import sys

import time

import cv2
from PIL import Image
import glob
import multiprocessing as mp
import math

# loading configures
parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--src_datapath', type=str, default="./datasets/example/car-turn", help='noisy img-folder path')
parser.add_argument('--dst_datapath', type=str, default="./outputs/example/car-turn", help='clean img-folder path')
parser.add_argument('--src_file_stye', type=str, default="%06d.jpg", help='clean img-folder path')
parser.add_argument('--dst_file_stye', type=str, default="%06d.png", help='clean img-folder path')
parser.add_argument('--numbias', type=int, default=0, help='frame number bias in GoPro dataset')
parser.add_argument('--sequence_length', type=int, default=200, help='how many frames in the sequence')
args = parser.parse_args()
# args = parser.parse_config()

config = Config.from_file(args.config)

# preparing datasets
normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
normalize2 = TF.Normalize([0, 0, 0], config.std)
trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

revmean = [-x for x in config.mean]
revstd = [1.0 / x for x in config.std]
revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
revNormalize = TF.Compose([revnormalize1, revnormalize2])

revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])

#testset = datas.Sequence(config.testset_root, trans, config.test_size, config.test_crop_size, config.inter_frames)
#sampler = torch.utils.data.SequentialSampler(testset)
#validationloader = torch.utils.data.DataLoader(testset, sampler=sampler, batch_size=1, shuffle=False, num_workers=1)

# model
model = getattr(models, config.model)(config.pwc_path).cuda()
model = nn.DataParallel(model)

tot_time = 0
tot_frames = 0

print('Everything prepared. Ready to test...')

def pil_imread(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pil_imwrite(path, img):
    img.save(path)
    

def store_data(out_imgs,start,end,start_pos,store_path):
    for k in range(start,end):     
        out = revtrans(out_imgs[k])
        # print('write to {}'.format(store_path + "/"+(args.dst_file_stye %(k+start_pos))))
        pil_imwrite(store_path + "/"+(args.dst_file_stye %(k+start_pos)),out)
        

def store_data_parallel(out_imgs,n_process,start_pos,store_path):
    img_length = len(out_imgs)
    n_process = min(n_process,img_length)
#     n_process = max(n_process,1)#bao
#     process_work = math.ceil(img_length / n_process)
    process_work_base = img_length // n_process
    remain_work = img_length - process_work_base*n_process
    process_work = [process_work_base]*n_process
    
    for i in range(remain_work):
        process_work[i] += 1
        
    job_pos = 0
    p_list = []
    
    for i in range(n_process):
        start = job_pos
        end = start + process_work[i]
        job_pos = end
        print("Process %d, start=%d, end=%d" %(i,start,end))
        store_data(out_imgs,start,end,start_pos,store_path)
    #     p=mp.Process(target=store_data,args=(out_imgs,start,end,start_pos,store_path,))
    #     p.start()
    #     p_list.append(p)
        
    # for p in p_list:
    #     p.join()

        

        

        
def generate_1():
    #print(os.path.exists("./datasets/example/car-turn/00000.jpg"))
    #data_path="/home/ma-user/work/data/mobile_sinagawa_240FPS_denoised/001"
    #store_path = "/home/ma-user/work/data/mobile_sinagawa_1920FPS_denoised/001"
    data_path=args.src_datapath
    store_path = args.dst_datapath
    os.makedirs(store_path,exist_ok=True)
    img_file = sorted(os.listdir(data_path))
    start_idx = 0
    n_process = 1
    store_num = 512
    batch_size = 4
    print(len(img_file))
    sequence_length = min(args.sequence_length, len(img_file))

    

    #--------processing--------
    start = time.time()
    out_imgs=[]
    frame_numbias = int(args.numbias)
    I0, I1, I2, I3 = [], [], [], []
    indexInBatch = 0
    print('tool_qvi: save {} images to {}'.format((frame_numbias, sequence_length+frame_numbias-3), store_path))
    for i in range(frame_numbias, sequence_length+frame_numbias-3):
        img_0 = np.array(pil_imread(data_path + "/" + (args.src_file_stye  %(i-1))))
        img_1 = np.array(pil_imread(data_path + "/" + (args.src_file_stye  %(i+0))))
        img_2 = np.array(pil_imread(data_path + "/" + (args.src_file_stye  %(i+1))))
        img_3 = np.array(pil_imread(data_path + "/" + (args.src_file_stye  %(i+2))))
        paddingwidth = 1
        img_0 = np.pad(img_0, ((paddingwidth,paddingwidth),(paddingwidth,paddingwidth),(0,0)), 'edge')
        img_1 = np.pad(img_1, ((paddingwidth,paddingwidth),(paddingwidth,paddingwidth),(0,0)), 'edge')
        img_2 = np.pad(img_2, ((paddingwidth,paddingwidth),(paddingwidth,paddingwidth),(0,0)), 'edge')
        img_3 = np.pad(img_3, ((paddingwidth,paddingwidth),(paddingwidth,paddingwidth),(0,0)), 'edge')
        
        img_0 = trans(img_0).unsqueeze(0)
        img_1 = trans(img_1).unsqueeze(0)
        img_2 = trans(img_2).unsqueeze(0)
        img_3 = trans(img_3).unsqueeze(0)
        
        
        I0.append(img_0.cuda())
        I1.append(img_1.cuda())
        I2.append(img_2.cuda())
        I3.append(img_3.cuda())

        if indexInBatch < (batch_size - 1) and i < (len(img_file)+frame_numbias-3-1):
            indexInBatch += 1
            continue
        else:
            I0 = torch.concat(I0, dim=0)
            I1 = torch.concat(I1, dim=0)
            I2 = torch.concat(I2, dim=0)
            I3 = torch.concat(I3, dim=0)
            indexInBatch = 0
        
        interFramesOutput = []
        with torch.no_grad():
            for tt in range(config.inter_frames):
                x = config.inter_frames
                t = 1.0/(x+1) * (tt + 1)
                output = model(I0, I1, I2, I3, t)
                It_warp = output
                interFramesOutput.append(It_warp.cpu())
        # stack output in correct order
        I1 = I1.cpu()
        for indexBatch in range(I1.shape[0]):
            out_imgs.append(I1[indexBatch, :, paddingwidth:-paddingwidth, paddingwidth:-paddingwidth])
            for indexInterFrame in range(config.inter_frames):
                out_imgs.append(interFramesOutput[indexInterFrame][indexBatch, :, paddingwidth:-paddingwidth, paddingwidth:-paddingwidth])

        I0, I1, I2, I3 = [], [], [], []
        # print("{}-th batch finish!".format(i))
                
        if len(out_imgs) == store_num:
            
            store_data_parallel(out_imgs, n_process, start_idx, store_path)
            start_idx += len(out_imgs)
            out_imgs=[]
            
            
    print ('tool_qvi: Proc time cost: %f s' %(time.time()-start))
    start = time.time()    
    
    
    store_data_parallel(out_imgs, n_process, start_idx, store_path)
        
    print ('tool_qvi: Save time cost: %f s' %(time.time()-start))


    '''
    main branch time record:
    Save time cost: 23.183207 s
    Done! cost: 5220.871939 s
    '''
    

    


        
        
def test():

    dict1 = torch.load(config.checkpoint)
    model.load_state_dict(dict1['model_state_dict'])

    if not os.path.exists(config.store_path):
        os.makedirs(config.store_path, exist_ok=False)
    generate_1()

#print(testset)
start = time.time()
test()
print ('Done! cost: %f s' %(time.time()-start))
#print ('Avg time is {} second'.format(tot_time))
