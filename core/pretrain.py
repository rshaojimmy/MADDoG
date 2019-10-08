"""Adversarial adaptation to train target encoder."""

import os
from collections import OrderedDict
import torchvision.utils as vutils
import torch
import torch.optim as optim
from torch import nn
from misc.utils import get_inf_iterator, mkdir
from misc import evaluate
from torch.nn import DataParallel
from models import loss
import numpy as np
import h5py


from pdb import set_trace as st


def Pre_train(args, FeatExtor, DepthEsmator, data_loader_real, data_loader_fake,
            summary_writer, saver, savefilename):


    # savepath = os.path.join(args.results_path, savefilename)
    # mkdir(savepath)  
    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    FeatExtor.train()
    DepthEsmator.train()

    FeatExtor = DataParallel(FeatExtor)    
    DepthEsmator = DataParallel(DepthEsmator) 

    criterionDepth = torch.nn.MSELoss()

    optimizer_DG_depth = optim.Adam(list(FeatExtor.parameters())+list(DepthEsmator.parameters()),
                           lr=args.lr_DG_depth,
                           betas=(args.beta1, args.beta2))    

    iternum = max(len(data_loader_real),len(data_loader_fake))   

    print('iternum={}'.format(iternum))

    ####################
    # 2. train network #
    ####################
    global_step = 0

    for epoch in range(args.pre_epochs):

        # epoch=epochNum+5

        data_real = get_inf_iterator(data_loader_real)
        data_fake = get_inf_iterator(data_loader_fake)


        for step in range(iternum):

            cat_img_real, depth_img_real, lab_real = next(data_real)
            cat_img_fake, depth_img_fake, lab_fake = next(data_fake)

            ori_img = torch.cat([cat_img_real,cat_img_fake],0)
            ori_img = ori_img.cuda()

            depth_img = torch.cat([depth_img_real,depth_img_fake],0)
            depth_img = depth_img.cuda()

            feat_ext,_= FeatExtor(ori_img)
            depth_Pre = DepthEsmator(feat_ext)

            Loss_depth = criterionDepth(depth_Pre, depth_img)

            optimizer_DG_depth.zero_grad()
            Loss_depth.backward()
            optimizer_DG_depth.step()

            info = {
                'Loss_depth': Loss_depth.item(),               
                    }           
            for tag, value in info.items():
                summary_writer.add_scalar(tag, value, global_step)   

            #============ print the log info ============# 
            if (step+1) % args.log_step == 0:
                errors = OrderedDict([('Loss_depth', Loss_depth.item())])  
                saver.print_current_errors((epoch+1), (step+1), errors)   

            global_step+=1

        if ((epoch + 1) % args.model_save_epoch == 0):
            model_save_path = os.path.join(args.results_path, 'snapshots', savefilename)     
            mkdir(model_save_path) 

            torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
                "DGFA-Ext-{}.pt".format(epoch+1)))
            torch.save(DepthEsmator.state_dict(), os.path.join(model_save_path,
                "DGFA-Depth-{}.pt".format(epoch+1)))        


    torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
        "DGFA-Ext-final.pt"))
    torch.save(DepthEsmator.state_dict(), os.path.join(model_save_path,
        "DGFA-Depth-final.pt"))



