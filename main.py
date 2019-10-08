"""Main script for MADDoG."""
import os
import os.path as osp
import argparse

import torch
from torch import nn
from tensorboardX import SummaryWriter
from core import Train, Pre_train
from datasets.DatasetLoader import get_dataset_loader
from datasets.TargetDatasetLoader import get_tgtdataset_loader
from misc.utils import init_model, init_random_seed, mkdirs
from misc.saver import Saver
import models

from pdb import set_trace as st

def main(args):

    if args.training_type is 'Train':
        savefilename = osp.join(args.dataset1+args.dataset2+args.dataset3+'1')
    elif  args.training_type is 'Pre_train':    
        savefilename = osp.join(args.dataset_target+'') 
    elif  args.training_type is 'Test':    
        savefilename = osp.join(args.tstfile, args.tstdataset+args.snapshotnum) 

    args.seed = init_random_seed(args.manual_seed)

    if args.training_type in ['Train', 'Pre_train', 'Test']:
        summary_writer = SummaryWriter(osp.join(args.results_path, 'log', savefilename))
        saver = Saver(args,savefilename)
        saver.print_config()

    ##################### load seed#####################  

    #####################load datasets##################### 

    if args.training_type is 'Train':

        data_loader1_real = get_dataset_loader(name=args.dataset1, getreal=True, batch_size=args.batchsize)
        data_loader1_fake = get_dataset_loader(name=args.dataset1, getreal=False, batch_size=args.batchsize)

        data_loader2_real = get_dataset_loader(name=args.dataset2, getreal=True, batch_size=args.batchsize)
        data_loader2_fake = get_dataset_loader(name=args.dataset2, getreal=False, batch_size=args.batchsize)

        data_loader3_real = get_dataset_loader(name=args.dataset3, getreal=True, batch_size=args.batchsize)
        data_loader3_fake = get_dataset_loader(name=args.dataset3, getreal=False, batch_size=args.batchsize)

        data_loader_target = get_tgtdataset_loader(name=args.dataset_target, batch_size=args.batchsize) 

    elif args.training_type is 'Test':

        data_loader_target = get_tgtdataset_loader(name=args.dataset_target, batch_size=args.batchsize) 

    elif args.training_type is 'Pre_train':
        data_loader_real = get_dataset_loader(name=args.dataset_target, getreal=True, batch_size=args.batchsize)
        data_loader_fake = get_dataset_loader(name=args.dataset_target, getreal=False, batch_size=args.batchsize)

    ##################### load models##################### 

    FeatExtmodel = models.create(args.arch_FeatExt)
    FeatExtmodel_pre1 = models.create(args.arch_FeatExt)
    FeatExtmodel_pre2 = models.create(args.arch_FeatExt)
    FeatExtmodel_pre3 = models.create(args.arch_FeatExt)


    FeatEmbdmodel = models.create(args.arch_FeatEmbd, embed_size=args.embed_size)
    DepthEstmodel = models.create(args.arch_DepthEst)

    Dismodel1 = models.create(args.arch_Dis1)
    Dismodel2 = models.create(args.arch_Dis2)
    Dismodel3 = models.create(args.arch_Dis3)

    if args.training_type is 'Train':

        FeatExtS1_restore = osp.join('results', 'Pre_train', 'snapshots', args.dataset1, 'DGFA-Ext-final.pt')
        FeatExtS2_restore = osp.join('results', 'Pre_train', 'snapshots', args.dataset2, 'DGFA-Ext-final.pt')
        FeatExtS3_restore = osp.join('results', 'Pre_train', 'snapshots', args.dataset3, 'DGFA-Ext-final.pt')

        FeatExtorS1 = init_model(net=FeatExtmodel_pre1, init_type = args.init_type, restore=FeatExtS1_restore)
        FeatExtorS2 = init_model(net=FeatExtmodel_pre2, init_type = args.init_type, restore=FeatExtS2_restore)
        FeatExtorS3 = init_model(net=FeatExtmodel_pre3, init_type = args.init_type, restore=FeatExtS3_restore)


        Dis_restore1 = None
        Dis_restore2 = None
        Dis_restore3 = None

        FeatExt_restore = None
        DepthEst_restore = None
        FeatEmbd_restore = None

        FeatEmbder= init_model(net=FeatEmbdmodel, init_type = args.init_type, restore=FeatEmbd_restore)


    elif args.training_type is 'Pre_train':
        FeatExt_restore = None
        DepthEst_restore = None

        Dis_restore1 = None
        Dis_restore2 = None
        Dis_restore3 = None

    elif args.training_type is 'Test':
        FeatExt_restore = osp.join('results', args.tstfile, 'snapshots', args.tstdataset, 'DGFA-Ext-'+args.snapshotnum+'.pt')
        DepthEst_restore = osp.join('results', args.tstfile, 'snapshots', args.tstdataset, 'DGFA-Depth-'+args.snapshotnum+'.pt')
        FeatEmbd_restore = osp.join('results', args.tstfile, 'snapshots', args.tstdataset, 'DGFA-Embd-'+args.snapshotnum+'.pt')
        FeatEmbder= init_model(net=FeatEmbdmodel, init_type = args.init_type, restore=FeatEmbd_restore)

        Dis_restore1 = None
        Dis_restore2 = None
        Dis_restore3 = None

    else:
        raise NotImplementedError('method type [%s] is not implemented' % args.training_type)



    FeatExtor = init_model(net=FeatExtmodel, init_type = args.init_type, restore=FeatExt_restore)
    DepthEstor= init_model(net=DepthEstmodel, init_type = args.init_type, restore=DepthEst_restore)

    Discriminator1 = init_model(net=Dismodel1, init_type = args.init_type, restore=Dis_restore1)
    Discriminator2 = init_model(net=Dismodel2, init_type = args.init_type, restore=Dis_restore2)
    Discriminator3 = init_model(net=Dismodel3, init_type = args.init_type, restore=Dis_restore3)


    print(">>> FeatExtor <<<")
    print(FeatExtor)

    print(">>> FeatEmbder <<<")
    print(FeatEmbder)    

    print(">>> DepthEstor <<<")
    print(DepthEstor)

    print(">>> Discriminator <<<")
    print(Discriminator1)

    ##################### tarining models##################### 

    if args.training_type is 'Train':

        Train(args, FeatExtor, DepthEstor, FeatEmbder, Discriminator1, Discriminator2, Discriminator3,
               FeatExtorS1, FeatExtorS2, FeatExtorS3,
               data_loader1_real, data_loader1_fake,
               data_loader2_real, data_loader2_fake,
               data_loader3_real, data_loader3_fake,
               data_loader_target,
               summary_writer, saver, savefilename)  

    elif args.training_type is 'Test':     

        Test(args, FeatExtor, DepthEstor, FeatEmbder, data_loader_target, savefilename)

    elif args.training_type is 'Pre_train':

        Pre_train(args, FeatExtor, DepthEstor, data_loader_real, data_loader_fake,
                summary_writer, saver, savefilename)
    else:
        raise NotImplementedError('method type [%s] is not implemented' % args.training_type)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MADDoG")

    # datasets 
        # OMI
    parser.add_argument('--dataset1', type=str, default='OULU')
    parser.add_argument('--dataset2', type=str, default='MSU')
    parser.add_argument('--dataset3', type=str, default='idiap')
    parser.add_argument('--dataset_target', type=str, default='CASIA')

        #OIC
    # parser.add_argument('--dataset1', type=str, default='OULU')
    # parser.add_argument('--dataset2', type=str, default='idiap')
    # parser.add_argument('--dataset3', type=str, default='CASIA')
    # parser.add_argument('--dataset_target', type=str, default='MSU')
        #ICM    
    # parser.add_argument('--dataset1', type=str, default='idiap')
    # parser.add_argument('--dataset2', type=str, default='CASIA')
    # parser.add_argument('--dataset3', type=str, default='MSU')
    # parser.add_argument('--dataset_target', type=str, default='OULU')
        #OCM
    # parser.add_argument('--dataset1', type=str, default='OULU')
    # parser.add_argument('--dataset2', type=str, default='CASIA')
    # parser.add_argument('--dataset3', type=str, default='MSU')
    # parser.add_argument('--dataset_target', type=str, default='idiap')  

    # model
    parser.add_argument('--arch_FeatExt', type=str, default='FeatExtractor')
    parser.add_argument('--arch_FeatEmbd', type=str, default='FeatEmbedder')
    parser.add_argument('--arch_DepthEst', type=str, default='DepthEstmator')
    parser.add_argument('--arch_Dis1', type=str, default='Discriminator1')
    parser.add_argument('--arch_Dis2', type=str, default='Discriminator2')
    parser.add_argument('--arch_Dis3', type=str, default='Discriminator3')

    parser.add_argument('--init_type', type=str, default='xavier')
    parser.add_argument('--embed_size', type=int, default=128)

    # optimizer
    parser.add_argument('--lr_DG_depth', type=float, default=0.0001)
    parser.add_argument('--lr_DG_conf', type=float, default=0.00001)
    parser.add_argument('--lr_critic', type=float, default=0.00001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # # training configs
    parser.add_argument('--training_type', type=str, default='Train')
    parser.add_argument('--results_path', type=str, default='./results/Train_20191008')
    parser.add_argument('--batchsize', type=int, default=10)

    # parser.add_argument('--training_type', type=str, default='Pre_train')
    # parser.add_argument('--results_path', type=str, default='./results/Pre_train/')
    # parser.add_argument('--batchsize', type=int, default=10)
    # parser.add_argument('--dataset_target', type=str, default='MSU')   


    # parser.add_argument('--training_type', type=str, default='Test')
    # parser.add_argument('--results_path', type=str, default='./results/Test_20191008/')
    # parser.add_argument('--batchsize', type=int, default=1)
    # parser.add_argument('--tstfile', type=str, default='Train_20191008')
    # parser.add_argument('--tstdataset', type=str, default='OULUCASIAMSU')    
    # parser.add_argument('--snapshotnum', type=str, default='2')


    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pre_epochs', type=int, default=10)
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--tst_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=500)
    parser.add_argument('--model_save_epoch', type=int, default=1)
    parser.add_argument('--manual_seed', type=int, default=None)

    parser.add_argument('--W_trip', type=int, default=1)
    parser.add_argument('--W_depth', type=int, default=1)
    parser.add_argument('--W_gen', type=int, default=1)
    parser.add_argument('--W_intra', type=int, default=0.1)
    parser.add_argument('--W_cls', type=int, default=1)
    parser.add_argument('--W_genave', type=int, default=1/3)


    print(parser.parse_args())
    main(parser.parse_args())

