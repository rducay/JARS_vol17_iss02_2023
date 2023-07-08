import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from models.SSRNET import SSRNET
from models.SingleCNN import SpatCNN, SpecCNN
from models.TFNet import TFNet, ResTFNet
from models.SSFCNN import SSFCNN, ConSSFCNN
from models.MSDCNN import MSDCNN
from utils import *
from data_loader import build_datasets
from validate import validate
from train import train
import pdb
import args_parser
from torch.nn import functional as F
import numpy as np
import spectral.io.envi as envi
import os

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print (args)


def main():
    # Custom dataloader
    train_list, test_list = build_datasets(args.root, 
                                           args.dataset, 
                                           args.image_size, 
                                           args.n_select_bands, 
                                           args.scale_ratio)
    if args.dataset=='avon':
        args.n_bands = 327
        
    elif args.dataset=='chikusei':
        args.n_bands = 128
    
    elif args.dataset=='cookecity':
        args.n_bands = 126
    
    elif args.dataset=='cupriteng':
        args.n_bands = 372
    
    elif args.dataset=='gudalur':
        args.n_bands = 370
    
    elif args.dataset=='ritcampus':
        args.n_bands = 327
    
    # Build the models
    if args.arch == 'SSFCNN':
        model = SSFCNN(args.scale_ratio,
                       args.n_select_bands,
                       args.n_bands).cuda()
        
    elif args.arch == 'ConSSFCNN':
        
        model = ConSSFCNN(args.scale_ratio,
                          args.n_select_bands,
                          args.n_bands).cuda()
    
    elif args.arch == 'TFNet':
        model = TFNet(args.scale_ratio,
                      args.n_select_bands,
                      args.n_bands).cuda()
    
    elif args.arch == 'ResTFNet':
        model = ResTFNet(args.scale_ratio,
                         args.n_select_bands, 
                         args.n_bands).cuda()
    
    elif args.arch == 'MSDCNN':
        model = MSDCNN(args.scale_ratio,
                       args.n_select_bands,
                       args.n_bands).cuda()
        
    elif args.arch == 'SSRNET' or args.arch == 'SpatRNET' or args.arch == 'SpecRNET':
        model = SSRNET(args.arch,
                       args.scale_ratio,
                       args.n_select_bands, 
                       args.n_bands).cuda()
    
    elif args.arch == 'SpatCNN':
        model = SpatCNN(args.scale_ratio,
                        args.n_select_bands, 
                        args.n_bands).cuda()
    
    elif args.arch == 'SpecCNN':
        model = SpecCNN(args.scale_ratio,
                        args.n_select_bands, 
                        args.n_bands).cuda()

    # Loss and optimizer
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('arch', args.arch) 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))
        recent_psnr = validate(test_list, 
                                args.arch,
                                model,
                                0,
                                args.n_epochs,
                                args.dataset,
                                args.image_size)
        print ('psnr: ', recent_psnr)

    best_psnr = 0
    best_psnr = validate(test_list,
                          args.arch, 
                          model,
                          0,
                          args.n_epochs,
                          args.dataset,
                          args.image_size)
    print ('psnr: ', best_psnr)

    # Epochs
    print ('Start Training: ')
    for epoch in range(args.n_epochs):
        # One epoch's training
        print ('Train_Epoch_{}: '.format(epoch))
        train(train_list, 
              args.image_size,
              args.scale_ratio,
              args.n_bands, 
              args.arch,
              model, 
              optimizer, 
              criterion, 
              epoch, 
              args.n_epochs)

        # One epoch's validation
        print ('Val_Epoch_{}: '.format(epoch))
        recent_psnr = validate(test_list, 
                                args.arch,
                                model,
                                epoch,
                                args.n_epochs,
                                args.dataset,
                                args.image_size)
        print ('psnr: ', recent_psnr)
        
        # # save model
        is_best = recent_psnr > best_psnr
        best_psnr = max(recent_psnr, best_psnr)
        if is_best:
          torch.save(model.state_dict(), model_path)
          print ('Saved!')
          print ('')
          
    
    print ('best_psnr: ', best_psnr)

if __name__ == '__main__':
    main()
