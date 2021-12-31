import os
import argparse
from typing import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import log10
from PIL import Image

import torch
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import REDSDataset
from model import basicVSR
from loss import CharbonnierLoss
from utils import resize_sequences

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', default='../datasets/REDS/train_sharp')
parser.add_argument('--lq_dir', default='../datasets/REDS/train_sharp_bicubic/X4')
parser.add_argument('--log_dir', default='./log_dir')
parser.add_argument('--spynet_pretrained', default='spynet_20210409-c6c1bd09.pth')
parser.add_argument('--scale_factor', default=4,type=int)
parser.add_argument('--batch_size', default=8,type=int)
parser.add_argument('--patch_size', default=64,type=int)
parser.add_argument('--epochs', default=300000,type=int)
parser.add_argument('--num_input_frames', default=15,type=int)
parser.add_argument('--val_interval', default=1000,type=int)
args = parser.parse_args()

train_set=REDSDataset(args.gt_dir,args.lq_dir,args.scale_factor,args.patch_size,args.num_input_frames,is_test=False)
val_set=REDSDataset(args.gt_dir,args.lq_dir,args.scale_factor,args.patch_size,args.num_input_frames,is_test=True)

train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=os.cpu_count(),pin_memory=True)
val_loader=DataLoader(val_set,batch_size=1,num_workers=os.cpu_count(),pin_memory=True)

model=basicVSR(spynet_pretrained=args.spynet_pretrained).cuda()

criterion=CharbonnierLoss().cuda()
criterion_mse=nn.MSELoss().cuda()
optimizer = torch.optim.Adam([
        {'params': model.spynet.parameters(), 'lr': 2.5e-5},
        {'params': model.backward_resblocks.parameters()},
        {'params': model.forward_resblocks.parameters()},
        {'params': model.fusion.parameters()},
        {'params': model.upsample1.parameters()},
        {'params': model.upsample2.parameters()},
        {'params': model.conv_hr.parameters()},
        {'params': model.conv_last.parameters()}
        ], lr=2e-4, betas=(0.9,0.99)
    )

max_epoch=args.epochs
scheduler=CosineAnnealingLR(optimizer,T_max=max_epoch,eta_min=1e-7)

os.makedirs(f'{args.log_dir}/models',exist_ok=True)
os.makedirs(f'{args.log_dir}/images',exist_ok=True)
train_loss=[]
validation_loss=[]
for epoch in range(max_epoch):
    model.train()
    # fix SPyNet and EDVR at first 5000 iteration
    if epoch < 5000:
        for k, v in model.named_parameters():
            if 'spynet' in k or 'edvr' in k:
                v.requires_grad_(False)
    elif epoch == 5000:
        # train all the parameters
        model.requires_grad_(True)

    epoch_loss = 0
    with tqdm(train_loader, ncols=100) as pbar:
        for idx, data in enumerate(pbar):
            gt_sequences, lq_sequences = Variable(data[0]),Variable(data[1])
            gt_sequences=gt_sequences.to('cuda:0')
            lq_sequences=lq_sequences.to('cuda:0')

            optimizer.zero_grad()
            pred_sequences = model(lq_sequences)
            loss = criterion(pred_sequences, gt_sequences)
            epoch_loss += loss.item()
            #epoch_psnr += 10 * log10(1 / loss.data)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description(f'[Epoch {epoch+1}]')
            pbar.set_postfix(OrderedDict(loss=f'{loss.data:.3f}'))

        train_loss.append(epoch_loss/len(train_loader))

    if (epoch + 1) % args.val_interval != 0:
        continue

    model.eval()
    val_psnr,lq_psnr = 0,0
    os.makedirs(f'{args.log_dir}/images/epoch{epoch+1:05}',exist_ok=True)
    with torch.no_grad():
        for idx,data in enumerate(val_loader):
            gt_sequences, lq_sequences = data
            gt_sequences=gt_sequences.to('cuda:0')
            lq_sequences=lq_sequences.to('cuda:0')
            pred_sequences = model(lq_sequences)
            lq_sequences=resize_sequences(lq_sequences,(gt_sequences.size(dim=3),gt_sequences.size(dim=4)))
            val_mse = criterion_mse(pred_sequences, gt_sequences)
            lq_mse = criterion_mse(lq_sequences,gt_sequences)
            val_psnr += 10 * log10(1 / val_mse.data)
            lq_psnr += 10 * log10(1 / lq_mse.data)
            
            save_image(pred_sequences[0], f'{args.log_dir}/images/epoch{epoch+1:05}/{idx}_SR.png',nrow=5)
            save_image(lq_sequences[0], f'{args.log_dir}/images/epoch{epoch+1:05}/{idx}_LQ.png',nrow=5)
            save_image(gt_sequences[0], f'{args.log_dir}/images/epoch{epoch+1:05}/{idx}_GT.png',nrow=5)
    
    print(f'==[validation]== PSNR:{val_psnr / len(val_loader):.2f},(lq:{lq_psnr/len(val_loader):.2f})')
    torch.save(model.state_dict(),f'{args.log_dir}/models/model_{epoch}.pth')

x_train=list(range(max_epoch))
x_val=[x for x in range(max_epoch) if (epoch + 1) % args.val_interval == 0]
fig=plt.figure()
train_loss=[loss.cpu() for loss in train_loss]
validation_loss=[loss.cpu() for loss in validation_loss]
plt.plot(x_train,train_loss)
plt.plot(x_val,validation_loss)

fig.savefig(f'{args.log_dir}/loss.png')