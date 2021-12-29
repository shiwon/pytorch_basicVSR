import os
import argparse
from typing import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import log10

import torch
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from dataset import REDSDataset
from model import basicVSR
from loss import CharbonnierLoss

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', default='../datasets/REDS/train_sharp')
parser.add_argument('--lq_dir', default='../datasets/REDS/train_sharp_bicubic/X4')
parser.add_argument('--log_dir', default='./log_dir')
parser.add_argument('--spynet_pretrained', default='spynet_20210409-c6c1bd09.pth')
parser.add_argument('--scale_factor', default=4,type=int)
parser.add_argument('--batch_size', default=10,type=int)
parser.add_argument('--patch_size', default=128,type=int)
parser.add_argument('--epochs', default=300000,type=int)
parser.add_argument('--num_input_frames', default=15,type=int)
args = parser.parse_args()

train_set=REDSDataset(args.gt_dir,args.lq_dir,args.scale_factor,args.patch_size,args.num_input_frames,is_test=False)
val_set=REDSDataset(args.gt_dir,args.lq_dir,args.scale_factor,args.patch_size,args.num_input_frames,is_test=True)

train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=os.cpu_count(),pin_memory=True)
val_loader=DataLoader(val_set,batch_size=1,num_workers=os.cpu_count(),pin_memory=True)

model=basicVSR(spynet_pretrained=args.spynet_pretrained).cuda()

criterion=CharbonnierLoss().cuda()
#下URLのparamwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})とはいったい・・・
#https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_reds4.py
optimizer = torch.optim.Adam(model.parameters(),lr=2e-4,betas=(0.9,0.99))

os.makedirs(f'{args.log_dir}/models',exist_ok=True)
max_epoch=args.epochs
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

            pbar.set_description(f'[Epoch {epoch+1}]')
            pbar.set_postfix(OrderedDict(loss=f'{loss.data:.3f}'))

        train_loss.append(epoch_loss/len(train_loader))

    if (epoch + 1) % 1000 != 0:
        continue

    model.eval()
    val_loss,lq_loss = 0
    with torch.no_grad():
        for idx,data in enumerate(val_loader):
            gt_sequences, lq_sequences = data
            gt_sequences=gt_sequences.to('cuda:0')
            lq_sequences=lq_sequences.to('cuda:0')
            pred_sequences = model(lq_sequences)
            loss = criterion(pred_sequences, gt_sequences)
            val_loss+=loss.item()
            lq_loss+=criterion(lq_sequences,gt_sequences).item()
            #後で変える
            for n in range(args.batch_size):
                save_image(pred_sequences[n], f'{args.log_dir}/images/{args.batch_size*idx+n}_epoch{epoch+1:05}_SR.png')
                save_image(lq_sequences[n], f'{args.log_dir}/images/{args.batch_size*idx+n}_epoch{epoch+1:05}_LQ.png')
    
    print(f'==[validation]== loss:{val_loss / len(val_loader):.4f},(lq:{lq_loss/len(val_loader)})')
    torch.save(model.state_dict(),f'{args.log_dir}/models/model_{epoch}.pth')

x_train=list(range(max_epoch))
x_val=list(range(max_epoch//1000))
x_val=[x for x in range(max_epoch) if (epoch + 1) % 1000 == 0]
fig=plt.figure()
train_loss=[loss.cpu() for loss in train_loss]
validation_loss=[loss.cpu() for loss in validation_loss]
plt.plot(x_train,train_loss)
plt.plot(x_val,validation_loss)

fig.savefig(f'{args.log_dir}/loss.png')