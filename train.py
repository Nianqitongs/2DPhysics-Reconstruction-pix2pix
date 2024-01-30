# 院校：南京信息工程大学
# 院系：自动化学院
# 开发时间：2024/1/30 8:54
from __future__ import print_function
from utils.utils import array2tensor
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import torch.backends.cudnn as cudnn
from model.networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model.unet_model import UNet
#from tensorboardX import SummaryWriter
import time
from test import test_per_epoch

def train(opt):
    #writer = SummaryWriter(opt.logdir)
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    cudnn.benchmark = True

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    data = np.load(opt.data_path)
    condition = np.load(opt.cond_path)
    init_condition = np.load(opt.repeat_path)

    train_data, test_data, train_condition, test_condition, train_label, test_label = train_test_split(init_condition, condition, data, test_size=0.02, random_state=10)

    train_data = array2tensor(train_data).unsqueeze(1)
    test_data = array2tensor(test_data).unsqueeze(1)

    train_condition = array2tensor(train_condition)
    test_condition = array2tensor(test_condition)

    train_label = array2tensor(train_label).unsqueeze(1)
    test_label = array2tensor(test_label).unsqueeze(1)

    train_dataset = TensorDataset(train_data, train_condition, train_label)
    test_dataset = TensorDataset(test_data, test_condition, test_label)
    #print(train_dataset[5550][2].shape)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size)  # 128
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size)  # 64

    device = torch.device("cuda:0")

    print('===> Building models')
    # net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)#.to(device)
    net_g = UNet(3, 1).to(device)
    net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)  # .to(device)

    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    #criterionMSE = nn.MSELoss().to(device)

    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)
    net_d_scheduler = get_scheduler(optimizer_d, opt)

    val_loss = 1000000
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        # train
        # train_loss = 0
        D_loss = 0
        G_loss = 0
        l1_loss = 0
        for iteration, batch in enumerate(train_dataloader, 1):
            # forward
            real_a, condition_a, real_b = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            condition_input = condition_a.view(real_a.size(0), 2, 1, 1)
            real_input_a = torch.cat([real_a, torch.ones_like(real_a) * condition_input], dim=1)
            fake_b = net_g(real_input_a)

            ######################
            # (1) Update D network
            ######################

            optimizer_d.zero_grad()

            # train with fake
            fake_ab = torch.cat((real_input_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab.detach())
            loss_d_fake = criterionGAN(pred_fake, False)  # 0.79

            # train with real
            real_ab = torch.cat((real_input_a, real_b), 1)
            pred_real = net_d.forward(real_ab)
            loss_d_real = criterionGAN(pred_real, True)  # 1.34

            # Combined D loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5  # 1.09

            D_loss += loss_d.item()

            loss_d.backward()

            optimizer_d.step()

            ######################
            # (2) Update G network
            ######################

            optimizer_g.zero_grad()

            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_input_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab)
            loss_g_gan = criterionGAN(pred_fake, True)  # 7.2743

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb  # 23.60

            loss_g = loss_g_gan + loss_g_l1

            G_loss += loss_g.item()
            l1_loss += loss_g_l1.item()

            loss_g.backward()

            optimizer_g.step()

            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, iteration, len(train_dataloader), loss_d.item(), loss_g.item()))

        #writer.add_scalar('G_loss', G_loss / iteration, global_step=epoch)
        #writer.add_scalar('D_loss', D_loss / iteration, global_step=epoch)
        #writer.add_scalar('l1_loss', l1_loss / iteration, global_step=epoch)
        lossl1 = test_per_epoch(net_g, test_dataloader, epoch, opt, criterionL1)
        #writer.add_scalar('test_l1_loss', lossl1, global_step=epoch)
        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        model_path_G = 'output/weights/model_gtest.pth'
        model_path_D = 'output/weights/model_dtest.pth'
        if lossl1 <= val_loss:
            torch.save(net_g, model_path_G)
            torch.save(net_d, model_path_D)
            val_loss = lossl1

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    parser.add_argument('--dataset', default='facades', help='facades')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
    parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--niter', type=int, default=10000, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=10000, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
    parser.add_argument('--label_path', default='/data1/zzd/pix2pix-pytorch-master/output/label/')
    parser.add_argument('--predicate_path', default='/data1/zzd/pix2pix-pytorch-master/output/predicate/')
    parser.add_argument('--logdir', default='/data1/zzd/pix2pix-pytorch-master/output/result_test')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--data_path', default='data/reconstruct.npy')
    parser.add_argument('--cond_path', default='data/ut.npy')
    parser.add_argument('--repeat_path', default='data/repeat.npy')
    opt = parser.parse_args()

    train(opt)