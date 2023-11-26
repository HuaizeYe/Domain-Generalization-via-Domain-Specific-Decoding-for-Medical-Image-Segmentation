import os
import argparse
import sys

import numpy as np
from networks.deeplabv3_single import Deeplabv3p
from networks.unet import Unet2D, UnetDecoder
from utils.utils import count_params
from tensorboardX import SummaryWriter
import random
import dataset.transform as trans
import dataset.np_transform as np_trans
from torchvision.transforms import Compose

from dataset.fundus import Fundus_Multi
from dataset.prostate import Prostate_Multi
import torch.backends.cudnn as cudnn

from torch.nn import BCELoss, CrossEntropyLoss, DataParallel, MSELoss, L1Loss
import torch

from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
from utils.losses import dice_loss, dice_loss_multi
from utils.utils import decode_seg_map_sequence
import shutil
from validation_func import test_fundus, test_prostate
from model.model import PrivateDecoder, Discriminator
import warnings

warnings.filterwarnings('ignore')

l1_loss = L1Loss(reduction='mean').cuda()
mse_loss = MSELoss(reduction='mean').cuda()
sg_loss = CrossEntropyLoss(ignore_index=-1, reduction='mean')
recon_loss = l1_loss
dp_loss = mse_loss
di_loss = mse_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Train')
    # basic settings
    parser.add_argument('--data_root', type=str, default='/data/yhz', help='root path of training dataset')
    parser.add_argument('--data_cutmix', type=str, default='/data/yhz/cutmix', help='root path of training dataset')
    parser.add_argument('--dataset', type=str, default='fundus', choices=['fundus', 'prostate'],
                        help='training dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of training')
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size of testing')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--lr-rec', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr-dic', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lam-rec', type=float, default=1.0, help='learning rate')
    parser.add_argument('--lam-trans', type=float, default=0, help='learning rate')
    parser.add_argument('--lam-gen', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lam-adv', type=float, default=0.5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='training epochs')
    parser.add_argument('--domain_idxs', type=str, default='0,1,2', help='training epochs')
    parser.add_argument('--test_domain_idx', type=int, default=3, help='training epochs')
    parser.add_argument('--image_size', type=int, default=256, help='cropping size of training samples')
    parser.add_argument('--backbone', type=str, default='mobilenet', help='backbone of semantic segmentation model')
    parser.add_argument('--model', type=str, default=None, help='head of semantic segmentation model')
    parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pretrained backbone')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--output_stride', type=int, default=16, help='output stride of deeplab')
    parser.add_argument('--deterministic', action='store_true', help='whether use deterministic training')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='path of saved checkpoints')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    args = parser.parse_args()
    return args


def train_fundus(trainloader, model, writer, args, optimizer, testloader=None):
    criterion = BCELoss()
    true_label = 1
    fake_label = 0
    model = DataParallel(model).cuda()
    dec_0 = PrivateDecoder(256, 3).cuda()
    dec_1 = PrivateDecoder(256, 3).cuda()
    dec_2 = PrivateDecoder(256, 3).cuda()
    dec_3 = PrivateDecoder(256, 3).cuda()
    dis_0 = Discriminator(3, 1)
    dis_1 = Discriminator(3, 1)
    dis_2 = Discriminator(3, 1)
    dis_3 = Discriminator(3, 1)
    dec_0 = DataParallel(dec_0).cuda()
    dec_1 = DataParallel(dec_1).cuda()
    dec_2 = DataParallel(dec_2).cuda()
    dec_3 = DataParallel(dec_3).cuda()
    dis_0 = DataParallel(dis_0).cuda()
    dis_1 = DataParallel(dis_1).cuda()
    dis_2 = DataParallel(dis_2).cuda()
    dis_3 = DataParallel(dis_3).cuda()
    dec_0_opt = torch.optim.Adam(dec_0.parameters(), lr=args.lr_rec, betas=(0.5, 0.999))
    dec_1_opt = torch.optim.Adam(dec_1.parameters(), lr=args.lr_rec, betas=(0.5, 0.999))
    dec_2_opt = torch.optim.Adam(dec_2.parameters(), lr=args.lr_rec, betas=(0.5, 0.999))
    dec_3_opt = torch.optim.Adam(dec_3.parameters(), lr=args.lr_rec, betas=(0.5, 0.999))
    dis_0_opt = torch.optim.Adam(dis_0.parameters(), lr=args.lr_dic, betas=(0.5, 0.999))
    dis_1_opt = torch.optim.Adam(dis_1.parameters(), lr=args.lr_dic, betas=(0.5, 0.999))
    dis_2_opt = torch.optim.Adam(dis_2.parameters(), lr=args.lr_dic, betas=(0.5, 0.999))
    dis_3_opt = torch.optim.Adam(dis_3.parameters(), lr=args.lr_dic, betas=(0.5, 0.999))

    dec_0.train()
    dec_1.train()
    dec_2.train()
    dec_3.train()

    total_iters = len(trainloader) * args.epochs
    previous_best = 0.0
    iter_num = 0
    lam_trans = 0
    lam_rec = 0
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.6f" %
              (epoch, optimizer.param_groups[0]["lr"]))

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader, ncols=150)

        for i, (img, mask, idx) in enumerate(tbar):
            if iter_num >= 2000:
                lam_trans = args.lam_trans
            if iter_num >= 0:
                lam_rec = args.lam_rec
            loss_seg = 0
            loss_sgt = 0
            loss_rec = 0
            loss_adv = 0
            loss_gen = 0
            img, mask = img.cuda(), mask.cuda()
            # seg_loss
            pred_hard, pred_soft, code_common = model(img)

            loss_bce = criterion(pred_soft, mask)
            loss_dice = dice_loss(pred_soft, mask)
            loss = loss_bce + loss_dice
            loss_seg += loss.item()
            total_loss += loss.item()
            if lam_rec >= 0:
                # classify the different domains
                img0 = torch.empty(0).cuda()
                mask0 = torch.empty(0).cuda()
                fea0 = torch.empty(0).cuda()
                name0 = []
                len0 = 0
                img1 = torch.empty(0).cuda()
                mask1 = torch.empty(0).cuda()
                fea1 = torch.empty(0).cuda()
                name1 = []
                len1 = 0
                img2 = torch.empty(0).cuda()
                mask2 = torch.empty(0).cuda()
                fea2 = torch.empty(0).cuda()
                name2 = []
                len2 = 0
                img3 = torch.empty(0).cuda()
                mask3 = torch.empty(0).cuda()
                fea3 = torch.empty(0).cuda()
                name3 = []
                len3 = 0
                for domain_i in range(len(idx)):
                    if idx[domain_i][6] == '1':
                        img0 = torch.cat([img0, img[domain_i:domain_i + 1]], dim=0)
                        mask0 = torch.cat([mask0, mask[domain_i:domain_i + 1]], dim=0)
                        fea0 = torch.cat([fea0, code_common[domain_i:domain_i + 1]], dim=0)
                        name0.append(idx[domain_i])
                        len0 += 1
                    if idx[domain_i][6] == '2':
                        img1 = torch.cat([img1, img[domain_i:domain_i + 1]], dim=0)
                        mask1 = torch.cat([mask1, mask[domain_i:domain_i + 1]], dim=0)
                        fea1 = torch.cat([fea1, code_common[domain_i:domain_i + 1]], dim=0)
                        name1.append(idx[domain_i])
                        len1 += 1
                    if idx[domain_i][6] == '3':
                        img2 = torch.cat([img2, img[domain_i:domain_i + 1]], dim=0)
                        mask2 = torch.cat([mask2, mask[domain_i:domain_i + 1]], dim=0)
                        fea2 = torch.cat([fea2, code_common[domain_i:domain_i + 1]], dim=0)
                        name2.append(idx[domain_i])
                        len2 += 1
                    if idx[domain_i][6] == '4':
                        img3 = torch.cat([img3, img[domain_i:domain_i + 1]], dim=0)
                        mask3 = torch.cat([mask3, mask[domain_i:domain_i + 1]], dim=0)
                        fea3 = torch.cat([fea3, code_common[domain_i:domain_i + 1]], dim=0)
                        name3.append(idx[domain_i])
                        len3 += 1

                imgn0 = torch.cat([img1, img2, img3], dim=0)
                imgn1 = torch.cat([img0, img2, img3], dim=0)
                imgn2 = torch.cat([img0, img1, img3], dim=0)
                imgn3 = torch.cat([img0, img1, img2], dim=0)
                fean0 = torch.cat([fea1, fea2, fea3], dim=0)
                fean1 = torch.cat([fea0, fea2, fea3], dim=0)
                fean2 = torch.cat([fea0, fea1, fea3], dim=0)
                fean3 = torch.cat([fea0, fea1, fea2], dim=0)
                maskn0 = torch.cat([mask1, mask2, mask3], dim=0)
                maskn1 = torch.cat([mask0, mask2, mask3], dim=0)
                maskn2 = torch.cat([mask0, mask1, mask3], dim=0)
                maskn3 = torch.cat([mask0, mask1, mask2], dim=0)
                # trans
                imgt0_0 = torch.empty(0).cuda()
                imgt1_0 = torch.empty(0).cuda()
                imgt2_0 = torch.empty(0).cuda()
                imgt3_0 = torch.empty(0).cuda()
                imgt0_i = torch.empty(0).cuda()
                imgt1_i = torch.empty(0).cuda()
                imgt2_i = torch.empty(0).cuda()
                imgt3_i = torch.empty(0).cuda()

                # reconstuction loss & retrain segmentation loss

                if len0 != 0:
                    _, imgt0_0 = dec_0(fea0)
                    loss_rec0 = recon_loss(imgt0_0, img0) * len0 / args.batch_size
                    loss += lam_rec * loss_rec0
                    loss_rec += (lam_rec * loss_rec0).item()
                    if len0 != args.batch_size:
                        _, imgt0_i = dec_0(fean0)
                    imgt0 = torch.cat([imgt0_0, imgt0_i], dim=0)
                    mask = torch.cat([mask0, maskn0], dim=0)
                    pred_hard, pred_soft, code_common = model(imgt0.detach())
                    loss_bce = criterion(pred_soft, mask)
                    loss_dice = dice_loss(pred_soft, mask)
                    loss += lam_trans * (loss_bce + loss_dice)
                    loss_sgt += (lam_trans * (loss_bce + loss_dice)).item()
                if len1 != 0:
                    _, imgt1_0 = dec_1(fea1)
                    loss_rec1 = recon_loss(imgt1_0, img1) * len1 / args.batch_size
                    loss += lam_rec * loss_rec1
                    loss_rec += (lam_rec * loss_rec1).item()
                    if len1 != args.batch_size:
                        _, imgt1_i = dec_1(fean1)
                    imgt1 = torch.cat([imgt1_0, imgt1_i], dim=0)
                    mask = torch.cat([mask1, maskn1], dim=0)
                    pred_hard, pred_soft, code_common = model(imgt1.detach())
                    loss_bce = criterion(pred_soft, mask)
                    loss_dice = dice_loss(pred_soft, mask)
                    loss += lam_trans * (loss_bce + loss_dice)
                    loss_sgt += (lam_trans * (loss_bce + loss_dice)).item()
                if len2 != 0:
                    _, imgt2_0 = dec_2(fea2)
                    loss_rec2 = recon_loss(imgt2_0, img2) * len2 / args.batch_size
                    loss_rec += (lam_rec * loss_rec2).item()
                    loss += lam_rec * loss_rec2
                    if len2 != args.batch_size:
                        _, imgt2_i = dec_2(fean2)
                    imgt2 = torch.cat([imgt2_0, imgt2_i], dim=0)
                    mask = torch.cat([mask2, maskn2], dim=0)
                    pred_hard, pred_soft, code_common = model(imgt2.detach())
                    loss_bce = criterion(pred_soft, mask)
                    loss_dice = dice_loss(pred_soft, mask)
                    loss += lam_trans * (loss_bce + loss_dice)
                    loss_sgt += (lam_trans * (loss_bce + loss_dice)).item()
                if len3 != 0:
                    _, imgt3_0 = dec_3(fea3)
                    loss_rec3 = recon_loss(imgt3_0, img3) * len3 / args.batch_size
                    loss_rec += (lam_rec * loss_rec3).item()
                    loss += lam_rec * loss_rec3
                    if len3 != args.batch_size:
                        _, imgt3_i = dec_3(fean3)
                    imgt3 = torch.cat([imgt3_0, imgt3_i], dim=0)
                    mask = torch.cat([mask3, maskn3], dim=0)
                    pred_hard, pred_soft, code_common = model(imgt3.detach())
                    loss_bce = criterion(pred_soft, mask)
                    loss_dice = dice_loss(pred_soft, mask)
                    loss += lam_trans * (loss_bce + loss_dice)
                    loss_sgt += (lam_trans * (loss_bce + loss_dice)).item()
                if args.lam_adv > 0:
                    for p in dis_0.parameters():
                        p.requires_grad = True
                    for p in dis_1.parameters():
                        p.requires_grad = True
                    for p in dis_2.parameters():
                        p.requires_grad = True
                    for p in dis_3.parameters():
                        p.requires_grad = True
                    if len0 != 0 and len0 != args.batch_size:
                        _, imgt0_i = dec_0(fean0)
                        preal0 = dis_0(img0)
                        pfake0 = dis_0(imgt0_i.detach())
                        loss_d_t = 0.5 * di_loss(preal0,
                                                 torch.empty(preal0.shape).fill_(
                                                     true_label).cuda()) * len0 / args.batch_size + 0.5 * di_loss(
                            pfake0, torch.empty(pfake0.shape).fill_(fake_label).cuda()) * (
                                               args.batch_size - len0) / args.batch_size
                        loss_adv += loss_d_t.item() * args.lam_adv
                        loss_d_t *= args.lam_adv
                        loss_d_t.backward()
                        dis_0_opt.zero_grad()
                        dis_0_opt.step()
                    if len1 != 0 and len1 != args.batch_size:
                        _, imgt1_i = dec_1(fean1)
                        preal1 = dis_1(img1)
                        pfake1 = dis_1(imgt1_i.detach())
                        loss_d_t = 0.5 * di_loss(preal1,
                                                 torch.empty(preal1.shape).fill_(
                                                     true_label).cuda()) * len1 / args.batch_size + 0.5 * di_loss(
                            pfake1, torch.empty(pfake1.shape).fill_(fake_label).cuda()) * (
                                               args.batch_size - len1) / args.batch_size
                        loss_adv += loss_d_t.item() * args.lam_adv
                        loss_d_t *= args.lam_adv
                        loss_d_t.backward()
                        dis_1_opt.zero_grad()
                        dis_1_opt.step()
                    if len2 != 0 and len2 != args.batch_size:
                        _, imgt2_i = dec_2(fean2)
                        preal2 = dis_2(img2)
                        pfake2 = dis_2(imgt2_i.detach())
                        loss_d_t = 0.5 * di_loss(preal2,
                                                 torch.empty(preal2.shape).fill_(
                                                     true_label).cuda()) * len2 / args.batch_size + 0.5 * di_loss(
                            pfake2, torch.empty(pfake2.shape).fill_(fake_label).cuda()) * (
                                               args.batch_size - len2) / args.batch_size
                        loss_adv += loss_d_t.item() * args.lam_adv
                        loss_d_t *= args.lam_adv
                        loss_d_t.backward()
                        dis_2_opt.zero_grad()
                        dis_2_opt.step()
                    if len3 != 0 and len3 != args.batch_size:
                        _, imgt3_i = dec_3(fean3)
                        preal3 = dis_3(img3)
                        pfake3 = dis_3(imgt3_i.detach())
                        loss_d_t = 0.5 * di_loss(preal3,
                                                 torch.empty(preal3.shape).fill_(
                                                     true_label).cuda()) * len3 / args.batch_size + 0.5 * di_loss(
                            pfake3, torch.empty(pfake3.shape).fill_(fake_label).cuda()) * (
                                               args.batch_size - len3) / args.batch_size
                        loss_adv += loss_d_t.item() * args.lam_adv
                        loss_d_t *= args.lam_adv
                        loss_d_t.backward()
                        dis_3_opt.zero_grad()
                        dis_3_opt.step()
                    for p in dis_0.parameters():
                        p.requires_grad = False
                    for p in dis_1.parameters():
                        p.requires_grad = False
                    for p in dis_2.parameters():
                        p.requires_grad = False
                    for p in dis_3.parameters():
                        p.requires_grad = False
                    #     generator
                    if len0 != 0 and len0 != args.batch_size:
                        gfake0 = dis_0(imgt0_i)
                        loss_gen_0 = di_loss(gfake0,
                                             torch.empty(gfake0.shape).fill_(true_label).cuda()) * (
                                                 args.batch_size - len0) / args.batch_size
                        loss_gen += args.lam_gen *loss_gen_0.item()
                        loss += args.lam_gen * loss_gen_0
                    if len1 != 0 and len1 != args.batch_size:
                        gfake1 = dis_1(imgt1_i)
                        loss_gen_1 = di_loss(gfake1,
                                             torch.empty(gfake1.shape).fill_(true_label).cuda()) * (
                                                 args.batch_size - len1) / args.batch_size
                        loss_gen += args.lam_gen *loss_gen_1.item()
                        loss += args.lam_gen * loss_gen_1
                    if len2 != 0 and len2 != args.batch_size:
                        gfake2 = dis_2(imgt2_i)
                        loss_gen_2 = di_loss(gfake2,
                                             torch.empty(gfake2.shape).fill_(true_label).cuda()) * (
                                                 args.batch_size - len2) / args.batch_size
                        loss_gen += args.lam_gen *loss_gen_2.item()
                        loss += args.lam_gen * loss_gen_2
                    if len3 != 0 and len3 != args.batch_size:
                        gfake3 = dis_3(imgt3_i)
                        loss_gen_3 = di_loss(gfake3,
                                             torch.empty(gfake3.shape).fill_(true_label).cuda()) * (
                                                 args.batch_size - len3) / args.batch_size
                        loss_gen += args.lam_gen *loss_gen_3.item()
                        loss += args.lam_gen * loss_gen_3

            total_loss += loss_sgt + loss_rec + loss_gen
            # discriminator loss

            optimizer.zero_grad()
            dec_0.zero_grad()
            dec_1.zero_grad()
            dec_2.zero_grad()
            dec_3.zero_grad()
            loss.backward()
            optimizer.step()
            dec_0_opt.step()
            dec_1_opt.step()
            dec_2_opt.step()
            dec_3_opt.step()

            lr = args.lr * (1 - iter_num / total_iters) ** 0.9
            # lr_rec = args.lr_rec
            # if iter_num > 2000:
            lr_rec = args.lr_rec * (1 - iter_num / total_iters) ** 0.9
            dec_0_opt.param_groups[0]["lr"] = lr_rec
            dec_1_opt.param_groups[0]["lr"] = lr_rec
            dec_2_opt.param_groups[0]["lr"] = lr_rec
            dec_3_opt.param_groups[0]["lr"] = lr_rec
            optimizer.param_groups[0]["lr"] = lr

            iter_num = iter_num + 1

            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_sgt', loss_sgt, iter_num)
            writer.add_scalar('loss/loss_rec', loss_rec, iter_num)
            writer.add_scalar('loss/loss_adv', loss_adv, iter_num)
            writer.add_scalar('loss/loss_gen', loss_gen, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 100 == 0:
                image = img[0:3, 0:3, ...]
                grid_image = make_grid(image, 3, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                grid_image = make_grid(pred_soft[0:3, 0, ...].unsqueeze(1), 3, normalize=True)
                writer.add_image('train/Soft_Predicted_OC', grid_image, iter_num)

                grid_image = make_grid(pred_soft[0:3, 1, ...].unsqueeze(1), 3, normalize=True)
                writer.add_image('train/Soft_Predicted_OD', grid_image, iter_num)

                grid_image = make_grid(mask[0:3, 0, ...].unsqueeze(1), 3, normalize=False)
                writer.add_image('train/GT_OC', grid_image, iter_num)

                grid_image = make_grid(mask[0:3, 1, ...].unsqueeze(1), 3, normalize=False)
                writer.add_image('train/GT_OD', grid_image, iter_num)

            tbar.set_description(
                'Loss:%.3f SEG:%.3f REC:%.3f SGT:%.3f ADV:%.3f GEN:%.3f' % (
                    (total_loss / (i + 1)), loss_seg, loss_rec, loss_sgt, loss_adv, loss_gen))
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(args.save_path, 'checkpoints', 'iter_' + str(iter_num) + '.pth')
                save_dec0_path = os.path.join(args.save_path, 'checkpoints', 'dec0_' + str(iter_num) + '.pth')
                save_dec1_path = os.path.join(args.save_path, 'checkpoints', 'dec1_' + str(iter_num) + '.pth')
                save_dec2_path = os.path.join(args.save_path, 'checkpoints', 'dec2_' + str(iter_num) + '.pth')
                save_dec3_path = os.path.join(args.save_path, 'checkpoints', 'dec3_' + str(iter_num) + '.pth')
                torch.save(model.module.state_dict(), save_mode_path)
                torch.save(dec_0.module.state_dict(), save_dec0_path)
                torch.save(dec_1.module.state_dict(), save_dec1_path)
                torch.save(dec_2.module.state_dict(), save_dec2_path)
                torch.save(dec_3.module.state_dict(), save_dec3_path)

        if (epoch + 1) % 1 == 0:
            print("Test on target domain {}".format(args.test_domain_idx))
            test_fundus(model, epoch, args.data_root, args.test_domain_idx, args.save_path, args.test_batch_size,
                        dataset=args.dataset)

    save_mode_path = os.path.join(args.save_path, 'final_model.pth')
    torch.save(model.module.state_dict(), save_mode_path)
    print('\nSave Final Model to {}'.format(args.save_path))


def train_prostate(trainloader, model, writer, args, optimizer, testloader=None):
    criterion = CrossEntropyLoss()

    model = DataParallel(model).cuda()

    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0
    iter_num = 0
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.6f" %
              (epoch, optimizer.param_groups[0]["lr"]))

        model.train()
        total_loss = 0.0

        tbar = tqdm(trainloader, ncols=150)

        for i, (img, mask, onehot_label) in enumerate(tbar):
            img, mask, onehot_label = img.cuda(), mask.cuda(), onehot_label.cuda()

            pred = model(img)
            pred_soft = torch.softmax(pred, dim=1)

            loss_ce = criterion(pred, mask)
            loss_dice = dice_loss_multi(pred_soft, mask, num_classes=args.num_classes, ignore_index=0)

            loss = loss_ce + loss_dice

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = args.lr * (1 - iter_num / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            iter_num = iter_num + 1

            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar('loss/loss_ce', loss_ce, iter_num)
            writer.add_scalar('loss/loss_dice', loss_dice, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 100 == 0:
                image = img[0:3, 1, ...].unsqueeze(1)
                grid_image = make_grid(image, 3, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = torch.max(pred_soft[0:3, ...], 1)[1].detach().data.cpu().numpy()
                image = decode_seg_map_sequence(image)
                grid_image = make_grid(image, 3, normalize=False)
                writer.add_image('train/Predicted', grid_image, iter_num)

                image = mask[0:3, ...].detach().data.cpu().numpy()
                image = decode_seg_map_sequence(image)
                grid_image = make_grid(image, 3, normalize=False)
                writer.add_image('train/GT', grid_image, iter_num)

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(args.save_path, 'checkpoints', 'iter_' + str(iter_num) + '.pth')
                torch.save(model.module.state_dict(), save_mode_path)

        if (epoch + 1) % 1 == 0:
            print("Test on target domain {}".format(args.test_domain_idx))
            test_prostate(model, epoch, args.data_root, args.test_domain_idx, args.save_path, args.test_batch_size,
                          dataset=args.dataset)

    save_mode_path = os.path.join(args.save_path, 'final_model.pth')
    torch.save(model.module.state_dict(), save_mode_path)
    print('\nSave Final Model to {}'.format(args.save_path))


def main(args):
    data_root = os.path.join(args.data_root, args.dataset)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(os.path.join(args.save_path, 'checkpoints')):
        os.mkdir(os.path.join(args.save_path, 'checkpoints'))
    if os.path.exists(args.save_path + '/code'):
        shutil.rmtree(args.save_path + '/code')
    # shutil.copytree('.', args.save_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    writer = SummaryWriter(args.save_path + '/log')

    dataset_zoo = {'fundus': Fundus_Multi, 'prostate': Prostate_Multi}
    transform = {'fundus': Compose([trans.Resize((256, 256)), trans.RandomScaleCrop((256, 256)), trans.Normalize()]),
                 'prostate': Compose([np_trans.CreateOnehotLabel(args.num_classes)])}

    domain_idx_list = args.domain_idxs.split(',')
    domain_idx_list = [int(item) for item in domain_idx_list]

    trainset = dataset_zoo[args.dataset](base_dir=data_root, split='train',
                                         domain_idx_list=domain_idx_list, transform=transform[args.dataset])

    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=8,
                             shuffle=True, drop_last=True, pin_memory=True)

    model = Unet2D(num_classes=args.num_classes)
    optimizer = Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    print('\nParams: %.1fM' % count_params(model))

    if args.dataset == 'fundus':
        train_fundus(trainloader, model, writer, args, optimizer)
    elif args.dataset == 'prostate':
        train_prostate(trainloader, model, writer, args, optimizer)
    else:
        raise ValueError('Not support Dataset {}'.format(args.dataset))


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.epochs is None:
        args.epochs = {'fundus': 300, 'prostate': 160}[args.dataset]
    if args.lr is None:
        args.lr = {'fundus': 1e-3, 'prostate': 1e-3}[args.dataset]
    if args.num_classes is None:
        args.num_classes = {'fundus': 2, 'prostate': 2}[args.dataset]

    print(args)

    main(args)
