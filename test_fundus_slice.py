#!/usr/bin/env python
import argparse
import os
import os.path as osp
import sys

import torch.nn as nn
from torchvision import transforms

from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

import torch
from tqdm import tqdm
from dataset.fundus import Fundus
from dataset.prostate import Prostate
from torch.utils.data import DataLoader
import dataset.transform as trans
from torchvision.transforms import Compose
from utils.metrics import *
from datetime import datetime
from dataset import utils
from utils.utils import postprocessing, save_per_img
from test_utils import *
# from models.model_zoo import get_model
from networks.deeplabv3_single import Deeplabv3p
from networks.unet import Unet2D, UnetDecoder
from model.model import PrivateDecoder
import numpy as np
from medpy.metric import binary
from torch.nn import DataParallel
import torch
from PIL import Image
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

unloader = transforms.ToPILImage()


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def save_image(tensor, name, **para):
    dir = './trans_images'
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    if not osp.exists(dir):
        os.makedirs(dir)
    image.save(name + '.png')


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    """1.论文里的公式2，求出B的rw,rh"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    """2.论文里的公式2，求出B的rx,ry（bbox的中心点）"""
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # 限制坐标区域不超过样本大小

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    """3.返回剪裁B区域的坐标值"""


def parse_args():
    parser = argparse.ArgumentParser(description='Test on Fundus dataset (2D slice)')
    # basic settings
    parser.add_argument('--model_file', type=str, default='./out/adv0123/',
                        help='Model path')
    parser.add_argument('--dataset', type=str, default='fundus', help='training dataset')
    parser.add_argument('--data_dir', default='/data/yhz', help='data root path')
    parser.add_argument('--datasetTest', type=int, default=3, help='test folder id contain images ROIs to test')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of testing')
    parser.add_argument('--image_size', type=int, default=800, help='cropping size of training samples')
    parser.add_argument('--backbone', type=str, default='mobilenet', help='backbone of semantic segmentation model')
    parser.add_argument('--model', type=str, default='unet', help='head of semantic segmentation model')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--test_prediction_save_path', type=str, default='./prediction',
                        help='Path root for test image and mask')
    parser.add_argument('--save_result', default= True, action='store_true', help='Save Results')
    parser.add_argument('--freeze_bn', action='store_true', help='Freeze Batch Normalization')
    parser.add_argument('--output_stride', type=int, default=16, help='output stride of deeplab')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    args = parser.parse_args()
    return args


def main(args):
    data_dir = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(args.test_prediction_save_path):
        os.makedirs(args.test_prediction_save_path)

    model_file = args.model_file
    output_path = os.path.join(args.test_prediction_save_path, 'test' + str(args.datasetTest))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataset_zoo = {'fundus': Fundus, 'prostate': Prostate}
    transform = {'fundus': Compose([trans.Resize((256, 256)), trans.Normalize()]), 'prostate': None}

    testset = dataset_zoo[args.dataset](base_dir=data_dir, split='test',
                                        domain_idx=args.datasetTest, transform=transform[args.dataset])

    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=8,
                            shuffle=False, drop_last=False, pin_memory=True)
    if not args.model == 'unet':
        model = Deeplabv3p(backbone=args.backbone, output_stride=args.output_stride, num_classes=args.num_classes,
                           pretrained=False)
    else:
        model = Unet2D(num_classes=args.num_classes)
    model.load_state_dict(torch.load(model_file + 'final_model.pth'))
    model = DataParallel(model).cuda()
    # dec0 = PrivateDecoder(256,3)
    # dec0.load_state_dict(torch.load(model_file + 'dec1_8000.pth'))
    # dec0 = DataParallel(dec0).cuda()
    if not args.freeze_bn:
        model.eval()
        for m in model.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.train()
            elif isinstance(m, nn.BatchNorm2d):
                m.train()
    else:
        model.eval()

    # dec0.eval()
    val_cup_dice = 0.0
    val_disc_dice = 0.0
    total_hd_OC = 0.0
    total_hd_OD = 0.0
    total_asd_OC = 0.0
    total_asd_OD = 0.0
    total_num = 0
    OC = []
    OD = []

    tbar = tqdm(testloader, ncols=150)

    with torch.no_grad():
        for batch_idx, (data, target, target_orgin, ids) in enumerate(tbar):
            data, target = data.cuda(), target.cuda()

            _, prediction, codecommon = model(data)
            # _,tans = dec0(codecommon)
            prediction = torch.nn.functional.interpolate(prediction,
                                                         size=(target_orgin.size()[2], target_orgin.size()[3]),
                                                         mode="bilinear")
            data = torch.nn.functional.interpolate(data, size=(target_orgin.size()[2], target_orgin.size()[3]),
                                                   mode="bilinear")
            # tans = torch.nn.functional.interpolate(tans, size=(target_orgin.size()[2], target_orgin.size()[3]),
            #                                        mode="bilinear")

            # save_image(data[1],'1')
            # save_image(tans[1],'2')
            # sys.exit(0)
            target_numpy = target_orgin.data.cpu().numpy()
            imgs = data.data.cpu().numpy()
            # timgs = tans.data.cpu().numpy()

            hd_OC = 100
            asd_OC = 100
            hd_OD = 100
            asd_OD = 100

            for i in range(prediction.shape[0]):
                prediction_post = postprocessing(prediction[i], dataset=args.dataset, threshold=0.75)
                cup_dice, disc_dice = dice_coeff_2label(prediction_post, target_orgin[i])
                OC.append(cup_dice)
                OD.append(disc_dice)
                if np.sum(prediction_post[0, ...]) < 1e-4:
                    hd_OC = 100
                    asd_OC = 100
                else:
                    hd_OC = binary.hd95(np.asarray(prediction_post[0, ...], dtype=np.bool),
                                        np.asarray(target_numpy[i, 0, ...], dtype=np.bool))
                    asd_OC = binary.asd(np.asarray(prediction_post[0, ...], dtype=np.bool),
                                        np.asarray(target_numpy[i, 0, ...], dtype=np.bool))
                if np.sum(prediction_post[1, ...]) < 1e-4:
                    hd_OD = 100
                    asd_OD = 100
                else:
                    hd_OD = binary.hd95(np.asarray(prediction_post[1, ...], dtype=np.bool),
                                        np.asarray(target_numpy[i, 1, ...], dtype=np.bool))

                    asd_OD = binary.asd(np.asarray(prediction_post[1, ...], dtype=np.bool),
                                        np.asarray(target_numpy[i, 1, ...], dtype=np.bool))
                val_cup_dice += cup_dice
                val_disc_dice += disc_dice
                total_hd_OC += hd_OC
                total_hd_OD += hd_OD
                total_asd_OC += asd_OC
                total_asd_OD += asd_OD
                total_num += 1
                if args.save_result:
                    for img, lt, lp in zip([imgs[i]], [target_numpy[i]], [prediction_post]):
                        img, lt = utils.untransform(img, lt)
                        save_per_img(img.transpose(1, 2, 0),
                                     output_path,
                                     ids[i],
                                     lp, lt, mask_path=None, ext="bmp")
                    # for img, lt, lp in zip([timgs[i]], [target_numpy[i]], [prediction_post]):
                    #     img, lt = utils.untransform(img, lt)
                    #     save_per_img(img.transpose(1, 2, 0),
                    #                  output_path+'t',
                    #                  ids[i],
                    #                  lp, lt, mask_path=None, ext="bmp")

        val_cup_dice /= total_num
        val_disc_dice /= total_num
        total_hd_OC /= total_num
        total_asd_OC /= total_num
        total_hd_OD /= total_num
        total_asd_OD /= total_num

        print('''\n==>val_cup_dice : {0}'''.format(val_cup_dice))
        print('''\n==>val_disc_dice : {0}'''.format(val_disc_dice))
        print('''\n==>average_hd_OC : {0}'''.format(total_hd_OC))
        print('''\n==>average_hd_OD : {0}'''.format(total_hd_OD))
        print('''\n==>average_asd_OC : {0}'''.format(total_asd_OC))
        print('''\n==>average_asd_OD : {0}'''.format(total_asd_OD))
        with open(osp.join(output_path, '../test' + str(args.datasetTest) + '_log.csv'), 'a') as f:
            log = [['batch-size: '] + [args.batch_size] + [args.model_file] + \
                   ['cup dice coefficence: '] + [val_cup_dice] + \
                   ['disc dice coefficence: '] + [val_disc_dice] + \
                   ['average_hd_OC: '] + [total_hd_OC] + \
                   ['average_hd_OD: '] + [total_hd_OD] + \
                   ['average_asd_OC: '] + [total_asd_OC] + \
                   ['average_asd_OD: '] + [total_asd_OD]]
            log = map(str, log)
            f.write(','.join(log) + '\n')


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)

    # elif args.dataset == 'prostate':
    #     val_dice = 0.0
    #     total_hd = 0.0
    #     total_asd = 0.0
    #     timestamp_start = datetime.now(pytz.timezone('Asia/Hong_Kong'))
    #     total_num = 0

    #     DICE = []

    #     tbar = tqdm(testloader, ncols=150)

    #     with torch.no_grad():
    #         for batch_idx, (data, target, ids) in enumerate(tbar):
    #             data, target = data.cuda(), target.cuda()

    #             prediction = torch.max(torch.softmax(model(data), dim=1), 1)[1]

    #             target_numpy = target.data.cpu().numpy()
    #             imgs = data.data.cpu().numpy()
    #             _prediction = prediction.data.cpu().numpy().copy()

    #             hd = 100
    #             asd = 100

    #             for i in range(prediction.shape[0]):
    #                 prediction_post = postprocessing(prediction[i], dataset=args.dataset)
    #                 _prediction[i] = prediction_post

    #                 dice_coeff = dice_coeff_1label(prediction_post, target[i])
    #                 DICE.append(dice_coeff)

    #                 if np.sum(prediction_post) < 1e-4:
    #                     hd = 100
    #                     asd = 100
    #                 else:
    #                     hd = binary.hd95(np.asarray(prediction_post, dtype=np.bool),
    #                                         np.asarray(target_numpy[i, ...], dtype=np.bool))
    #                     asd = binary.asd(np.asarray(prediction_post, dtype=np.bool),
    #                                         np.asarray(target_numpy[i, ...], dtype=np.bool))
    #                 val_dice += dice_coeff
    #                 total_hd += hd
    #                 total_asd += asd
    #                 total_num += 1
    #                 if args.save_result:
    #                     for img, lt, lp in zip([imgs[i]], [target_numpy[i]], [prediction_post]):
    #                         img, lt = utils.untransform_prostate(img[1, ...], lt)
    #                         img = np.repeat(np.expand_dims(img, axis=0), repeats=3, axis=0)
    #                         save_per_img_prostate(img.transpose(1, 2, 0),
    #                                     output_path,
    #                                     ids[i],
    #                                     lp, lt, mask_path=None, ext="bmp")

    # dice_coeff = dice_coeff_1label(_prediction, target)
    # DICE.append(dice_coeff)
    # if np.sum(_prediction) < 1e-4:
    #     hd = 100
    #     asd = 100
    # else:
    #     hd = binary.hd95(np.asarray(_prediction, dtype=np.bool),
    #                         np.asarray(target_numpy, dtype=np.bool))
    #     asd = binary.asd(np.asarray(_prediction, dtype=np.bool),
    #                         np.asarray(target_numpy, dtype=np.bool))
    # val_dice += dice_coeff
    # total_hd += hd
    # total_asd += asd
    # total_num += 1

    # val_dice /= total_num
    # total_hd /= total_num
    # total_asd /= total_num

    # print('''\n==>val_dice : {0}'''.format(val_dice))
    # print('''\n==>average_hd : {0}'''.format(total_hd))
    # print('''\n==>average_asd : {0}'''.format(total_asd))
    # with open(osp.join(output_path, '../test' + str(args.datasetTest) + '_log.csv'), 'a') as f:
    #     elapsed_time = (
    #             datetime.now(pytz.timezone('Asia/Hong_Kong')) -
    #             timestamp_start).total_seconds()
    #     log = [['batch-size: '] + [args.batch_size] + [args.model_file] + ['dice coefficence: '] + \
    #         [val_dice] + \
    #         ['average_hd: '] + \
    #         [total_hd] + \
    #         ['average_asd: '] + \
    #         [total_asd] + \
    #         [elapsed_time]]
    #     log = map(str, log)
    #     f.write(','.join(log) + '\n')
