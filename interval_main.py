"""Training script for interval change prediction using AGXNet_Siamese model."""
import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
import os
import json
import shutil
import time
from enum import Enum
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tensorboard_logger import configure, log_value
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from scipy import interp
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer


import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataloader.dataset_interval import MIMICCXRInterval
from models.AGXNet_Saimese import AGXNet_Siamese
from utils import im2double, show_cam_on_image, BoundingBoxGenerator


# Define arguments used in the program.
parser = argparse.ArgumentParser(description='CXR Interval Change Prediction')

# Dataset
parser.add_argument('--img-chexpert-file', metavar='PATH',
                    default='./preprocessing/mimic-cxr-chexpert.csv',
                    help='master table including the image path and chexpert labels.')
parser.add_argument('--sids-file', metavar='PATH', default='./preprocessing/landmark_observation_sids.npy',
                    help='Array of study ids')
parser.add_argument('--adj-mtx-file', metavar='PATH', default='./preprocessing/landmark_observation_adj_mtx.npy',
                    help='Array of adjacency matrix corresponding to sids in the sids-file')
parser.add_argument('--prognosis-dict-file', metavar='PATH', default='./preprocessing/prognosis_dict.npy',
                    help='Array of prognosis dictionary')
parser.add_argument('--sequence-file', metavar='PATH', default='./preprocessing/mimic-cxr-seq.csv',
                    help='Sequence of images of each subject')
parser.add_argument('--imagenome-bounding-box-file', metavar='PATH', default='./preprocessing/imagenome_bbox.pkl',
                    help='ImaGenome bounding boxes for 21 landmarks.')
parser.add_argument('--imagenome-radgraph-landmark-mapping-file', metavar='PATH', default='./preprocessing/landmark_mapping.json',
                    help='Landmark mapping between ImaGenome and RadGraph.')

# prognosis terms
parser.add_argument('--new-words', nargs='+', default=['new', 'developing', 'onset'])
parser.add_argument('--worsened-words', nargs='+', default=['increased', 'increase', 'increasing', 'worsening',
                                                            'worsened', 'progression', 'progressed',
                                                            'progressive', 'development'])
parser.add_argument('--unchanged-words', nargs='+', default=['unchanged'])
parser.add_argument('--improved-words', nargs='+', default=['reduced', 'decreased', 'decrease', 'decreasing',
                                                            'improved', 'improvement', 'improving', 'resolved'])

# landmark terms
parser.add_argument('--full-anatomy-names', nargs='+', default=['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other'])
parser.add_argument('--landmark-names-spec', nargs='+', default=['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
'cardiophrenic_sulcus', 'mediastinal', 'spine', 'rib', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
'interstitium', 'parenchymal', 'cavoatrial_junction', 'stomach', 'clavicle'])
parser.add_argument('--landmark-names-unspec', nargs='+', default=['cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other'])

# observation terms
parser.add_argument('--full-obs', nargs='+', default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
'tail_abnorm_obs', 'excluded_obs'])
parser.add_argument('--norm-obs', nargs='+', default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free', 'expand', 'hyperinflate'])
parser.add_argument('--abnorm-obs', nargs='+', default=['effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
'device', 'engorgement', 'picc', 'clip', 'elevation', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration'])
parser.add_argument('--tail-abnorm-obs', nargs='+', default=['tail_abnorm_obs'])
parser.add_argument('--excluded-obs', nargs='+', default=['excluded_obs'])
parser.add_argument('--selected-obs', default='pneumothorax')

# model
parser.add_argument('--ckpt-dir', metavar='PATH', default='./checkpoints',
                    help='Checkpoint directory')
parser.add_argument('--ckpt-name', metavar='PATH', default='model_best.pth.tar',
                    help='Checkpoint directory')
parser.add_argument('--gloria-ckpt-dir', metavar='PATH', default='PATH_TO_GLOIRA_CHECKPOINT',
                    help='GLoRIA checkpoint directory')
parser.add_argument('--gloria-ckpt-name', metavar='PATH', default='chexpert_densenet121.ckpt',
                    help='GLoRIA checkpoint name')
parser.add_argument('--gloria-mimic-ckpt-dir', metavar='PATH', default='PATH_TO_GLOIRA_MIMIC_CHECKPOINT',
                    help='GLoRIA-MIMIC checkpoint directory')
parser.add_argument('--gloria-mimic-ckpt-name', metavar='PATH', default='gloria_seed_0.ckpt',
                    help='GLoRIA-MIMIC checkpoint name')
parser.add_argument('--convirt-ckpt-dir', metavar='PATH', default='PATH_TO_CONVIRT_CHECKPOINT',
                    help='ConVIRT checkpoint directory')
parser.add_argument('--convirt-ckpt-name', metavar='PATH', default='convirt_encoder.pth',
                    help='ConVIRT checkpoint name')
parser.add_argument('--biovil-ckpt-dir', metavar='PATH', default='PATH_TO_BIOVIL_CHECKPOINT',
                    help='BioVIL checkpoint directory')
parser.add_argument('--biovil-ckpt-name', metavar='PATH', default='biovil_image_resnet50_proj_size_128.pt',
                    help='BioVIL checkpoint name')
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',
                    help='PyTorch image models')
parser.add_argument('--freeze_net1', default='T',
                    help='whether or not to freeze the anatomy network. T=True, F=False')
parser.add_argument('--loss-type', default='CrossEntropy',
                    help='Loss type.')
parser.add_argument('--pretrained-type', default='AGXNet',
                    help='Pretrained model type, which can take value from [Random, ImageNet, ConVIRT, GLoRIA, GLoRIA_MIMIC, BioVIL, AGXNet]')
parser.add_argument('--anatomy-attention-type', default='Residual',
                    help='Anatomy attention type, which can take value from [None, Mask, Residual]')
parser.add_argument('--cam-norm-type', default='indep',
                    help='CAM1 normalization method, which can take value from [indep, dep]')

# experiment parameters
parser.add_argument('--exp-dir', metavar='DIR', default='./experiments/debug',
                    help='experiment directory')
parser.add_argument('--max-interval-days', default=365, type=int,
                    help='Maximum interval days between two scans')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resize', default=512, type=int,
                    help='input image resize')
parser.add_argument('--epsilon', default=0.5, type=float,
                    help='scaling weight of CAM1')
parser.add_argument('--frac', default=1.0, type=float,
                    help='fraction of random samples')
parser.add_argument('--seed', default=2, type=int,
                    help='Random seed that controls data sampling.')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

best_auc = 0
best_epoch = 0
class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def main():
    global best_auc, best_epoch

    args = parser.parse_args()

    # create experiment directory
    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    # save args to a dictionary
    with open(os.path.join(args.exp_dir, 'configs.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    f.close()

    # create tensorboard
    configure(args.exp_dir)

    # load state_dict of pretrained AGXNet
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location='cpu')  # load on cpu to avoid GPU RAM surge
    agxnet_state_dict = checkpoint['state_dict']

    # load state_dict of GLoRIA pretrained model
    pretrained_state_dict = None
    if args.pretrained_type == 'GLoRIA':
        gloria_ckpt_path = os.path.join(args.gloria_ckpt_dir, args.gloria_ckpt_name)
        gloria_checkpoint = torch.load(gloria_ckpt_path, map_location='cpu')
        gloria_state_dict = gloria_checkpoint['state_dict']
        pretrained_state_dict = gloria_state_dict

    if args.pretrained_type == 'GLoRIA_MIMIC':
        gloria_mimic_ckpt_name = 'seed' + str(args.seed) + '/last.ckpt'
        gloria_mimic_ckpt_path = os.path.join(args.gloria_mimic_ckpt_dir, gloria_mimic_ckpt_name)
        gloria_mimic_checkpoint = torch.load(gloria_mimic_ckpt_path, map_location='cpu')
        gloria_mimic_state_dict = gloria_mimic_checkpoint['state_dict']
        pretrained_state_dict = gloria_mimic_state_dict

    # load state_dict of ConVIRT pretrained model
    if args.pretrained_type == 'ConVIRT':
        convirt_ckpt_name = 'seed' + str(args.seed) + '/checkpoints/encoder_model.pth'
        convirt_ckpt_path = os.path.join(args.convirt_ckpt_dir, convirt_ckpt_name)
        convirt_state_dict = torch.load(convirt_ckpt_path, map_location='cpu')
        pretrained_state_dict = convirt_state_dict

    # load state_dict of  BioVIL pretrained model
    if args.pretrained_type == 'BioVIL':
        biovil_ckpt_path = os.path.join(args.biovil_ckpt_dir, args.biovil_ckpt_name)
        biovil_state_dict = torch.load(biovil_ckpt_path, map_location='cpu')
        pretrained_state_dict = biovil_state_dict


    # initialize model
    model = AGXNet_Siamese(args, agxnet_state_dict, pretrained_state_dict)
    model.cuda()

    # Set optimizer.
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Prepare dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = MIMICCXRInterval(args=args,
                                     mode='train',
                                     transform=transforms.Compose([
                                        transforms.Resize(args.resize),
                                        transforms.CenterCrop(args.resize),
                                        transforms.ToTensor(),  # convert pixel value to [0, 1]
                                        normalize
                                     ]))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    valid_dataset = MIMICCXRInterval(args=args,
                                     mode='valid',
                                     transform=transforms.Compose([
                                        transforms.Resize(args.resize),
                                        transforms.CenterCrop(args.resize),
                                        transforms.ToTensor(),  # convert pixel value to [0, 1]
                                        normalize
                                     ]))
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    test_dataset = MIMICCXRInterval(args=args,
                                     mode='test',
                                     transform=transforms.Compose([
                                        transforms.Resize(args.resize),
                                        transforms.CenterCrop(args.resize),
                                        transforms.ToTensor(),  # convert pixel value to [0, 1]
                                        normalize
                                     ]))

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    res_lst = []
    for epoch in range(args.start_epoch, args.epochs):
        # train one epoch
        train(train_loader, model, optimizer, epoch, args)

        # evaluate on validation set
        dict_res = validate(valid_loader, model, epoch, args, 'valid')

        # evaluate on validation set
        test_res = validate(test_loader, model, epoch, args, 'test')
        res_lst.append(test_res)

        # update learning rate
        scheduler.step()

        # remember best acc@1 and save checkpoint
        macro_auc = dict_res['macro_auc']
        is_best = macro_auc > best_auc
        if is_best:
            best_epoch = epoch
        best_auc = max(macro_auc, best_auc)

        # save checkpoint of the best model on validate dataset
        filename = os.path.join(args.exp_dir, 'model_epoch_latest.pth.tar')
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, filename)

    # save results to a json file
    with open(os.path.join(args.exp_dir, 'results.json'), 'w') as f:
        json.dump(res_lst[best_epoch], f)


def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to training mode
    if args.freeze_net1 == 'T':
        model.net1.eval()  # freeze dropout, batchnorm layers
        model.fc1.eval()
        model.net2.train()
        model.dense.train()
        model.cls.train()
    if args.freeze_net1 == 'F':
        model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        did_x, did_y, sid_x, sid_y, img_pth_x, img_pth_y, image_x, image_y, interval_hours, landmark_idx, target, weight, landmark_bbox_x, landmark_bbox_y = data
        image_x = image_x.cuda()
        image_y = image_y.cuda()
        landmark_idx = landmark_idx.cuda()
        target = target.cuda()

        logit = model(image_x, image_y, landmark_idx)
        inverse_weights = train_loader.dataset.prognosis_label_weights.cuda()
        criterion = nn.CrossEntropyLoss(weight=inverse_weights)
        loss = criterion(logit, target)

        # measure accuracy and record loss
        acc1 = accuracy(logit, target, topk=(1,))
        losses.update(loss.item(), image_x.size(0))
        top1.update(acc1[0].item(), image_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            step = i + len(train_loader) * epoch
            progress.display(i + 1)
            log_value('train/epoch', epoch, step)
            log_value('train/loss', progress.meters[2].avg, step)


def validate(valid_loader, model, epoch, args, mode):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(valid_loader),
        [batch_time, losses, top1],
        prefix=mode + ': ')

    # switch to evaluate mode
    model.eval()

    predictions = []
    softmax = []
    targets = []
    # set require_grad = False in all layers

    end = time.time()
    for i, data in enumerate(valid_loader):
        did_x, did_y, sid_x, sid_y, img_pth_x, img_pth_y, image_x, image_y, interval_hours, landmark_idx, target, weight, landmark_bbox_x, landmark_bbox_y = data
        image_x = image_x.cuda()
        image_y = image_y.cuda()
        landmark_idx = landmark_idx.cuda()
        target = target.cuda()

        #GradCAM
        model.gradients_x = None
        model.gradients_y = None
        logit = model(image_x, image_y, landmark_idx)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logit, target)

        # measure accuracy and record loss
        acc1 = accuracy(logit, target, topk=(1,))
        losses.update(loss.item(), image_x.size(0))
        top1.update(acc1[0].item(), image_x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # prepare for confusion matrix
        _, pred = logit.topk(1, 1, True, True)
        softmax.append(F.softmax(logit).detach().cpu().numpy())
        predictions.append(pred.t().squeeze().detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())

        if i % (args.print_freq) == 0:
            progress.display(i + 1)
            # visualize a random disease index
            if int(target.detach().cpu().numpy()) == 3:
               rdn_idx = random.randrange(len(did_x))
               visualization(image_x[rdn_idx], image_y[rdn_idx], img_pth_x[rdn_idx], img_pth_y[rdn_idx],
                             did_x[rdn_idx], did_y[rdn_idx], sid_x[rdn_idx], sid_y[rdn_idx], landmark_idx[rdn_idx], landmark_bbox_x[rdn_idx], landmark_bbox_y[rdn_idx],
                             interval_hours[rdn_idx], logit[rdn_idx], target[rdn_idx], model, epoch, args)

    # update progress bar
    progress.display_summary()

    # convert to 1D array
    predictions_arr = np.array(predictions)
    softmax_arr = np.concatenate(softmax)  # N * 4
    targets_arr = np.concatenate(targets)

    # print confusion matrix
    cm = confusion_matrix(targets_arr, predictions_arr)
    print(mode + ' confusion Matrix: ')
    print(cm)

    df_macro_auc_report = class_report(y_true=targets_arr, y_pred=predictions_arr, y_score=softmax_arr)
    print(df_macro_auc_report)

    # per class F1 scores
    dict_res = {}
    dict_res['f1_improved'] = df_macro_auc_report.iloc[0, 2]
    dict_res['f1_unchanged'] = df_macro_auc_report.iloc[1, 2]
    dict_res['f1_worsened'] = df_macro_auc_report.iloc[2, 2]
    dict_res['f1_new'] = df_macro_auc_report.iloc[3, 2]

    dict_res['auc_improved'] = df_macro_auc_report.iloc[0, 4]
    dict_res['auc_unchanged'] = df_macro_auc_report.iloc[1, 4]
    dict_res['auc_worsened'] = df_macro_auc_report.iloc[2, 4]
    dict_res['auc_new'] = df_macro_auc_report.iloc[3, 4]

    dict_res['micro_auc'] = df_macro_auc_report.iloc[4, 4]
    dict_res['macro_auc'] = df_macro_auc_report.iloc[4, 5]

    # update tensorboard
    log_value(mode + '/loss', progress.meters[1].avg, epoch)
    log_value(mode + '/accuracy', top1.avg, epoch)
    log_value(mode + '/f1_improved', dict_res['f1_improved'], epoch)
    log_value(mode + '/f1_unchanged', dict_res['f1_unchanged'], epoch)
    log_value(mode + '/f1_worsened', dict_res['f1_worsened'], epoch)
    log_value(mode + '/f1_new', dict_res['f1_new'], epoch)
    log_value(mode + '/auc_improved', dict_res['auc_improved'], epoch)
    log_value(mode + '/auc_unchanged', dict_res['auc_unchanged'], epoch)
    log_value(mode + '/auc_worsened', dict_res['auc_worsened'], epoch)
    log_value(mode + '/auc_new', dict_res['auc_new'], epoch)
    log_value(mode + '/micro_auc', dict_res['micro_auc'], epoch)
    log_value(mode + '/macro_auc', dict_res['macro_auc'], epoch)

    return dict_res


def save_checkpoint(args, state, is_best, filename):
    torch.save(state, os.path.join(args.exp_dir, filename))
    if is_best:
        ckpt_name = 'model_best.pth.tar'
        shutil.copyfile(filename, os.path.join(args.exp_dir, ckpt_name))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def class_report(y_true, y_pred, y_score=None):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt_true = np.unique(
        y_true,
        return_counts=True
    )
    n_classes = len(labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum()
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int),
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        # compute micro auc
        if n_classes <= 2:
            fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                lb.transform(y_true).ravel(),
                y_score[:, 1].ravel())
        else:
            fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score.ravel())

        roc_auc["avg / total"] = auc(
            fpr["avg / total"],
            tpr["avg / total"])

        class_report_df['MICRO_AUC'] = pd.Series(roc_auc)

        # compute macro auc
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([
            fpr[i] for i in labels]
        ))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in labels:
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr

        roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['MACRO_AUC'] = pd.Series(roc_auc)

    return class_report_df


def visualization(image_x, image_y, img_pth_x, img_pth_y, did_x, did_y, sid_x, sid_y,
                  landmark_idx, landmark_bbox_x, landmark_bbox_y, interval_hours, logit, target, model, epoch, args):
    subdir = args.exp_dir + '/plots' + '/epoch_' + str(epoch) + '/'
    Path(subdir).mkdir(parents=True, exist_ok=True)
    df_text = pd.read_csv('./preprocessing/mimic-cxr-text.csv')

    # print text
    try:
        report_y = df_text[df_text['study_id'] == sid_y.item()].iloc[0, -1]
    except:
        report_y = ''
    try:
        report_x = df_text[df_text['study_id'] == sid_x.item()].iloc[0, -1]
    except:
        report_x = ''
    filename = subdir + did_x + '.txt'
    original_stdout = sys.stdout
    if args.loss_type == 'CrossEntropy':
        pred_str = 'Softmax: ' + str(F.softmax(logit))
    txt = 'Interval hours = ' + str(round(interval_hours.item(), 1)) + '\n' + 'True label: ' + str(target.item()) + '\n' \
          + pred_str + '\n' + 'study id: ' + str(sid_y.item()) \
          + report_y + '\n' + 'study id: ' + str(sid_x.item()) + report_x
    with open(filename, 'w') as f:
        sys.stdout = f
        print(txt)
        sys.stdout = original_stdout

    # Compute GradCAM
    p_ic = F.softmax(logit).detach().cpu().numpy()
    true_label = int(target.cpu().detach())
    logit[true_label].backward()
    gradients_x = model.get_activations_gradient_x()
    gradients_y = model.get_activations_gradient_y()
    pooled_gradients_x = torch.mean(gradients_x, dim=[0, 2, 3])
    pooled_gradients_y = torch.mean(gradients_y, dim=[0, 2, 3])
    activations_x = model.get_activations_x(image_x.unsqueeze(0))
    activations_y = model.get_activations_y(image_y.unsqueeze(0))
    for c in range(len(pooled_gradients_x)):
        activations_x[:, c, :, :] *= pooled_gradients_x[c]
    for c in range(len(pooled_gradients_y)):
        activations_y[:, c, :, :] *= pooled_gradients_y[c]
    cam2_x = F.relu(torch.mean(activations_x, dim=1).squeeze())
    cam2_y = F.relu(torch.mean(activations_y, dim=1).squeeze())
    cam2_norm_x = cam2_x / torch.max(cam2_x)
    cam2_norm_y = cam2_y / torch.max(cam2_y)
    cam2_norm_x_resize = im2double(cv2.resize(cam2_norm_x.detach().cpu().numpy(), (args.resize, args.resize)))
    cam2_norm_y_resize = im2double(cv2.resize(cam2_norm_y.detach().cpu().numpy(), (args.resize, args.resize)))

    # print original images
    np_transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.resize),
        lambda x: np.float32(x) / 255
    ])
    landmark = args.landmark_names_spec[landmark_idx.item()]
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), dpi=100)
    # previous CXR
    ax1 = axs[0, 0]
    img_y = Image.open(img_pth_y).convert('RGB')
    img_y_np = np_transform(img_y)
    ax1.imshow(img_y_np, cmap='gray')
    t1 = did_y
    ax1.set_title(t1)

    # current CXR
    ax2 = axs[1, 0]
    img_x = Image.open(img_pth_x).convert('RGB')
    img_x_np = np_transform(img_x)
    ax2.imshow(img_x_np, cmap='gray')
    t2 = did_x
    ax2.set_title(t2)

    # previous CXR heatmap
    with torch.no_grad():
        f1_y = model.net1(image_y.unsqueeze(0))[-1]
        f1_y_p = model.pool(f1_y)
        logit1_y = model.fc1(f1_y_p.squeeze())  # b * a
        p1_y = torch.sigmoid(logit1_y).cpu().detach().numpy()
        cam1_y = torch.einsum('bchw, ac -> bahw', f1_y, model.fc1.weight)  # b * a * h * w

        f1_x = model.net1(image_x.unsqueeze(0))[-1]
        f1_x_p = model.pool(f1_x)
        logit1_x = model.fc1(f1_x_p.squeeze())  # b * a
        p1_x = torch.sigmoid(logit1_x).cpu().detach().numpy()
        cam1_x = torch.einsum('bchw, ac -> bahw', f1_x, model.fc1.weight)  # b * a * h * w

        if args.cam_norm_type == 'indep':
            cam1_norm_y = model.normalize_cam1(cam1_y)
            cam1_norm_x = model.normalize_cam1(cam1_x)
        elif args.cam_norm_type == 'dep':
            cam1_norm_x, cam1_norm_y = model.normalize_cams(cam1_x, cam1_y)
        else:
            raise ValueError('invalid cam normalization type %r' % args.cam_norm_type)

        cam1_sel_y = cam1_norm_y[0][landmark_idx]
        cam1_sel_y = cam1_sel_y.detach().cpu().numpy()
        cam1_sel_y_resize = im2double(cv2.resize(cam1_sel_y, (args.resize, args.resize)))

        cam1_sel_x = cam1_norm_x[0][landmark_idx]
        cam1_sel_x = cam1_sel_x.detach().cpu().numpy()
        cam1_sel_x_resize = im2double(cv2.resize(cam1_sel_x, (args.resize, args.resize)))

    ax3 = axs[0, 1]
    vis_y = show_cam_on_image(img_y_np, cam1_sel_y_resize, use_rgb=True)
    ax3.imshow(vis_y)
    # add ImaGenome bbox
    b = landmark_bbox_y.detach().cpu().numpy()
    rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='lime',
                             facecolor='none')
    ax3.add_patch(rect)
    t3 = landmark + ': ' + str(round(p1_y[landmark_idx.item()], 2))
    ax3.set_title(t3)

    ax4 = axs[1, 1]
    vis_x = show_cam_on_image(img_x_np, cam1_sel_x_resize, use_rgb=True)
    ax4.imshow(vis_x)
    # add ImaGenome bbox
    b = landmark_bbox_x.detach().cpu().numpy()
    rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='lime',
                             facecolor='none')
    ax4.add_patch(rect)
    t4 = landmark + ': ' + str(round(p1_x[landmark_idx.item()], 2))
    ax4.set_title(t4)

    ax5 = axs[0, 2]
    vis_y = show_cam_on_image(img_y_np, cam2_norm_y_resize, use_rgb=True)
    ax5.imshow(vis_y)
    # add ImaGenome bbox
    b = landmark_bbox_y.detach().cpu().numpy()
    rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='lime',
                             facecolor='none')
    ax5.add_patch(rect)
    t5 = 'IC Target =' + str(true_label) + ', Prediction ='+ str(round(p_ic[true_label], 2))
    ax5.set_title(t5)

    ax6 = axs[1, 2]
    vis_x = show_cam_on_image(img_x_np, cam2_norm_x_resize, use_rgb=True)
    ax6.imshow(vis_x)
    # add ImaGenome bbox
    b = landmark_bbox_x.detach().cpu().numpy()
    rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='lime',
                             facecolor='none')
    ax6.add_patch(rect)
    t6 = 'IC Target =' + str(true_label) + ', Prediction ='+ str(round(p_ic[true_label], 2))
    ax6.set_title(t6)

    filename = subdir + did_x + '.png'
    plt.savefig(filename)


def eval_gradcam_bbox(image_x, image_y, landmark_bbox_x, landmark_bbox_y, logit, target, model, epoch, args):

    # Compute GradCAM
    true_label = int(target.cpu().detach())
    logit[true_label].backward()
    gradients_x = model.get_activations_gradient_x()
    gradients_y = model.get_activations_gradient_y()
    pooled_gradients_x = torch.mean(gradients_x, dim=[0, 2, 3])
    pooled_gradients_y = torch.mean(gradients_y, dim=[0, 2, 3])
    activations_x = model.get_activations_x(image_x.unsqueeze(0))
    activations_y = model.get_activations_y(image_y.unsqueeze(0))
    for c in range(len(pooled_gradients_x)):
        activations_x[:, c, :, :] *= pooled_gradients_x[c]
    for c in range(len(pooled_gradients_y)):
        activations_y[:, c, :, :] *= pooled_gradients_y[c]
    cam2_x = F.relu(torch.mean(activations_x, dim=1).squeeze())
    cam2_y = F.relu(torch.mean(activations_y, dim=1).squeeze())
    cam2_norm_x = cam2_x / torch.max(cam2_x)
    cam2_norm_y = cam2_y / torch.max(cam2_y)
    cam2_norm_x_resize = im2double(cv2.resize(cam2_norm_x.detach().cpu().numpy(), (args.resize, args.resize)))
    cam2_norm_y_resize = im2double(cv2.resize(cam2_norm_y.detach().cpu().numpy(), (args.resize, args.resize)))



if __name__ == '__main__':
    main()