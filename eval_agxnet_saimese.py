"""Evaluate AGXNet_Siamese model on the MIMIC-CXR dataset."""

import warnings

warnings.filterwarnings("ignore")

import os
import sys
import argparse
import time
from enum import Enum
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy import interp
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from dataloader.dataset_interval import MIMICCXRInterval
from models.AGXNet_Saimese import AGXNet_Siamese
from utils import im2double, show_cam_on_image, BoundingBoxGenerator, get_cumlative_attention, iou, iobb1, iobb2, \
    get_recall_precision, average_precision

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define arguments used in the program.
parser = argparse.ArgumentParser(description='CXR anatomy evaluation.')

# Dataset
parser.add_argument('--img-chexpert-file', metavar='PATH',
                    default='./preprocessing/mimic-cxr-chexpert.csv',
                    help='master table including the image path and chexpert labels.')
parser.add_argument('--sids-file', metavar='PATH',
                    default='./preprocessing/landmark_observation_sids.npy',
                    help='Array of study ids')
parser.add_argument('--adj-mtx-file', metavar='PATH',
                    default='./preprocessing/landmark_observation_adj_mtx.npy',
                    help='Array of adjacency matrix corresponding to sids in the sids-file')
parser.add_argument('--prognosis-dict-file', metavar='PATH',
                    default='./preprocessing/prognosis_dict.npy',
                    help='Array of prognosis dictionary')
parser.add_argument('--sequence-file', metavar='PATH',
                    default='./preprocessing/mimic-cxr-seq.csv',
                    help='Sequence of images of each subject')
parser.add_argument('--imagenome-bounding-box-file', metavar='PATH',
                    default='./preprocessing/imagenome_bbox.pkl',
                    help='ImaGenome bounding boxes for 21 landmarks.')
parser.add_argument('--imagenome-radgraph-landmark-mapping-file', metavar='PATH',
                    default='./preprocessing/landmark_mapping.json',
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
parser.add_argument('--full-anatomy-names', nargs='+',
                    default=['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
                             'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
                             'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe',
                             'upper_left_lobe',
                             'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung',
                             'left_mid_lung', 'left_upper_lung',
                             'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung',
                             'right_upper_lung', 'right_apical_lung',
                             'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic',
                             'right_costophrenic', 'costophrenic_unspec',
                             'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach',
                             'right_atrium', 'right_ventricle', 'aorta', 'svc',
                             'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary',
                             'lung_volumes', 'unspecified', 'other'])
parser.add_argument('--landmark-names-spec', nargs='+',
                    default=['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
                             'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
                             'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe',
                             'upper_left_lobe',
                             'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung',
                             'left_mid_lung', 'left_upper_lung',
                             'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung',
                             'right_upper_lung', 'right_apical_lung',
                             'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic',
                             'right_costophrenic', 'costophrenic_unspec',
                             'cardiophrenic_sulcus', 'mediastinal', 'spine', 'rib', 'right_atrium', 'right_ventricle',
                             'aorta', 'svc',
                             'interstitium', 'parenchymal', 'cavoatrial_junction', 'stomach', 'clavicle'])
parser.add_argument('--landmark-names-unspec', nargs='+',
                    default=['cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other'])

# observation terms
parser.add_argument('--full-obs', nargs='+',
                    default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
                             'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation',
                             'process', 'abnormality', 'enlarge', 'tip', 'low',
                             'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air',
                             'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                             'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid',
                             'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                             'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate',
                             'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
                             'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline',
                             'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
                             'tail_abnorm_obs', 'excluded_obs'])
parser.add_argument('--norm-obs', nargs='+',
                    default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
                             'expand', 'hyperinflate'])
parser.add_argument('--abnorm-obs', nargs='+',
                    default=['effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation',
                             'process', 'abnormality', 'enlarge', 'tip', 'low',
                             'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air',
                             'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                             'device', 'engorgement', 'picc', 'clip', 'elevation', 'nodule', 'wire', 'fluid',
                             'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                             'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd',
                             'infiltrate', 'obscure', 'deformity', 'hernia',
                             'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline',
                             'hardware', 'dilation', 'chf', 'redistribution', 'aspiration'])
parser.add_argument('--tail-abnorm-obs', nargs='+', default=['tail_abnorm_obs'])
parser.add_argument('--excluded-obs', nargs='+', default=['excluded_obs'])
parser.add_argument('--selected-obs', default='pneumothorax')

# model
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',
                    help='PyTorch image models')
parser.add_argument('--pretrained-type', default='AGXNet_Siamese',
                    help='Pretrained model type, which can only be AGXNet during evaluation time.')
parser.add_argument('--ckpt-name', metavar='PATH', default='model_best.pth.tar',
                    help='Checkpoint directory')
parser.add_argument('--freeze_net1', default='T',
                    help='whether or not to freeze the anatomy network. T=True, F=False')
parser.add_argument('--anatomy-attention-type', default='Residual',
                    help='Anatomy attention type, which can take value from [None, Mask, Residual]')
parser.add_argument('--cam-norm-type', default='indep',
                    help='CAM1 normalization method, which can take value from [indep, dep]')
parser.add_argument('--loss-type', default='CrossEntropy',
                    help='Loss type.')

# experiment parameters
parser.add_argument('--exp-dir', metavar='DIR',
                    default='./experiments/ckpt_6/seed_1/pneumothorax/100/AGXNet_Residual_T',
                    help='experiment directory')
parser.add_argument('--max-interval-days', default=365, type=int,
                    help='Maximum interval days between two scans')
parser.add_argument('-b', '--batch-size', default=1, type=int,
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
parser.add_argument('--epsilon', default=0.0, type=float,
                    help='scaling weight of CAM1')
parser.add_argument('--frac', default=1.0, type=float,
                    help='fraction of random samples')
parser.add_argument('--seed', default=1, type=int,
                    help='Random seed that controls data sampling.')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def main():
    args = parser.parse_args()

    # read saved configurations
    f = open(os.path.join(args.exp_dir, 'configs.json'))
    configs = Bunch(json.load(f))
    args.configs = configs

    # load state_dict of pretrained Saimese AGXNet
    ckpt_path = os.path.join(args.exp_dir, args.ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location='cpu')  # load on cpu to avoid GPU RAM surge
    agxnet_saimese_state_dict = checkpoint['state_dict']
    print('Best epoch: ', str(checkpoint['epoch'] - 1))

    # initialize model
    model = AGXNet_Siamese(args, agxnet_saimese_state_dict, agxnet_saimese_state_dict)

    # load weights
    model.load_state_dict(agxnet_saimese_state_dict)
    model.cuda()

    # test dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_dataset = MIMICCXRInterval(args=args,
                                    mode='test',
                                    transform=transforms.Compose([
                                        transforms.Resize(args.resize),
                                        transforms.CenterCrop(args.resize),
                                        transforms.ToTensor(),  # convert pixel value to [0, 1]
                                        normalize
                                    ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1],
        prefix='test' + ': ')

    # switch to evaluate mode
    model.eval()

    predictions = []
    softmax = []
    targets = []

    dicoms = []
    dicom_xs = []
    dicom_ys = []
    image_types = []
    landmark_names = []
    bb_landmark_gens = []
    bb_observation_gens = []
    bb_landmark_gts = []
    ic_labels = []

    end = time.time()
    cnt_new = cnt_worsened = cnt_unchanged = cnt_improved = 0
    for i, data in enumerate(test_loader):
        did_x, did_y, sid_x, sid_y, img_pth_x, img_pth_y, image_x, image_y, interval_hours, landmark_idx, target, weight, landmark_bbox_x, landmark_bbox_y = data
        image_x = image_x.cuda()
        image_y = image_y.cuda()
        landmark_idx = landmark_idx.cuda()
        target = target.cuda()

        model.gradients_x = None
        model.gradients_y = None
        logit = model(image_x, image_y, landmark_idx)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logit, target)

        # get net1 CAMs
        cam1_sel_x_resize, cam1_sel_y_resize = get_net1_cams(image_x, image_y, landmark_idx, model, args)
        # get net2 GradCAMs
        cam2_norm_x_resize, cam2_norm_y_resize = get_net2_gradcam(image_x, image_y, logit, target, model, args)

        # generate bbox
        bb_landmark_gens_x = generate_bbox(cam1_sel_x_resize)
        bb_landmark_gens_y = generate_bbox(cam1_sel_y_resize)
        bb_observation_gens_x = generate_bbox(cam2_norm_x_resize)
        bb_observation_gens_y = generate_bbox(cam2_norm_y_resize)

        # image x
        dicoms.append(did_x[0])
        dicom_xs.append(did_x[0])
        dicom_ys.append(did_y[0])
        image_types.append('x')
        landmark_names.append(args.landmark_names_spec[int(landmark_idx.detach().cpu())])
        ic_labels.append(int(target.detach().cpu()))
        bb_landmark_gts.append(list(landmark_bbox_x.detach().cpu().numpy()[0]))
        bb_landmark_gens.append(bb_landmark_gens_x)
        bb_observation_gens.append(bb_observation_gens_x)

        # image y
        dicoms.append(did_y[0])
        dicom_xs.append(did_x[0])
        dicom_ys.append(did_y[0])
        image_types.append('y')
        landmark_names.append(args.landmark_names_spec[int(landmark_idx.detach().cpu())])
        ic_labels.append(int(target.detach().cpu()))
        bb_landmark_gts.append(list(landmark_bbox_y.detach().cpu().numpy()[0]))
        bb_landmark_gens.append(bb_landmark_gens_y)
        bb_observation_gens.append(bb_observation_gens_y)

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

        progress.display(i + 1)
        # visualization 10 images for each class
        true_label = int(target.detach().cpu().numpy())
        if true_label == 0 and cnt_improved < 10:
            visualization(cam1_sel_x_resize, cam1_sel_y_resize, cam2_norm_x_resize, cam2_norm_y_resize,
                          image_x, image_y, img_pth_x[0], img_pth_y[0],
                          did_x[0], did_y[0], sid_x[0], sid_y[0], landmark_idx,
                          landmark_bbox_x[0], landmark_bbox_y[0],
                          interval_hours, logit, target, model, 'improved', args)
            cnt_improved += 1
        if true_label == 1 and cnt_unchanged < 10:
            visualization(cam1_sel_x_resize, cam1_sel_y_resize, cam2_norm_x_resize, cam2_norm_y_resize,
                          image_x, image_y, img_pth_x[0], img_pth_y[0],
                          did_x[0], did_y[0], sid_x[0], sid_y[0], landmark_idx,
                          landmark_bbox_x[0], landmark_bbox_y[0],
                          interval_hours, logit, target, model, 'unchanged', args)
            cnt_unchanged += 1
        if true_label == 2 and cnt_worsened < 10:
            visualization(cam1_sel_x_resize, cam1_sel_y_resize, cam2_norm_x_resize, cam2_norm_y_resize,
                          image_x, image_y, img_pth_x[0], img_pth_y[0],
                          did_x[0], did_y[0], sid_x[0], sid_y[0], landmark_idx,
                          landmark_bbox_x[0], landmark_bbox_y[0],
                          interval_hours, logit, target, model, 'worsened', args)
            cnt_worsened += 1
        if true_label == 3 and cnt_new < 10:
            visualization(cam1_sel_x_resize, cam1_sel_y_resize, cam2_norm_x_resize, cam2_norm_y_resize,
                          image_x, image_y, img_pth_x[0], img_pth_y[0],
                          did_x[0], did_y[0], sid_x[0], sid_y[0], landmark_idx,
                          landmark_bbox_x[0], landmark_bbox_y[0],
                          interval_hours, logit, target, model, 'new', args)
            cnt_new += 1


    # update progress bar
    progress.display_summary()

    # interval change classification results
    # convert to 1D array
    predictions_arr = np.array(predictions)
    softmax_arr = np.concatenate(softmax)
    targets_arr = np.concatenate(targets)

    # print confusion matrix
    cm = confusion_matrix(targets_arr, predictions_arr)
    print('test' + ' confusion Matrix: ')
    print(cm)

    df_macro_auc_report = class_report(y_true=targets_arr, y_pred=predictions_arr, y_score=softmax_arr)
    print(df_macro_auc_report)

    # per class F1 scores
    dict_cls = {}
    dict_cls['f1_improved'] = df_macro_auc_report.iloc[0, 2]
    dict_cls['f1_unchanged'] = df_macro_auc_report.iloc[1, 2]
    dict_cls['f1_worsened'] = df_macro_auc_report.iloc[2, 2]
    dict_cls['f1_new'] = df_macro_auc_report.iloc[3, 2]

    dict_cls['auc_improved'] = df_macro_auc_report.iloc[0, 4]
    dict_cls['auc_unchanged'] = df_macro_auc_report.iloc[1, 4]
    dict_cls['auc_worsened'] = df_macro_auc_report.iloc[2, 4]
    dict_cls['auc_new'] = df_macro_auc_report.iloc[3, 4]

    dict_cls['micro_auc'] = df_macro_auc_report.iloc[4, 4]
    dict_cls['macro_auc'] = df_macro_auc_report.iloc[4, 5]

    with open(os.path.join(args.exp_dir, 'results_without_cam.json'), 'w') as f:
        json.dump(dict_cls, f)

    # interval change localization results
    df_bbox = pd.DataFrame({'dicom_id': dicoms, 'dicom_id_x': dicom_xs, 'dicom_id_y': dicom_ys, 'image_type':image_types,
                            'landmark': landmark_names, 'ic_label': ic_labels, 'bbox_landmark_gt': bb_landmark_gts,
                            'bbox_landmark_gen': bb_landmark_gens, 'bbox_interval_gen': bb_observation_gens})
    # expand df_bbox and compute IoU between each pair of ground truth and generated bboxes
    df_bbox_landmark, df_bbox_interval = expand_df_bbox(df_bbox)
    thres = [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
    dict_loc = {}

    # landmark, new, image_x
    idx = (df_bbox_landmark['ic_label'] == 3) & (df_bbox_landmark['image_type'] == 'x')
    df = df_bbox_landmark[idx]
    recalls, precisions = get_recall_precision(df, thres)
    mAP = average_precision(recalls, precisions)
    dict_loc['landmark_new_x'] = {}
    dict_loc['landmark_new_x']['threshold'] = thres
    dict_loc['landmark_new_x']['recall'] = list(recalls)
    dict_loc['landmark_new_x']['precisions'] = list(precisions)
    dict_loc['landmark_new_x']['mAP'] = str(mAP)

    # interval, new, image_x
    idx = (df_bbox_interval['ic_label'] == 3) & (df_bbox_interval['image_type'] == 'x')
    df = df_bbox_interval[idx]
    recalls, precisions = get_recall_precision(df, thres)
    mAP = average_precision(recalls, precisions)
    dict_loc['interval_new_x'] = {}
    dict_loc['interval_new_x']['threshold'] = thres
    dict_loc['interval_new_x']['recall'] = list(recalls)
    dict_loc['interval_new_x']['precisions'] = list(precisions)
    dict_loc['interval_new_x']['mAP'] = str(mAP)

    # landmark, worsened, image_x
    idx = (df_bbox_landmark['ic_label'] == 2) & (df_bbox_landmark['image_type'] == 'x')
    df = df_bbox_landmark[idx]
    recalls, precisions = get_recall_precision(df, thres)
    mAP = average_precision(recalls, precisions)
    dict_loc['landmark_worsened_x'] = {}
    dict_loc['landmark_worsened_x']['threshold'] = thres
    dict_loc['landmark_worsened_x']['recall'] = list(recalls)
    dict_loc['landmark_worsened_x']['precisions'] = list(precisions)
    dict_loc['landmark_worsened_x']['mAP'] = str(mAP)

    # interval, worsened, image_x
    idx = (df_bbox_interval['ic_label'] == 2) & (df_bbox_interval['image_type'] == 'x')
    df = df_bbox_interval[idx]
    recalls, precisions = get_recall_precision(df, thres)
    mAP = average_precision(recalls, precisions)
    dict_loc['interval_worsened_x'] = {}
    dict_loc['interval_worsened_x']['threshold'] = thres
    dict_loc['interval_worsened_x']['recall'] = list(recalls)
    dict_loc['interval_worsened_x']['precisions'] = list(precisions)
    dict_loc['interval_worsened_x']['mAP'] = str(mAP)

    # landmark, worsened, image_y
    idx = (df_bbox_landmark['ic_label'] == 2) & (df_bbox_landmark['image_type'] == 'y')
    df = df_bbox_landmark[idx]
    recalls, precisions = get_recall_precision(df, thres)
    mAP = average_precision(recalls, precisions)
    dict_loc['landmark_worsened_y'] = {}
    dict_loc['landmark_worsened_y']['threshold'] = thres
    dict_loc['landmark_worsened_y']['recall'] = list(recalls)
    dict_loc['landmark_worsened_y']['precisions'] = list(precisions)
    dict_loc['landmark_worsened_y']['mAP'] = str(mAP)

    # interval, new, image_y
    idx = (df_bbox_interval['ic_label'] == 2) & (df_bbox_interval['image_type'] == 'y')
    df = df_bbox_interval[idx]
    recalls, precisions = get_recall_precision(df, thres)
    mAP = average_precision(recalls, precisions)
    dict_loc['interval_worsened_y'] = {}
    dict_loc['interval_worsened_y']['threshold'] = thres
    dict_loc['interval_worsened_y']['recall'] = list(recalls)
    dict_loc['interval_worsened_y']['precisions'] = list(precisions)
    dict_loc['interval_worsened_y']['mAP'] = str(mAP)

    with open(os.path.join(args.exp_dir, 'results_localization.json'), 'w') as f:
        json.dump(dict_loc, f)

    # results per landmark
    # report number of ImaGenome bbox per landmark
    idx = df_bbox['image_type'] == 'x'
    df = df_bbox[idx]
    df['bbox_landmark_flag'] = df.apply(lambda x: get_bbox_landmark_gt_flag(x), axis=1)
    df1 = df.groupby(['landmark', 'ic_label']).size().reset_index()
    df1.columns = ['landmark', 'ic_label', 'num_samples']
    df2 = df.groupby(['landmark', 'ic_label'])['bbox_landmark_flag'].sum().reset_index()
    df2.columns = ['landmark', 'ic_label', 'num_bboxes']
    df_bbox_summary = df1.merge(df2, how='left', on=['landmark', 'ic_label'])

    mAPs = []
    recalls_iou_10 = []
    precisions_iou_10 = []
    recalls_iou_25 = []
    precisions_iou_25 = []
    thres = [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
    for i, r in df_bbox_summary.iterrows():
        # interval, new, image_x
        idx = (df_bbox_interval['ic_label'] == r.ic_label) & (df_bbox_interval['image_type'] == 'x') & \
              (df_bbox_interval['landmark'] == r.landmark)
        df = df_bbox_interval[idx]
        recalls, precisions = get_recall_precision(df, thres)
        mAP = average_precision(recalls, precisions)
        recalls_iou_10.append(recalls[thres.index(0.1)])
        precisions_iou_10.append(precisions[thres.index(0.1)])
        recalls_iou_25.append(recalls[thres.index(0.25)])
        precisions_iou_25.append(precisions[thres.index(0.25)])
        mAPs.append(mAP)

    df_bbox_summary['recall_iou_10'] = recalls_iou_10
    df_bbox_summary['precision_iou_10'] = precisions_iou_10
    df_bbox_summary['recall_iou_25'] = recalls_iou_25
    df_bbox_summary['precision_iou_25'] = precisions_iou_25
    df_bbox_summary['mAP'] = mAPs

    filename = os.path.join(args.exp_dir, 'results_localization_landmark.csv')
    df_bbox_summary.to_csv(filename, index=False)






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

    # Value counts of predictions
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


def visualization(cam1_sel_x_resize, cam1_sel_y_resize, cam2_norm_x_resize, cam2_norm_y_resize, image_x, image_y, img_pth_x, img_pth_y, did_x, did_y, sid_x, sid_y,
                  landmark_idx, landmark_bbox_x, landmark_bbox_y, interval_hours, output, target, model, ic_cls, args):
    subdir = args.exp_dir + '/plots' + '/eval/' + ic_cls + '/'
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
    pred_str = 'Softmax: ' + str(F.softmax(output))
    txt = 'Interval hours = ' + str(round(interval_hours.item(), 1)) + '\n' + 'True label: ' + str(target.item()) + '\n' \
          + pred_str + '\n' + 'study id: ' + str(sid_y.item()) \
          + report_y + '\n' + 'study id: ' + str(sid_x.item()) + report_x
    with open(filename, 'w') as f:
        sys.stdout = f
        print(txt)
        sys.stdout = original_stdout

    # Compute GradCAM
    p_ic = F.softmax(output).detach().cpu().numpy()
    true_label = int(target.cpu().detach())

    # print original images
    np_transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.resize),
        lambda x: np.float32(x) / 255
    ])
    landmark = args.landmark_names_spec[landmark_idx.item()]

    try:
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
        model.eval()
        with torch.no_grad():
            f1_y = model.net1(image_y)[-1]
            f1_y_p = model.pool(f1_y)
            logit1_y = model.fc1(f1_y_p.squeeze())  # b * a
            p1_y = torch.sigmoid(logit1_y).cpu().detach().numpy()

            f1_x = model.net1(image_x)[-1]
            f1_x_p = model.pool(f1_x)
            logit1_x = model.fc1(f1_x_p.squeeze())  # b * a
            p1_x = torch.sigmoid(logit1_x).cpu().detach().numpy()

        ax3 = axs[0, 1]
        vis_y = show_cam_on_image(img_y_np, cam1_sel_y_resize, use_rgb=True)
        ax3.imshow(vis_y)
        # add ImaGenome bbox
        b = landmark_bbox_y.detach().cpu().numpy()
        rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='lime',
                                 facecolor='none')
        ax3.add_patch(rect)
        # add detected bbox
        BBG = BoundingBoxGenerator(cam1_sel_y_resize, percentile=0.95)
        bb_gen = BBG.get_bbox_pct()
        # compute importance score of each bbox
        scores = []
        if isinstance(bb_gen[0], list): # multiple bounding boxes
            for b in bb_gen:
                score = get_cumlative_attention(cam1_sel_y_resize, b)
                scores.append(score)
        else:
            b = bb_gen
            score = get_cumlative_attention(cam1_sel_y_resize, b)
            scores.append(score)
        scores = np.array(scores)
        score_threshold = 0.5 * np.max(scores)
        if isinstance(bb_gen[0], list): # multiple bounding boxes
            for ib in range(len(bb_gen)):
                if scores[ib] > score_threshold:
                    b = bb_gen[ib]
                    rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='blue',
                                     facecolor='none')
                    ax3.add_patch(rect)
        else:
            b = bb_gen
            rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='blue',
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
        # add detected bbox
        BBG = BoundingBoxGenerator(cam1_sel_x_resize, percentile=0.95)
        bb_gen = BBG.get_bbox_pct()
        # compute importance score of each bbox
        scores = []
        if isinstance(bb_gen[0], list): # multiple bounding boxes
            for b in bb_gen:
                score = get_cumlative_attention(cam1_sel_x_resize, b)
                scores.append(score)
        else:
            b = bb_gen
            score = get_cumlative_attention(cam1_sel_x_resize, b)
            scores.append(score)
        scores = np.array(scores)
        score_threshold = 0.5 * np.max(scores)
        if isinstance(bb_gen[0], list): # multiple bounding boxes
            for ib in range(len(bb_gen)):
                if scores[ib] > score_threshold:
                    b = bb_gen[ib]
                    rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='blue',
                                     facecolor='none')
                    ax4.add_patch(rect)
        else:
            b = bb_gen
            rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='blue',
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
        # add detected bbox
        BBG = BoundingBoxGenerator(cam2_norm_y_resize, percentile=0.95)
        bb_gen = BBG.get_bbox_pct()
        # compute importance score of each bbox
        scores = []
        if isinstance(bb_gen[0], list): # multiple bounding boxes
            for b in bb_gen:
                score = get_cumlative_attention(cam2_norm_y_resize, b)
                scores.append(score)
        else:
            b = bb_gen
            score = get_cumlative_attention(cam2_norm_y_resize, b)
            scores.append(score)
        scores = np.array(scores)
        score_threshold = 0.5 * np.max(scores)
        if isinstance(bb_gen[0], list): # multiple bounding boxes
            for ib in range(len(bb_gen)):
                if scores[ib] > score_threshold:
                    b = bb_gen[ib]
                    rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='red',
                                     facecolor='none')
                    ax5.add_patch(rect)
        else:
            b = bb_gen
            rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='red',
                                     facecolor='none')
            ax5.add_patch(rect)
        t5 = 'IC Target =' + str(true_label) + ', Prediction ='+ str(round(p_ic[0, true_label], 2))
        ax5.set_title(t5)

        ax6 = axs[1, 2]
        vis_x = show_cam_on_image(img_x_np, cam2_norm_x_resize, use_rgb=True)
        ax6.imshow(vis_x)
        # add ImaGenome bbox
        b = landmark_bbox_x.detach().cpu().numpy()
        rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='lime',
                                 facecolor='none')
        ax6.add_patch(rect)
        # add disease bbox
        # generate bbox
        BBG = BoundingBoxGenerator(cam2_norm_x_resize, percentile=0.95)
        bb_gen = BBG.get_bbox_pct()
        # compute importance score of each bbox
        scores = []
        if isinstance(bb_gen[0], list): # multiple bounding boxes
            for b in bb_gen:
                score = get_cumlative_attention(cam2_norm_x_resize, b)
                scores.append(score)
        else:
            b = bb_gen
            score = get_cumlative_attention(cam2_norm_x_resize, b)
            scores.append(score)
        scores = np.array(scores)
        score_threshold = 0.5 * np.max(scores)

        if isinstance(bb_gen[0], list): # multiple bounding boxes
            for ib in range(len(bb_gen)):
                if scores[ib] > score_threshold:
                    b = bb_gen[ib]
                    rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='red',
                                     facecolor='none')
                    ax6.add_patch(rect)
        else:
            b = bb_gen
            rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=3, edgecolor='red',
                                     facecolor='none')
            ax6.add_patch(rect)
        t6 = 'IC Target =' + str(true_label) + ', Prediction ='+ str(round(p_ic[0, true_label], 2))
        ax6.set_title(t6)

        filename = subdir + did_x + '.png'
        plt.savefig(filename)
    except:
        pass


def get_net1_cams(image_x, image_y, landmark_idx, model, args):
    # compute anatomical CAMs
    f1_y = model.net1(image_y)[-1]
    cam1_y = torch.einsum('bchw, ac -> bahw', f1_y, model.fc1.weight)  # b * a * h * w
    f1_x = model.net1(image_x)[-1]
    cam1_x = torch.einsum('bchw, ac -> bahw', f1_x, model.fc1.weight)  # b * a * h * w
    if args.cam_norm_type == 'indep':
        cam1_norm_y = model.normalize_cam1(cam1_y)
        cam1_norm_x = model.normalize_cam1(cam1_x)
    elif args.cam_norm_type == 'dep':
        cam1_norm_x, cam1_norm_y = model.normalize_cams(cam1_x, cam1_y)
    else:
        raise ValueError('invalid cam normalization type %r' % args.cam_norm_type)
    cam1_sel_y = cam1_norm_y[0][int(landmark_idx)]
    cam1_sel_y = cam1_sel_y.detach().cpu().numpy()
    cam1_sel_y_resize = im2double(cv2.resize(cam1_sel_y, (args.resize, args.resize)))
    cam1_sel_x = cam1_norm_x[0][int(landmark_idx)]
    cam1_sel_x = cam1_sel_x.detach().cpu().numpy()
    cam1_sel_x_resize = im2double(cv2.resize(cam1_sel_x, (args.resize, args.resize)))
    return cam1_sel_x_resize, cam1_sel_y_resize



def get_net2_gradcam(image_x, image_y, logit, target, model, args):
    # Compute interval change GradCAM
    true_label = int(target.cpu().detach())
    logit[:, true_label].backward()
    gradients_x = model.get_activations_gradient_x()
    gradients_y = model.get_activations_gradient_y()
    pooled_gradients_x = torch.mean(gradients_x, dim=[0, 2, 3])
    pooled_gradients_y = torch.mean(gradients_y, dim=[0, 2, 3])
    activations_x = model.get_activations_x(image_x)
    activations_y = model.get_activations_y(image_y)
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
    return cam2_norm_x_resize, cam2_norm_y_resize


def generate_bbox(heatmap, threshold=0.5):
    BBG = BoundingBoxGenerator(heatmap, percentile=0.95)
    bb_gen = BBG.get_bbox_pct()

    # bb_gen is empty
    if len(bb_gen) == 0:
        return [[-1, -1, 0, 0]]

    # compute importance score of each bbox
    scores = []
    if isinstance(bb_gen[0], list):  # multiple bounding boxes
        for b in bb_gen:
            score = get_cumlative_attention(heatmap, b)
            scores.append(score)
    else:
        b = bb_gen
        score = get_cumlative_attention(heatmap, b)
        scores.append(score)
    scores = np.array(scores)
    score_threshold = threshold * np.max(scores)

    bs = []
    if isinstance(bb_gen[0], list):  # multiple bounding boxes
        for ib in range(len(bb_gen)):
            if scores[ib] > score_threshold:
                b = bb_gen[ib]
                bs.append(b)
    else:
        bs = [bb_gen]

    return bs


def expand_df_bbox(df_bbox):
    # expand detected bounding boxes into multiple rows, each of which has only one detected bbox
    dicom_ids = []
    image_types = []
    landmarks = []
    ic_labels = []
    bb_gt_idxs = []
    bb_gts = []
    bb_landmark_gen_idxs = []
    bb_landmark_gens = []
    # landmark detection dataframe
    for i, r in df_bbox.iterrows():
        bb_gt = np.array(r.bbox_landmark_gt)  # a list of 4 coordinates e.g., [x1, y1, x2, y2]
        # ground truth bounding box is available
        if np.sum(np.array(bb_gt)) != 0:
            bb_gen = r.bbox_landmark_gen # a list of one or more lists e.g., [[x1, y1, x2, y2], [x1, y1, x2, y2]]
            num_b = len(bb_gen)  # number of generated bbox
            for ib in range(num_b):
                dicom_ids.append(r.dicom_id)
                image_types.append(r.image_type)
                landmarks.append(r.landmark)
                ic_labels.append(r.ic_label)
                bb_gt_idxs.append(1)  # there can be only 1 ground truth landmark bounding box
                bb_gts.append(bb_gt)
                bb_landmark_gen_idxs.append(ib+1)
                bb_landmark_gens.append(bb_gen[ib])
    df_bbox_landmark = pd.DataFrame({'dicom_id': dicom_ids, 'image_type': image_types, 'landmark': landmarks, 'ic_label': ic_labels,
                                'bb_gt_idx': bb_gt_idxs, 'bb_gt': bb_gts, 'bb_gen_idx':bb_landmark_gen_idxs, 'bb_gen':bb_landmark_gens})
    df_bbox_landmark['iou'] = df_bbox_landmark.apply(lambda x: iou(x), axis=1)
    df_bbox_landmark['iobb1'] = df_bbox_landmark.apply(lambda x: iobb1(x), axis=1)
    df_bbox_landmark['iobb2'] = df_bbox_landmark.apply(lambda x: iobb2(x), axis=1)

    dicom_ids = []
    image_types = []
    landmarks = []
    ic_labels = []
    bb_gt_idxs = []
    bb_gts = []
    bb_interval_gen_idxs = []
    bb_interval_gens = []
    # interval change detection dataframe
    for i, r in df_bbox.iterrows():
        bb_gt = np.array(r.bbox_landmark_gt)  # a list of 4 coordinates e.g., [x1, y1, x2, y2]
        # ground truth bounding box is available
        if np.sum(np.array(bb_gt)) > 0:
            bb_gen = r.bbox_interval_gen # a list of one or more lists e.g., [[x1, y1, x2, y2], [x1, y1, x2, y2]]
            num_b = len(bb_gen)  # number of generated bbox
            for ib in range(num_b):
                dicom_ids.append(r.dicom_id)
                image_types.append(r.image_type)
                landmarks.append(r.landmark)
                ic_labels.append(r.ic_label)
                bb_gt_idxs.append(1)  # there can be only 1 ground truth landmark bounding box
                bb_gts.append(bb_gt)
                bb_interval_gen_idxs.append(ib+1)
                bb_interval_gens.append(bb_gen[ib])
    df_bbox_interval = pd.DataFrame({'dicom_id': dicom_ids, 'image_type': image_types, 'landmark': landmarks, 'ic_label': ic_labels,
                                'bb_gt_idx': bb_gt_idxs, 'bb_gt': bb_gts, 'bb_gen_idx': bb_interval_gen_idxs, 'bb_gen':bb_interval_gens})
    df_bbox_interval['iou'] = df_bbox_interval.apply(lambda x: iou(x), axis=1)
    df_bbox_interval['iobb1'] = df_bbox_interval.apply(lambda x: iobb1(x), axis=1)
    df_bbox_interval['iobb2'] = df_bbox_interval.apply(lambda x: iobb2(x), axis=1)

    return df_bbox_landmark, df_bbox_interval

def get_bbox_landmark_gt_flag(x):
    bbox = np.array(x.bbox_landmark_gt)
    if bbox.sum() > 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    main()