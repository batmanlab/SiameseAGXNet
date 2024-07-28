"""
Dataloader for CXR interval change prediction.
"""
import ast
import pickle
import json
from PIL import Image

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def prognosis_label_mapping(x):
        if x == 'improved':
            return 0
        if x == 'unchanged':
            return 1
        if x == 'worsened':
            return 2
        if x == 'new':
            return 3


class MIMICCXRInterval(Dataset):

    def __init__(self, args, mode, transform=None):

        self.args = args
        self.mode = mode
        self.transform = transform

        # read the original AGXNet datasplit
        # read master dataset
        df_all = pd.read_csv(self.args.img_chexpert_file)
        df_all['study_id'] = df_all['study_id'].apply(str)

        # create random splits 80%, 10%, 10%
        subject_ids = np.array(df_all['subject_id'].unique()) # ~65K patients
        np.random.seed(self.args.seed)
        np.random.shuffle(subject_ids)
        k1 = int(len(subject_ids) * 0.7)
        k2 = int(len(subject_ids) * 0.8)
        self.train_subject_ids = list(subject_ids[:k1])
        self.valid_subject_ids = list(subject_ids[k1:k2])
        self.test_subject_ids = list(subject_ids[k2:])

        # step 1: create study_id level prognosis table
        full_sids = np.load(self.args.sids_file)
        full_prognosis_dict = np.load(self.args.prognosis_dict_file)

        sid_lst = []
        ana_lst = []
        obs_lst = []
        prog_lst = []
        for i in range(len(full_sids)):
            d = ast.literal_eval(full_prognosis_dict[i])

            for k in d.keys():
                sid_lst.append(full_sids[i])
                ana_lst.append(self.args.full_anatomy_names[k[0]])
                obs_lst.append(self.args.full_obs[k[1]])
                words = d[k].split('|')

                # check whether words are consistent (e.g., 'increased|worsened' is inconsistent)
                new_flag = 0
                worsened_flag = 0
                unchanged_flag = 0
                improved_flag = 0
                other_flag = 0
                for w in words:
                    if w in self.args.new_words:
                        new_flag = 1
                    elif w in self.args.worsened_words:
                        worsened_flag = 1
                    elif w in self.args.unchanged_words:
                        unchanged_flag = 1
                    elif w in self.args.improved_words:
                        improved_flag = 1
                    else:
                        other_flag = 1
                sum_flag = new_flag + worsened_flag + unchanged_flag + improved_flag + other_flag

                # if prognosis labels are consistent
                if sum_flag == 1:
                    if worsened_flag == 1:
                        prog_lst.append('worsened')
                    elif unchanged_flag == 1:
                        prog_lst.append('unchanged')
                    elif improved_flag == 1:
                        prog_lst.append('improved')
                    elif new_flag == 1:
                        prog_lst.append('new')
                    else:
                        prog_lst.append('other')  # less confident prognosis weak labels
                else:
                    prog_lst.append('inconsistent')

        df = pd.DataFrame({'sid': sid_lst, 'landmark': ana_lst, 'observation': obs_lst, 'prognosis_label': prog_lst})
        # only include abnormal observations and specified landmarks
        idx1 = df['observation'].isin(self.args.abnorm_obs) & df['landmark'].isin(self.args.landmark_names_spec)
        # remove other (not confident) and inconsistent prognosis
        idx2 = df['prognosis_label'].isin(['new', 'worsened', 'unchanged', 'improved'])
        idx = idx1 & idx2
        df_prognosis = df[idx]
        # remove duplicated prognosis
        df_prognosis = df_prognosis.drop_duplicates()

        # step 2: read interval sequence table
        df_seq = pd.read_csv(self.args.sequence_file)

        # step 3: merge df_prognosis and df_seq (unique key: dicom_id_x * dicom_id_y * landmark * observation)
        df_master = df_seq.merge(df_prognosis, how='inner', left_on='study_id_x', right_on='sid')

        # step 4: define selection criteria and create the final dataset
        idx1 = df_master['observation'] == self.args.selected_obs
        idx2 = (df_master['interval_seconds'] > 3600) & (df_master['interval_seconds'] < (3600 * 24 * self.args.max_interval_days))

        if self.mode == 'train':
            idx3 = df_master['subject_id'].isin(self.train_subject_ids) # is in training dataset
            idx = idx1 & idx2 & idx3
        elif self.mode == 'valid':
            idx3 = df_master['subject_id'].isin(self.valid_subject_ids)  # is in validate dataset
            idx = idx1 & idx2 & idx3
        elif self.mode == 'test':
            idx3 = df_master['subject_id'].isin(self.test_subject_ids)  # is in test dataset
            idx = idx1 & idx2 & idx3
        else:
            raise Exception('Invalid split mode.')

        # selected master dataset
        self.df_master_sel = df_master[idx]
        self.df_master_sel['prognosis_label_embedding'] = self.df_master_sel['prognosis_label'].apply(prognosis_label_mapping)
        # compute label weights
        self.prognosis_label_weights = 1 / torch.FloatTensor(self.df_master_sel.groupby('prognosis_label_embedding').size())

        # random sample a fraction records
        if mode == 'train':
            self.df_master_sel = self.df_master_sel.sample(frac=self.args.frac, random_state=self.args.seed)
        # read ImaGenome landmark bbox and landmark name mapping files
        self.dict_imagenome_bbox = pickle.load(open(self.args.imagenome_bounding_box_file, "rb"))
        self.dict_landmark_mapping = json.load(open(self.args.imagenome_radgraph_landmark_mapping_file))

    def __len__(self):
        return self.df_master_sel.shape[0]

    def __getitem__(self, index):
        # get dicom ids, study ids
        did_x = self.df_master_sel.iloc[index, self.df_master_sel.columns.get_loc('dicom_id_x')] # t
        did_y = self.df_master_sel.iloc[index, self.df_master_sel.columns.get_loc('dicom_id_y')] # t-1
        sid_x = self.df_master_sel.iloc[index, self.df_master_sel.columns.get_loc('study_id_x')] # t
        sid_y = self.df_master_sel.iloc[index, self.df_master_sel.columns.get_loc('study_id_y')] # t-1

        # 2. load images
        img_pth_x = self.df_master_sel.iloc[index, self.df_master_sel.columns.get_loc('path_x')] # t
        img_pth_y = self.df_master_sel.iloc[index, self.df_master_sel.columns.get_loc('path_y')] # t-1
        image_x = Image.open(img_pth_x).convert('RGB')
        image_y = Image.open(img_pth_y).convert('RGB')
        if self.transform is not None:
            image_x = self.transform(image_x)
            image_y = self.transform(image_y)

        # 3. get interval times
        interval_seconds = self.df_master_sel.iloc[index, self.df_master_sel.columns.get_loc('interval_seconds')]
        interval_hours = interval_seconds/3600

        # 4. get landmark index
        landmark = self.df_master_sel.iloc[index, self.df_master_sel.columns.get_loc('landmark')]
        landmark_idx = self.args.landmark_names_spec.index(landmark)

        # 5. get interval change label
        label = self.df_master_sel.iloc[index, self.df_master_sel.columns.get_loc('prognosis_label_embedding')]
        weight = self.prognosis_label_weights[label]

        # 6. extract ImaGenome bounding box
        landmark_bbox_x = np.zeros((len(self.args.landmark_names_spec), 4)) # ImaGenome bbox is (x1, y1, x2, y2) on a 512 * 512 image
        landmark_bbox_y = np.zeros((len(self.args.landmark_names_spec), 4))  # ImaGenome bbox is (x1, y1, x2, y2) on a 512 * 512 image
        for k, v in self.dict_landmark_mapping.items():
            j = list(self.dict_landmark_mapping.keys()).index(k)
            l = self.args.landmark_names_spec.index(v) # find the index of corresponding landmark name in defined RadGraph landmark specs
            try:
                landmark_bbox_x[l] = self.dict_imagenome_bbox[did_x][j, :]
            except:
                pass
            try:
                landmark_bbox_y[l] = self.dict_imagenome_bbox[did_y][j, :]
            except:
                pass
        landmark_bbox_x = torch.FloatTensor(landmark_bbox_x[landmark_idx])
        landmark_bbox_y = torch.FloatTensor(landmark_bbox_y[landmark_idx])

        return did_x, did_y, sid_x, sid_y, img_pth_x, img_pth_y, image_x, image_y, interval_hours, landmark_idx, \
               label, weight, landmark_bbox_x, landmark_bbox_y



