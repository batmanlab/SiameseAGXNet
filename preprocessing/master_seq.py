import pickle
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# load IDs
df_master = pd.read_csv('./preprocessing/mimic-cxr-chexpert.csv')
# load scan date and time
df_study_date = pd.read_csv('./preprocessing/mimic-cxr-study-date-time.csv')
# read the adj. matrix, prognosis info.
with open('./preprocessing/mimic-cxr-adj-mtx-prognosis.pickle', 'rb') as handle:
    dict_output = pickle.load(handle)

df_merge = df_master.merge(df_study_date, how='left', on=['subject_id', 'study_id', 'dicom_id'])
df_merge['rank'] = df_merge.groupby(['subject_id'])['study_date'].rank('dense')

# only consider frontal views
df_merge_front = df_merge[df_merge['ViewPosition'].isin(['PA', 'AP'])]

# prepare for joining
df_merge_1 = df_merge_front[['subject_id', 'study_id',  'dicom_id', 'path', 'ViewPosition', 'study_date', 'rank']]
df_merge_2 = df_merge_1.copy()
df_merge_2.loc[:, 'rank'] = df_merge_2.loc[:, 'rank'] + 1

# merge two dfs
df_merge_seq = df_merge_1.merge(df_merge_2, how='left', on=['subject_id', 'rank'])

# add interval days
days = []
for i, r in tqdm(df_merge_seq.iterrows(), total=len(df_merge_seq)):
    try:
        d1 = datetime.strptime(r.study_date_x, "%Y%m%d")
        d2 = datetime.strptime(r.study_date_y, "%Y%m%d")
        days.append(abs((d2 - d1).days))
    except:
        days.append(None)
df_merge_seq['interval_days'] = days

# add prognosis dictionary in string format
prognosis_lst = []
for i, r in tqdm(df_merge_seq.iterrows(), total=len(df_merge_seq)):
    sid = r.study_id_x
    try:
        prognosis_lst.append(dict_output[sid]['prognosis'])
    except:
        prognosis_lst.append('{}')
df_merge_seq['prognosis'] = prognosis_lst

# save output file
df_merge_seq.to_csv('./preprocessing/mimic-cxr-seq.csv', index=False)