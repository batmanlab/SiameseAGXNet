"""Extract study date and time from DICOM files in MIMIC-CXR dataset.
This is used to identify the consecutive studies for interval change prediction.
"""

from tqdm import tqdm
import pydicom as dicom
import pandas as pd

df = pd.read_csv('PATH_TO_MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/cxr-record-list.csv')
files_dir = 'PATH_TO_MIMIC_CXR/MIMICCXR/2.0.0/'
df['dicom_path'] = files_dir + df['path']

subject_ids = []
study_ids = []
dicom_ids = []
study_dates = []
study_times = []
for i, r in tqdm(df.iterrows(), total=len(df)):
    subject_ids.append(r.subject_id)
    study_ids.append(r.study_id)
    dicom_ids.append(r.dicom_id)
    try:
        ds = dicom.dcmread(r.dicom_path)
        study_dates.append(ds.AcquisitionDate)
        study_times.append(ds.AcquisitionTime)
    except:
        study_dates.append('None')
        study_times.append('None')


df_study_dates = pd.DataFrame({'subject_id': subject_ids, 'study_id': study_ids,
                               'dicom_id': dicom_ids, 'study_date': study_dates, 'study_time': study_times})

df_study_dates.to_csv('./preprocessing/mimic-cxr-study-date-time.csv', index=False)