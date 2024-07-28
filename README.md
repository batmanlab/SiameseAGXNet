## Anatomy-specific Progression Classification in Chest Radiographs via Weakly Supervised Learning


<p align="center">
  <img width="500" alt="Siamese AGXNet" src="https://github.com/user-attachments/assets/6ff794b4-3d39-4a2c-9b19-06a3c41a03c1">
</p>

### Abstract
#### Purpose:
To develop a machine learning approach for classifying disease progression in chest radiographs using weak labels automatically derived from radiology reports.
#### Materials and Methods:
In this retrospective study, a twin neural network was developed to classify anatomy-specific disease progression into four categories: improved, unchanged, worsened, and new. A two-step weakly-supervised learning approach was employed, pre-training the model on 243,008 frontal chest radiographs from 63,877 MIMIC-CXR patients (mean age 51.7 years; female 34,813), and fine-tuning it on the subset with progression labels derived from consecutive studies.  Model performance was evaluated for six pathologic observations on test datasets of unseen MIMIC-CXR patients. Area under the receiver operating characteristic (AUC) analysis was used to evaluate classification performance. The algorithm is also capable of generating bounding-box predictions to localize areas of new progression. Recall, precision, and mean average precision (mAP) were used to evaluate the new progression localization. One-tailed paired t-tests were used to assess statistical significance.
#### Results:
The model outperformed most baselines in progression classification, achieving macro AUC scores of 0.72 ± 0.004 for atelectasis, 0.75 ± 0.007 for consolidation, 0.76 ± 0.017 for edema, 0.81 ± 0.006 for effusion, 0.7 ± 0.032 for pneumonia, and 0.69 ± 0.01 for pneumothorax. For new observation localization, the model achieved mAP scores of 0.25 ± 0.03 for atelectasis, 0.34 ± 0.03 for consolidation, 0.33 ± 0.03 for edema, and 0.31 ± 0.03 for pneumothorax.
#### Conclusion:
Disease progression classification models were developed on a large chest radiograph dataset, which can be used to monitor interval changes and detect new pathologies on chest radiographs.

### Paper Link
Radiology AI: https://pubs.rsna.org/doi/10.1148/ryai.230277

This is the PyTorch implementation of the [journal paper](https://pubs.rsna.org/doi/10.1148/ryai.230277)
```
@Article{doi:10.1148/ryai.230277,
author = {Yu, Ke and Ghosh, Shantanu and Liu, Zhexiong and Deible, Christopher and Poynton, Clare B. and Batmanghelich, Kayhan},
title = {Anatomy-specific Progression Classification in Chest Radiographs via Weakly Supervised Learning},
journal = {Radiology: Artificial Intelligence},
}
```

### Data Download
