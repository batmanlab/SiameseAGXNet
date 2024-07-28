#!/usr/bin/env bash
# Evaluation script for model and observation combinations

SEEDS="0"
OBS_Val="atelectasis consolidation edema effusion opacity pneumonia pneumothorax"
MODELS="AGXNet_Siamese"
FRAC_Val="1.0"
ATT_Type="None Residual"
FREEZE_Flag="T"
CAM_NORM_TYPE="indep"

for seed in $SEEDS;do
  for obs in $OBS_Val;do
    for mdl in $MODELS;do
      for frac in $FRAC_Val;do
        for att in $ATT_Type;do
          for frz in $FREEZE_Flag;do
            for nrm in $CAM_NORM_TYPE;do
              if [ $frac == "0.01" ]
              then
                fr="001"
              elif [ $frac == "0.1" ]
              then
                fr="010"
              elif [ $frac == "1.0" ]
              then
                fr="100"
              fi
              OUTPUT_DIR='./experiments/seed_'$seed'/'$obs'/'$fr'/'$mdl'_'$att'_'$frz
              mkdir -p ${OUTPUT_DIR}

              source activate WSL_Journal

              python -W ignore eval_agxnet_saimese.py \
                --exp-dir=${OUTPUT_DIR} \
                --pretrained-type='AGXNet_Siamese' \
                --freeze_net1=$frz \
                --anatomy-attention-type=$att \
                --epsilon=0.0 \
                --exp-dir=${OUTPUT_DIR} \
                --ckpt-name='model_best.pth.tar' \
                --selected-obs=$obs \
                --workers=4 \
                --batch-size=1 \
                --frac=$frac \
                --seed=$seed \
                --learning-rate=1e-4 >> ${OUTPUT_DIR}/eval.log 2>&1
            done
          done
        done
      done
    done
  done
done