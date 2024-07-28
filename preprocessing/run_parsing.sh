#!/usr/bin/env bash

INPUT_PATH='/ocean/projects/asc170022p/yuke/PythonProject/WSL_Journal/preprocessing/mimic-cxr-radgraph-itemized.csv'
OUTPUT_PATH='/ocean/projects/asc170022p/yuke/PythonProject/WSL_Journal/preprocessing/mimic-cxr-radgraph-sentence-parsed.csv'

python radgraph_parsed.py \
  --input-path=${INPUT_PATH} \
  --output-path=${OUTPUT_PATH} >> run_parsing.log 2>&1