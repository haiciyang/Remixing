#!/bin/bash
#!MUSDB 2 + MUSDB
for n in 0424_214831 0425_232636 0502_134149
do
  CUDA_VISIBLE_DEVICES=7, python3 eval_on_samples.py  --model_name1 $n --n_src 3 --dataset Slakh --ratio_on_rep1
done