#!/bin/bash
#!MUSDB 1 ,base + MUSDB
for n in 0425_193039 0427_231820 0425_193201 0502_134059
do
  CUDA_VISIBLE_DEVICES=4, python3 eval_on_samples.py  --model_name1 $n --n_src 3 --dataset MUSDB
done