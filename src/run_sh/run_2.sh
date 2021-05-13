#!/bin/bash
#!Slakh 1, base + MUSDB
for n in 0504_223630 0509_230425 0504_223405 0509_234021
do
  CUDA_VISIBLE_DEVICES=1, python3 eval_on_samples.py  --model_name1 $n --n_src 5 --dataset MUSDB
done