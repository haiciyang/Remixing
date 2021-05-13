#!/bin/bash
#!Slakh 2 + Slakh
for n in 0504_223710 0504_223534 0509_233827
do
  CUDA_VISIBLE_DEVICES=2, python3 eval_on_samples.py  --model_name1 $n --n_src 5 --dataset Slakh --ratio_on_rep1
done