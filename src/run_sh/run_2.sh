#!/bin/bash
#!Slakh 1, base + MUSDB/Slakh
for n in 0504_223710 0504_223534 0509_233827
do
    for ds in Slakh
    do
      CUDA_VISIBLE_DEVICES=1, python3 eval_on_samples.py  --model_name1 $n --n_src 5 --dataset $ds --with_silent --ratio_on_rep
    done
done