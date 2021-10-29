#!/bin/bash
#!Slakh 1, base + Slakh/MUSDB
for n in 0504_223630 0509_230425 0504_223405 0509_234021
do
    for ds in Slakh
    do
        CUDA_VISIBLE_DEVICES=0, python3 eval_on_samples.py  --model_name1 $n --n_src 5 --dataset $ds --with_silent
    done
done




# CUDA_VISIBLE_DEVICES=0, python3 eval_on_samples.py  --model_name1 0417_200733 --n_src 4 --dataset MUSDB --with_silent