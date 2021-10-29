#!/bin/bash
#!Slakh 2 + Slakh
for n in 0512_143605 0509_234319 0511_232843 0502_134403 0513_103102 0513_142747 0513_142533 0513_142727
do
    for ds in MUSDB Slakh
    do
      CUDA_VISIBLE_DEVICES=2, python3 eval_on_samples.py  --model_name1 $n --n_src 2 --dataset $ds --with_silent
    done
done