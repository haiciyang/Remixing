#!/bin/bash
#!Slakh 2 + MUSDB/Slakh
for n in 0511_233047 0511_232959 0512_143526 0513_121934 0513_122235 0513_103252
do
    for ds in MUSDB Slakh
    do
      CUDA_VISIBLE_DEVICES=3, python3 eval_on_samples.py  --model_name1 $n --n_src 2 --dataset $ds --with_silent --ratio_on_rep
    done
done