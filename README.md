# Don't Separate, Learn to Remix: End-to-End Neural Remixing with Joint Optimization
Open-source code for paper - H. Yang, S. Firodiya, N.J. Bryan, M. Kim, "Don't Separate, Learn to Remix: End-to-End Neural Remixing with Joint Optimization."  ICASSP 2022 - 2022 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP), 2022
## Prerequisites
### Environment
- Python 3.6.0 <br>
- torch 1.8.0 <br>
- torchaudio 0.8.0 <br>

### Data
- Download MUSDB18 from - [https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems)
- Download Slakh from - [http://www.slakh.com/](http://www.slakh.com/)
### Conv-TasNet model package
Asteroid - [https://github.com/asteroid-team/asteroid](https://github.com/asteroid-team/asteroid)<br>
asteroid/models/conv-tasnet.py

## Training steps
### 1. Data processing
- Prepare MUSDB for training or validation - 
<code>python mus_process.py</code>
Set _Dataset_ to "train" or "test" per needed. 
- Prepare MUSDB for evaluation - 
<code>python mus_eval_process.py</code>
- Prepare Slakh for training or validation - 
<code>python slakh_process.py</code>
Set _Dataset_ to "train" or "test" per needed. 
- Prepare Slakh for evaluation - 
<code>python slakh_eval_process.py</code>

### 3. Model training
#### Important Parameters:

| Symbol | Description |
| --- | ----------- |
| n_src          |  Number of sources contained in the mixture|
| weight_src                   |  source separation loss scale|
| weight_mix                  |  remix loss scale |
| transfer_model               |  The model label to start transferring training from |
| trainset            | dataset. MUSDB or Slakh |
| train_loss               |  SDR or SDSDR |
| with_silent          | Whether or not using the data, has actually contains less number of sources than the number the model is designed on |
| baseline                   | Whether or not training baseline |
| ratio_on_rep                |  Whether or not having ratios applying on the representation feature space  |

#### Examples:
Scenario 1:  we are training the following models, on MUSDB with 4 sources, and we want the ratio of loss weights to be 1:4.
- Model 1 -  <code>python train_model.py --n_src 4 --weight_src 1 --weight_mix 4 --trainset MUSDB </code>
- Model 2 - <code>python train_model.py --n_src 4 --weight_src 1 --weight_mix 4 --trainset MUSDB --ratio_on_rep</code>
- Baseline - <code>python train_model.py --n_src 4 --weight_src 1 --weight_mix 4 --trainset MUSDB --baseline</code>

#### Model Evaluation
To get the remix output of specific one track
<code>python3 eval_on_samples.py  --model_name 0826_123024 --n_src 4 --dataset MUSDB --loss_f SDSDR<code>
