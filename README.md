# Self-Distillation with no labels DINO

This repo provides a pytorch-lightning implementation of DINO, its instanciation within a configurable experiment bed, as well as tools to investigate the method in various settings. By default everything is logged to WANDB.

### Installation

Install with `pip install -e`. Currently tested with `python==3.8.5`, `torch==1.13.0+cu117`, `pytorch-lightning==1.6.3`, `torchmetrics==0.10.0` and `wandb==0.12.12`. Newer pytorch-lightning versions use inference_mode by default which seems to break some things.. update needed from my side here.

Specify data and results path in environment `DINO_DATA` `DINO_RESULTS` variables, e.g. by adding to your .bashrc
```
export DINO_DATA=$HOME/dinopl/datasets/data
export DINO_RESULTS=$HOME/dinopl/results
```

### Usage

```
python dino.py --help
usage: dino.py [-h] [--from_json FROM_JSON] [--n_workers N_WORKERS] [--seed SEED] [--log_every LOG_EVERY] [--ckpt_path CKPT_PATH] [--force_cpu] [--dataset {mnist,cifar10}] [--n_classes N_CLASSES] [--mc {2x128+4x96,2x128,1x128,2x32+4x32,2x32,1x32,2x28+4x28,2x28,1x28}] [--bs_train BS_TRAIN]
               [--bs_eval BS_EVAL] [--label_noise_ratio LABEL_NOISE_RATIO] [--logit_noise_temp LOGIT_NOISE_TEMP] [--resample_noise RESAMPLE_NOISE] [--enc {ConvNet,convnet_16_2,ResNet,resnet10,resnet18,resnet34,VGG,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,Flatten}]
               [--enc_seed ENC_SEED] [--enc_norm_layer {BatchNorm,InstanceNorm,GroupNorm8,LayerNorm,Identity}] [--tiny_input] [--mlp_act {ReLU,GELU}] [--mlp_bn] [--hid_dims [HID_DIMS [HID_DIMS ...]]] [--l2bot_dim L2BOT_DIM] [--l2bot_cfg L2BOT_CFG] [--out_dim OUT_DIM]
               [--s_init {s_ckpt,random,t_ckpt}] [--s_mode {supervised,distillation}] [--t_init {s_ckpt,t_ckpt,random,student}] [--t_mode {ema,no_update,prev_epoch}] [--t_mom T_MOM] [--t_update_every T_UPDATE_EVERY] [--t_bn_mode {from_student,from_data}] [--t_eval T_EVAL] [--t_cmom T_CMOM]
               [--s_cmom S_CMOM] [--t_temp T_TEMP] [--s_temp S_TEMP] [--loss {CE,H_pred,KL}] [--loss_pairing {all,same,opposite}] [--n_epochs N_EPOCHS] [--opt {sgd,adam,adamw}] [--opt_lr OPT_LR] [--opt_wd OPT_WD] [--clip_grad CLIP_GRAD] [--wn_freeze_epochs WN_FREEZE_EPOCHS]
               [--probe_every PROBE_EVERY] [--probing_epochs PROBING_EPOCHS] [--prober_seed PROBER_SEED] [--save_features [{embeddings,projections,logits,all} [{embeddings,projections,logits,all} ...]]]

optional arguments:
  -h, --help            show this help message and exit
  --from_json FROM_JSON
                        Loading configuration according to priority:
                                1. from commandline arguments
                                2. from JSON configuration file
                                3. from parser default values.

General:
  --n_workers N_WORKERS
                        Number of parallel threads for data loading.
  --seed SEED           Random number generator seed.
  --log_every LOG_EVERY
                        Log every so many steps.
  --ckpt_path CKPT_PATH
                        Path to checkpoint, used by '--t_init'.
  --force_cpu           Force training on CPU instead of GPU.

Data:
  --dataset {mnist,cifar10}
                        Datset to train on.
  --n_classes N_CLASSES
                        Number of classes. By default determined from dataset but can be overwritten for logit noise.
  --mc {2x128+4x96,2x128,1x128,2x32+4x32,2x32,1x32,2x28+4x28,2x28,1x28}
                        Specification of multicrop augmentation.
  --bs_train BS_TRAIN   Batch size for the training set.
  --bs_eval BS_EVAL     Batch size for valid/test set.
  --label_noise_ratio LABEL_NOISE_RATIO
                        Add label noise (random assignemt) for supervised training.
  --logit_noise_temp LOGIT_NOISE_TEMP
                        Add logit noise (sharpened gaussian logits) for supervised training.
  --resample_noise RESAMPLE_NOISE

Model:
  --enc {ConvNet,convnet_16_2,ResNet,resnet10,resnet18,resnet34,VGG,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,Flatten}
                        Defines the model to train on.
  --enc_seed ENC_SEED   The seed for model creation, use numbers with good balance of 0 and 1 bits.
  --enc_norm_layer {BatchNorm,InstanceNorm,GroupNorm8,LayerNorm,Identity}
                        Overwrite the normalization layer of the model if supported.
  --tiny_input          Adjust encoder for tiny inputs, e.g. resnet for cifar 10.
  --mlp_act {ReLU,GELU}
                        Activation function of DINOHead MLP.
  --mlp_bn              Use batchnorm in DINOHead MLP.
  --hid_dims [HID_DIMS [HID_DIMS ...]]
                        Hidden dimensions of DINOHead MLP.
  --l2bot_dim L2BOT_DIM
                        L2-Bottleneck dimension of DINOHead MLP.
  --l2bot_cfg L2BOT_CFG
                        L2-Bottleneck configuration string: '{wn,-}/{l,lb,-}/{fn,-}/{wn,-}/{l,lb,-}/{wn,-}'.
  --out_dim OUT_DIM     Output dimension of the DINOHead MLP.

DINO:
  --s_init {s_ckpt,random,t_ckpt}
                        Initialization of student, specify '--ckpt_path'.
  --s_mode {supervised,distillation}
                        Mode of student update.
  --t_init {s_ckpt,t_ckpt,random,student}
                        Initialization of teacher, specify '--ckpt_path'.
  --t_mode {ema,no_update,prev_epoch}
                        Mode of teacher update.
  --t_mom T_MOM         Teacher momentum for exponential moving average (float or Schedule).
  --t_update_every T_UPDATE_EVERY
                        Teacher update frequency for prev_epoch mode.
  --t_bn_mode {from_student,from_data}
                        Mode of teacher batchnorm updates: either from data stats or from student buffers.
  --t_eval T_EVAL       Run teacher in evaluation mode even on training data.
  --t_cmom T_CMOM       Teacher centering momentum of DINOHead (float or Schedule).
  --s_cmom S_CMOM       Student centering momentum of DINOHead (float or Schedule).
  --t_temp T_TEMP       Teacher temperature of DINOHead (float or Schedule).
  --s_temp S_TEMP       Student temperature of DINOHead (float or Schedule).
  --loss {CE,H_pred,KL}
                        Loss function to use in the multicrop loss.
  --loss_pairing {all,same,opposite}
                        Pairing strategy for the multicrop views in the loss function.

Training:
  --n_epochs N_EPOCHS   Number of epochs to train for.
  --opt {sgd,adam,adamw}
                        Optimizer to use for training.
  --opt_lr OPT_LR       Learning rate for optimizer (float or Schedule): specified wrt batch size 256 and linearly scaled.
  --opt_wd OPT_WD       Weight decay for optimizer (float or Schedule).
  --clip_grad CLIP_GRAD
                        Value to clip gradient norm to.
  --wn_freeze_epochs WN_FREEZE_EPOCHS
                        Epochs to freeze WeightNormalizedLinear layer in DINOHead.

addons:
  --probe_every PROBE_EVERY
                        Probe every so many epochs during training.
  --probing_epochs PROBING_EPOCHS
                        Number of epochs to train for linear probing.
  --prober_seed PROBER_SEED
                        The seed for reproducible probing, use numbers with good balance of 0 and 1 bits.
  --save_features [{embeddings,projections,logits,all} [{embeddings,projections,logits,all} ...]]
                        Save features for embeddings, projections and/or logits.
```


## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
```
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
