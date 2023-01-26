# Self-Distillation with no labels DINO

This repo provides a pytorch-lightning implementation of DINO, its instanciation within a configurable experiment bed, as well as tools to investigate the method in various settings. By default everything is logged to WANDB.

### Installation

Install with `pip install -e`. Currently tested with `python==3.8.5`, `torch==1.13.0+cu117`, `pytorch-lightning==1.6.3`, `torchmetrics==0.10.0` and `wandb==0.12.12`. Newer pytorch-lightning versions use inference_mode by default which seems to break some things.. update needed from my side here.

Specify data and results path in environment `DINO_DATA` `DINO_RESULTS` variables, e.g. by adding to your .bashrc
```
export DINO_DATA=$HOME/dinopl/datasets/data
export DINO_RESULTS=$HOME/dinopl/results
```
or any other desired path.

### Usage

```
usage: dino.py [-h] [--from_json FROM_JSON] [--n_workers N_WORKERS] [--seed SEED] [--log_every LOG_EVERY] [--ckpt_path CKPT_PATH] [--force_cpu] [--float64 FLOAT64] [--dataset {MNIST,CIFAR10,CIFAR100}] [--n_classes N_CLASSES] [--mc {2x128+4x96,2x128,1x128,2x32+4x32,2x32,1x32,2x28+4x28,2x28,1x28}]
               [--bs_train BS_TRAIN] [--bs_eval BS_EVAL] [--label_noise_ratio LABEL_NOISE_RATIO] [--logit_noise_temp LOGIT_NOISE_TEMP] [--resample_target_noise RESAMPLE_TARGET_NOISE] [--augs [AUGS [AUGS ...]]] [--per_crop_augs [PER_CROP_AUGS [PER_CROP_AUGS ...]]]
               [--enc {ConvNet,convnet_16_1,convnet_16_1e,convnet_16_2,convnet_16_2e,convnet_32_1,convnet_32_1e,convnet_32_2,convnet_32_2e,ResNet,resnet18,resnet34,resnet20,resnet56,VGG,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,Flatten}]
               [--enc_norm_layer {BatchNorm,InstanceNorm,GroupNorm8,LayerNorm,Identity}] [--tiny_input] [--mlp_act {ReLU,GELU}] [--mlp_bn] [--hid_dims [HID_DIMS [HID_DIMS ...]]] [--l2bot_dim L2BOT_DIM] [--l2bot_cfg L2BOT_CFG] [--out_dim OUT_DIM] [--t_init {t_ckpt,random,s_ckpt}] [--t_init_seed T_INIT_SEED]
               [--s_init {random,neighborhood,t_ckpt,s_ckpt,teacher,interpolated}] [--s_init_seed S_INIT_SEED] [--s_init_alpha S_INIT_ALPHA] [--s_init_eps S_INIT_EPS] [--s_init_var_preserving S_INIT_VAR_PRESERVING] [--s_mode {supervised,distillation}] [--t_mode {ema,prev_epoch,no_update}] [--t_mom T_MOM]
               [--t_update_every T_UPDATE_EVERY] [--t_bn_mode {from_data,from_student}] [--t_eval T_EVAL] [--t_cmom T_CMOM] [--s_cmom S_CMOM] [--t_temp T_TEMP] [--s_temp S_TEMP] [--loss {H_pred,KL,CE,MSE}] [--loss_pairing {all,same,opposite}] [--n_epochs N_EPOCHS] [--opt {adam,adamw,sgd}] [--opt_lr OPT_LR]
               [--opt_wd OPT_WD] [--opt_mom OPT_MOM] [--opt_beta1 OPT_BETA1] [--opt_beta2 OPT_BETA2] [--clip_grad CLIP_GRAD] [--wn_freeze_epochs WN_FREEZE_EPOCHS] [--probe_every PROBE_EVERY] [--probing_epochs PROBING_EPOCHS] [--probing_k PROBING_K] [--normalize_probe NORMALIZE_PROBE]
               [--prober_seed PROBER_SEED] [--save_features [{embeddings,projections,logits,all} [{embeddings,projections,logits,all} ...]]] [--save_paramstats [{student,teacher,logits,all} [{student,teacher,logits,all} ...]]]

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
  --float64 FLOAT64     Wether to set default to float64.

Data:
  --dataset {MNIST,CIFAR10,CIFAR100}
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
  --resample_target_noise RESAMPLE_TARGET_NOISE
                        Resample the the logits/labels at every access.
  --augs [AUGS [AUGS ...]]
                        Augmentation(s) to apply to Dataset. Supply multiple names as list or a string joined by '_'.
  --per_crop_augs [PER_CROP_AUGS [PER_CROP_AUGS ...]]
                        Augmentation(s) to apply to each crop individually. Supply multiple names as list or a string joined by '_'.

Model:
  --enc {ConvNet,convnet_16_1,convnet_16_1e,convnet_16_2,convnet_16_2e,convnet_32_1,convnet_32_1e,convnet_32_2,convnet_32_2e,ResNet,resnet18,resnet34,resnet20,resnet56,VGG,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,Flatten}
                        Defines the model to train on.
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
  --t_init_seed T_INIT_SEED
                        The seed for teacher initialization, use numbers with good balance of 0 and 1 bits. None will set a new seed randomly.
  --s_init_seed S_INIT_SEED
                        The seed for student initialization, use numbers with good balance of 0 and 1 bits. None will reuse teacher generator.

DINO:
  --t_init {t_ckpt,random,s_ckpt}
                        Initialization of teacher, specify '--ckpt_path'.
  --s_init {random,neighborhood,t_ckpt,s_ckpt,teacher,interpolated}
                        Initialization of student, specify '--ckpt_path'.
  --s_init_alpha S_INIT_ALPHA
                        Alpha for interpolated random initialization of student.
  --s_init_eps S_INIT_EPS
                        Epsilon for neighborhood random initialization of student.
  --s_init_var_preserving S_INIT_VAR_PRESERVING
                        Apply variance preserving correction for 'interpolated' and 'neighborhood' s_init
  --s_mode {supervised,distillation}
                        Mode of student update.
  --t_mode {ema,prev_epoch,no_update}
                        Mode of teacher update.
  --t_mom T_MOM         Teacher momentum for exponential moving average (float or Schedule).
  --t_update_every T_UPDATE_EVERY
                        Teacher update frequency for prev_epoch mode.
  --t_bn_mode {from_data,from_student}
                        Mode of teacher batchnorm updates: either from data stats or from student buffers.
  --t_eval T_EVAL       Run teacher in evaluation mode even on training data.
  --t_cmom T_CMOM       Teacher centering momentum of DINOHead (float or Schedule).
  --s_cmom S_CMOM       Student centering momentum of DINOHead (float or Schedule).
  --t_temp T_TEMP       Teacher temperature of DINOHead (float or Schedule).
  --s_temp S_TEMP       Student temperature of DINOHead (float or Schedule).
  --loss {H_pred,KL,CE,MSE}
                        Loss function to use in the multicrop loss.
  --loss_pairing {all,same,opposite}
                        Pairing strategy for the multicrop views in the loss function.

Training:
  --n_epochs N_EPOCHS   Number of epochs to train for.
  --opt {adam,adamw,sgd}
                        Optimizer to use for training.
  --opt_lr OPT_LR       Learning rate for optimizer (float or Schedule): specified wrt batch size 256 and linearly scaled.
  --opt_wd OPT_WD       Weight decay for optimizer (float or Schedule).
  --opt_mom OPT_MOM     Momentum for SGD optimizer.
  --opt_beta1 OPT_BETA1
                        Beta1 for Adam(W) optimizer.
  --opt_beta2 OPT_BETA2
                        Beta2 for Adam(W) optimizer.
  --clip_grad CLIP_GRAD
                        Value to clip gradient norm to.
  --wn_freeze_epochs WN_FREEZE_EPOCHS
                        Epochs to freeze WeightNormalizedLinear layer in DINOHead.

addons:
  --probe_every PROBE_EVERY
                        Probe every so many epochs during training.
  --probing_epochs PROBING_EPOCHS
                        Number of epochs to train for linear probing.
  --probing_k PROBING_K
                        Amount of neighbors for k-nearest neighbor probing.
  --normalize_probe NORMALIZE_PROBE
                        Apply feature normalization (standardization) for probing.
  --prober_seed PROBER_SEED
                        The seed for reproducible probing, use numbers with good balance of 0 and 1 bits.
  --save_features [{embeddings,projections,logits,all} [{embeddings,projections,logits,all} ...]]
                        Save features for embeddings, projections and/or logits.
  --save_paramstats [{student,teacher,logits,all} [{student,teacher,logits,all} ...]]
                        Save layerwise parameter and gradient statistics for teacher and/or student.
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
