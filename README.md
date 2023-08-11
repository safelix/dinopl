# Random Teachers are Good Teachers

This repo provides a pytorch-lightning implementation of self-distillation with no labels DINO, its instanciation within a configurable experiment bed, as well as tools to investigate the method in various settings. It is used to investiagate the role of teacher networks under a very simplified setting in the paper 'Random Teachers are Good Teachers'. By default everything is logged to WANDB.

### Installation

Currently tested with `python==3.8.5`, `torch==2.0.0+cu117`:
```
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
```
Install with `pip install -e .[all]`. Use `pip install -e .` to only install the core dependencies, `[tracking]` for advanced tracking functionality and `[notebook]` to run notebooks with results.

Specify data and results path in environment `DINO_DATA` `DINO_RESULTS` variables, e.g. by adding to your .bashrc
```
export DINO_DATA=$HOME/dinopl/datasets/data
export DINO_RESULTS=$HOME/dinopl/results
```
or any other desired path.

### Usage
To run random teacher distillation, please use 

```
python dino.py --from_json configs/cifar10_distillation_v2.json
```

The default configuration of `dino.py` corresponds to the fully fledged method from Caron et al., and is overwritten by the `.json` file. The loaded configuration can be overwritten by specific command-line arguments, such as `--n_epochs 150` or `--enc vgg11`. For a full specification, please refer to the `--help` command: 

```
usage: dino.py [-h] [--from_json FROM_JSON] [--n_workers N_WORKERS] [--seed SEED] [--log_every LOG_EVERY] [--ckpt_path CKPT_PATH] [--force_cpu] [--float64 FLOAT64]
               [--dataset {MNIST,CIFAR10,CIFAR100,TinyImageNet}] [--n_classes N_CLASSES] [--n_samples N_SAMPLES] [--bs_train BS_TRAIN] [--batchaccum BATCHACCUM]
               [--samples_per_epoch SAMPLES_PER_EPOCH] [--bs_eval BS_EVAL] [--label_noise_ratio LABEL_NOISE_RATIO] [--logit_noise_temp LOGIT_NOISE_TEMP]
               [--resample_target_noise RESAMPLE_TARGET_NOISE] [--augs [AUGS [AUGS ...]]]
               [--mc {2x128+4x96,2x128,1x128,2x64+4x64,1x64,2x64,2x32+4x32,2x32,1x32,2x28+4x28,2x28,1x28}] [--per_crop_augs [PER_CROP_AUGS [PER_CROP_AUGS ...]]]
               [--enc {flatten,mlp_512_1,mlp_512_2,mlp_512_3,mlp_512_4,mlp_512_5,mlp_1024_1,mlp_1024_2,mlp_1024_3,mlp_1024_4,mlp_1024_5,convnet_16_1,convnet_16_2,convnet_16_3,convnet_16_4,convnet_16_5,convnet_32_1,convnet_32_2,convnet_32_3,convnet_32_4,convnet_32_5,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,resnet18,resnet34,resnet50,resnet20,resnet56,vit_tiny,vit_small,vit_medium,vit_base}]
               [--enc_norm_layer {BatchNorm,InstanceNorm,GroupNorm8,LayerNorm,Identity}] [--tiny_input] [--head_init_method {default,trunc_normal}]
               [--mlp_act {ReLU,GELU}] [--mlp_bn] [--hid_dims [HID_DIMS [HID_DIMS ...]]] [--l2bot_dim L2BOT_DIM] [--l2bot_cfg L2BOT_CFG] [--out_dim OUT_DIM]
               [--t_init {random,s_ckpt,t_ckpt}] [--t_init_seed T_INIT_SEED] [--s_init {s_ckpt,random,t_ckpt,neighborhood,interpolated,teacher}]
               [--s_init_seed S_INIT_SEED] [--s_init_alpha S_INIT_ALPHA] [--s_init_eps S_INIT_EPS] [--s_init_var_preserving S_INIT_VAR_PRESERVING]
               [--s_mode {supervised,distillation}] [--t_mode {ema,no_update,prev_epoch}] [--t_mom T_MOM] [--t_update_every T_UPDATE_EVERY]
               [--t_bn_mode {from_data,from_student}] [--t_eval T_EVAL] [--t_cmom T_CMOM] [--s_cmom S_CMOM] [--t_temp T_TEMP] [--s_temp S_TEMP]
               [--loss {MSE,H_pred,KL,CE}] [--loss_pairing {all,same,opposite}] [--n_epochs N_EPOCHS] [--n_steps N_STEPS] [--opt {adam,adamw,sgd}] [--opt_lr OPT_LR]
               [--opt_wd OPT_WD] [--opt_mom OPT_MOM] [--opt_beta1 OPT_BETA1] [--opt_beta2 OPT_BETA2] [--clip_grad CLIP_GRAD] [--wn_freeze_epochs WN_FREEZE_EPOCHS]
               [--probe_every PROBE_EVERY] [--probing_epochs PROBING_EPOCHS] [--probing_k PROBING_K] [--normalize_probe NORMALIZE_PROBE] [--prober_seed PROBER_SEED]
               [--track_feathist TRACK_FEATHIST] [--track_gradvar TRACK_GRADVAR] [--save_ckpt [{probe_student,loss_max} [{probe_student,loss_max} ...]]]
               [--save_features [{embeddings,projections,logits,all} [{embeddings,projections,logits,all} ...]]]
               [--save_paramstats [{student,teacher,logits,all} [{student,teacher,logits,all} ...]]]

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
  --dataset {MNIST,CIFAR10,CIFAR100,TinyImageNet}
                        Datset to train on.
  --n_classes N_CLASSES
                        Number of classes. By default determined from dataset but can be overwritten for logit noise.
  --n_samples N_SAMPLES
                        Number of samples used for training. Use a deterministic, stratified subset.
  --bs_train BS_TRAIN   Batch size for the training set.
  --batchaccum BATCHACCUM
                        How many batches to accumulate for one gradient update. If -1, full batch is used.
  --samples_per_epoch SAMPLES_PER_EPOCH
                        Number of samples used by the dataloader per epoch.
  --bs_eval BS_EVAL     Batch size for valid/test set.
  --label_noise_ratio LABEL_NOISE_RATIO
                        Add label noise (random assignemt) for supervised training.
  --logit_noise_temp LOGIT_NOISE_TEMP
                        Add logit noise (sharpened gaussian logits) for supervised training.
  --resample_target_noise RESAMPLE_TARGET_NOISE
                        Resample the the logits/labels at every access.
  --augs [AUGS [AUGS ...]]
                        Augmentation(s) to apply to Dataset. Supply multiple names as list or a string joined by '_'.
  --mc {2x128+4x96,2x128,1x128,2x64+4x64,1x64,2x64,2x32+4x32,2x32,1x32,2x28+4x28,2x28,1x28}
                        Specification of multicrop augmentation.
  --per_crop_augs [PER_CROP_AUGS [PER_CROP_AUGS ...]]
                        Augmentation(s) to apply to each crop individually. Supply multiple names as list or a string joined by '_'.

Model:
  --enc {flatten,mlp_512_1,mlp_512_2,mlp_512_3,mlp_512_4,mlp_512_5,mlp_1024_1,mlp_1024_2,mlp_1024_3,mlp_1024_4,mlp_1024_5,convnet_16_1,     convnet_16_2,convnet_16_3,convnet_16_4,convnet_16_5,convnet_32_1,convnet_32_2,convnet_32_3,convnet_32_4,convnet_32_5,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,resnet18,resnet34,resnet50,resnet20,resnet56,vit_tiny,vit_small,vit_medium,vit_base}
                        Defines the model to train on.
  --enc_norm_layer {BatchNorm,InstanceNorm,GroupNorm8,LayerNorm,Identity}
                        Overwrite the normalization layer of the model if supported.
  --tiny_input          Adjust encoder for tiny inputs, e.g. resnet for cifar 10.
  --head_init_method {default,trunc_normal}
                        Initialization method for linear layers in head, 'default' refers to the torch default, but DINO uses 'trunc_normal'.
  --mlp_act {ReLU,GELU}
                        Activation function of DINOHead MLP.
  --mlp_bn              Use batchnorm in DINOHead MLP.
  --hid_dims [HID_DIMS [HID_DIMS ...]]
                        Hidden dimensions of DINOHead MLP.
  --l2bot_dim L2BOT_DIM
                        L2-Bottleneck dimension of DINOHead MLP. If 0, bottleneck is replaced by linear.
  --l2bot_cfg L2BOT_CFG
                        L2-Bottleneck configuration string: '{wn,-}/{l,lb,-}/{fn,-}/{wn,-}/{l,lb,-}/{wn,-}'.
  --out_dim OUT_DIM     Output dimension of the DINOHead MLP.

DINO:
  --t_init {random,s_ckpt,t_ckpt}
                        Initialization of teacher, specify '--ckpt_path'.
  --t_init_seed T_INIT_SEED
                        The seed for teacher initialization, use numbers with good balance of 0 and 1 bits. None will set a new seed randomly.
  --s_init {s_ckpt,random,t_ckpt,neighborhood,interpolated,teacher}
                        Initialization of student, specify '--ckpt_path'.
  --s_init_seed S_INIT_SEED
                        The seed for student initialization, use numbers with good balance of 0 and 1 bits. None will reuse teacher generator.
  --s_init_alpha S_INIT_ALPHA
                        Alpha for interpolated random initialization of student.
  --s_init_eps S_INIT_EPS
                        Epsilon for neighborhood random initialization of student.
  --s_init_var_preserving S_INIT_VAR_PRESERVING
                        Apply variance preserving correction for 'interpolated' and 'neighborhood' s_init
  --s_mode {supervised,distillation}
                        Mode of student update.
  --t_mode {ema,no_update,prev_epoch}
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
  --loss {MSE,H_pred,KL,CE}
                        Loss function to use in the multicrop loss.
  --loss_pairing {all,same,opposite}
                        Pairing strategy for the multicrop views in the loss function.

Training:
  --n_epochs N_EPOCHS   Number of epochs to train for.
  --n_steps N_STEPS     Number of steps to train for, stops at min(n_epochs, n_steps).
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
  --track_feathist TRACK_FEATHIST
                        Track gradient variances of model, encoder and head.
  --track_gradvar TRACK_GRADVAR
                        Track gradient variances of model, encoder and head.
  --save_ckpt [{probe_student,loss_max} [{probe_student,loss_max} ...]]
                        Save checkpoints for specific types of metrics.
  --save_features [{embeddings,projections,logits,all} [{embeddings,projections,logits,all} ...]]
                        Save features for embeddings, projections and/or logits.
  --save_paramstats [{student,teacher,logits,all} [{student,teacher,logits,all} ...]]
                        Save layerwise parameter and gradient statistics for teacher and/or student.
```


## Citation
```
@inproceedings{sarnthein2023random,
  title = {Random Teachers are Good Teachers},
  author = {Sarnthein, Felix and Bachmann, Gregor and Anagnostidis, Sotiris and Hofmann, Thomas},
  booktitle = {International Conference on Machine Learning},
  year = {2023},
  url = {https://arxiv.org/abs/2302.12091},
  doi = {10.48550/ARXIV.2302.12091},
}

@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
