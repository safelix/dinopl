'''
Define common constants that are used globally.

Also define a configuration object whose parameters can be set via the command line or loaded from an existing
json file. Here you can add more configuration parameters that should be exposed via the command line. In the code,
you can access them via `config.your_parameter`. All parameters are automatically saved to disk in JSON format.

Copyright ETH Zurich, Manuel Kaufmann & Felix Sarnthein

'''
import argparse
import copy
import json
import math
import os
import pprint
import typing
import warnings

import torch

import dinopl.utils as U
import models
import datasets
from dinopl import DINO, DINOModel, DINOHead
from dinopl.scheduling import *
import torchvision
from torch.utils.data import DataLoader


class Constants(object):
    '''
    This is a singleton.
    '''
    class __Constants:
        def __init__(self):
            # Environment setup.
            self.DTYPE = torch.float32

            
            # Set C.DEVICE
            if torch.cuda.is_available():
                self.DEVICE = torch.device('cuda')
                GPU = torch.cuda.get_device_name(torch.cuda.current_device())
                print(f'Torch ({torch.__version__}) running on {GPU}...' , flush=True)
            else:
                self.DEVICE = torch.device('cpu')
                print(f'Torch ({torch.__version__}) running on CPU...', flush=True)


            # Get directories from os.environ
            try: 
                self.DATA_DIR = os.environ['DINO_DATA']
                self.RESULTS_DIR = os.environ['DINO_RESULTS']
            except KeyError:
                warnings.warn(
                    '''Please configure the environment variables: 
                    - DINO_DATA: path to datasets
                    - DINO_RESULTS: path to store results''', stacklevel=4)

            print('SLURM_JOB_ID: ' + os.environ.get('SLURM_JOB_ID', ''), flush=True)
            print('SLURM_ARRAY_TASK_ID: ' + os.environ.get('SLURM_ARRAY_TASK_ID', ''), flush=True)

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)

CONSTANTS = Constants()

class Configuration(object):
    f'''Configuration parameters exposed via the commandline.'''

    def __init__(self, adict:dict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4, sort_dicts=False)

    @staticmethod
    def parser(pre_parser : argparse.ArgumentParser = None) -> argparse.ArgumentParser:

        # Argument parser.
        parents = [] if pre_parser is None else [pre_parser]
        parser = argparse.ArgumentParser(parents=parents,  
                    formatter_class=argparse.RawTextHelpFormatter)

        # General.
        general = parser.add_argument_group('General')
        general.add_argument('--n_workers', type=int, default=4, 
                            help='Number of parallel threads for data loading.')
        general.add_argument('--seed', type=int, default=None,
                            help='Random number generator seed.')
        general.add_argument('--log_every', type=int, default=1,
                            help='Log every so many steps.')
        general.add_argument('--ckpt_path', type=os.path.expandvars, default='',
                            help='Path to checkpoint, used by \'--t_init\'.')
        #general.add_argument('--resume', action='store_true',
        #                    help='Resume training from checkpoint specified by \'--ckpt_path\'.')
        general.add_argument('--force_cpu', action='store_true',
                            help='Force training on CPU instead of GPU.')
        general.add_argument('--float64', type=U.bool_parser, default=False,
                            help='Wether to set default to float64.')

        # Data.
        data = parser.add_argument_group('Data')
        data.add_argument('--dataset', choices=datasets.__all__, default='mnist',
                            help='Datset to train on.')
        data.add_argument('--n_classes', type=int, default=None,
                            help='Number of classes. By default determined from dataset but can be overwritten for logit noise.')
        data.add_argument('--n_samples', type=int, default=None,
                            help='Number of samples used for training. Use a deterministic, stratified subset.')
        data.add_argument('--bs_train', type=int, default=64, 
                            help='Batch size for the training set.')
        data.add_argument('--batchaccum', type=int, default=1, 
                            help='How many batches to accumulate for one gradient update. If -1, full batch is used.')
        data.add_argument('--samples_per_epoch', type=int, default=None,
                            help='Number of samples used by the dataloader per epoch.')
        data.add_argument('--bs_eval', type=int, default=256, 
                            help='Batch size for valid/test set.')
        data.add_argument('--label_noise_ratio', type=float, default=0,
                            help='Add label noise (random assignemt) for supervised training.')
        data.add_argument('--logit_noise_temp', type=float, default=0,
                            help='Add logit noise (sharpened gaussian logits) for supervised training.')
        data.add_argument('--resample_target_noise', type=bool, default=False,
                            help='Resample the the logits/labels at every access.')
        data.add_argument('--inputs_as_logits', type=U.bool_parser, default=False,
                            help='Use the inputs as logits and train in autoencoder fashion.')
        data.add_argument('--augs', type=str, nargs='*', default=[],
                            help='Augmentation(s) to apply to Dataset. ' 
                            +'Supply multiple names as list or a string joined by \'_\'.')
        data.add_argument('--mc', type=str, choices=[
                                '2x128+4x96', '2x128', '1x128',
                                '2x64+4x64', '1x64', '2x64',
                                '2x32+4x32', '2x32', '1x32',
                                '2x28+4x28', '2x28', '1x28'], 
                                default='2x128+4x96',
                            help='Specification of multicrop augmentation.')
        data.add_argument('--per_crop_augs', type=str, nargs='*', default=[],
                            help='Augmentation(s) to apply to each crop individually. ' 
                            +'Supply multiple names as list or a string joined by \'_\'.')

        # Model.
        model = parser.add_argument_group('Model')
        model.add_argument('--enc', type=str, choices=models.__all__, default='resnet18', 
                            help='Defines the model to train on.')
        model.add_argument('--enc_norm_layer', type=str, choices=['BatchNorm', 'InstanceNorm', 'GroupNorm8', 'LayerNorm', 'Identity'], default=None,
                            help='Overwrite the normalization layer of the model if supported.')
        model.add_argument('--tiny_input', action='store_true', 
                            help='Adjust encoder for tiny inputs, e.g. resnet for cifar 10.')
        model.add_argument('--head_init_method', type=str, choices=['default', 'trunc_normal'], default='trunc_normal',
                           help='Initialization method for linear layers in head, \'default\' refers to the torch default, but DINO uses \'trunc_normal\'.')
        model.add_argument('--mlp_act', type=str, choices={'GELU', 'ReLU'}, default='GELU',
                            help='Activation function of DINOHead MLP.')
        model.add_argument('--mlp_bn', action='store_true',
                            help='Use batchnorm in DINOHead MLP.')
        model.add_argument('--hid_dims', type=int, default=[2048, 2048], nargs='*',
                            help='Hidden dimensions of DINOHead MLP.')
        model.add_argument('--l2bot_dim', type=int, default=256,
                            help='L2-Bottleneck dimension of DINOHead MLP. If 0, bottleneck is replaced by linear.')
        model.add_argument('--l2bot_cfg', type=str, default='-/lb/fn/wn/l/-',
                            help='L2-Bottleneck configuration string: \'{wn,-}/{l,lb,-}/{fn,fnd,-}/{wn,-}/{l,lb,-}/{fn,fnd,-}\'.')
        model.add_argument('--out_dim', type=int, default=65536, 
                            help='Output dimension of the DINOHead MLP.')


        # Teacher Update, Temperature, Centering
        dino = parser.add_argument_group('DINO')
        dino.add_argument('--t_init', type=str, choices={'random', 's_ckpt', 't_ckpt'}, default='random',
                            help='Initialization of teacher, specify \'--ckpt_path\'.')
        dino.add_argument('--t_init_seed', type=int, default=None,
                            help='The seed for teacher initialization, use numbers with good balance of 0 and 1 bits. None will set a new seed randomly.')
        dino.add_argument('--s_init', type=str, choices={'teacher', 's_ckpt', 't_ckpt', 'random', 'interpolated', 'neighborhood'}, default='teacher',
                            help='Initialization of student, specify \'--ckpt_path\'.')
        dino.add_argument('--s_init_seed', type=int, default=None,
                            help='The seed for student initialization, use numbers with good balance of 0 and 1 bits. None will reuse teacher generator.')
        dino.add_argument('--s_init_alpha', type=float, default=0,
                            help='Alpha for interpolated random initialization of student.')
        dino.add_argument('--s_init_eps', type=float, default=0,
                            help='Epsilon for neighborhood random initialization of student.')
        dino.add_argument('--s_init_var_preserving', type=U.bool_parser, default=False,
                            help='Apply variance preserving correction for \'interpolated\' and \'neighborhood\' s_init')
        dino.add_argument('--s_mode', type=str, choices={'supervised', 'distillation'}, default='distillation',
                            help='Mode of student update.')
        dino.add_argument('--t_mode', type=str, choices={'ema', 'prev_epoch', 'no_update'}, default='ema',
                            help='Mode of teacher update.')
        dino.add_argument('--t_mom', type=str, default=str(CosSched(0.996, 1)),
                            help='Teacher momentum for exponential moving average (float or Schedule).')
        dino.add_argument('--t_update_every', type=int, default=1,
                            help='Teacher update frequency for prev_epoch mode.')
        dino.add_argument('--t_bn_mode', type=str, choices={'from_data', 'from_student'}, default='from_data',
                            help='Mode of teacher batchnorm updates: either from data stats or from student buffers.')
        dino.add_argument('--t_eval', type=U.bool_parser, default=False,
                            help='Run teacher in evaluation mode even on training data.')
        dino.add_argument('--t_cmom', type=str, default=str(ConstSched(0.9)), 
                            help='Teacher centering momentum of DINOHead (float or Schedule).')
        dino.add_argument('--s_cmom', type=str, default=str(ConstSched(torch.nan)), 
                            help='Student centering momentum of DINOHead (float or Schedule).')
        dino.add_argument('--t_temp', type=str, default=str(LinWarmup(0.04, 0.04, 0)), 
                            help='Teacher temperature of DINOHead (float or Schedule).')
        dino.add_argument('--s_temp', type=str, default=str(ConstSched(0.1)), 
                            help='Student temperature of DINOHead (float or Schedule).')
        dino.add_argument('--loss', type=str, choices={'CE', 'KL', 'H_pred', 'MSE'}, default='CE',
                            help='Loss function to use in the multicrop loss.')
        dino.add_argument('--loss_pairing', type=str, choices=['all', 'same', 'opposite'], default='opposite',
                            help='Pairing strategy for the multicrop views in the loss function.')


        
        # Training configurations.        
        training = parser.add_argument_group('Training')
        training.add_argument('--n_epochs', type=int, default=100, 
                            help='Number of epochs to train for.')
        training.add_argument('--n_steps', type=int, default=-1, 
                            help='Number of steps to train for, stops at min(n_epochs, n_steps).')
        training.add_argument('--stop_on_non_finite', type=U.bool_parser, default=True,
                              help='Stop training if some parameters are non-finite at end of an epoch.')
        training.add_argument('--opt', type=str, choices={'adamw', 'adam', 'sgd'}, default='adamw', 
                            help='Optimizer to use for training.')                   
        training.add_argument('--opt_lr', type=str, default=str(CatSched(LinSched(0, 5e-4), CosSched(5e-4, 1e-6), 10)), 
                            help='Learning rate for optimizer (float or Schedule): specified wrt batch size 256 and linearly scaled.')
        training.add_argument('--opt_wd', type=str, default=str(CosSched(0.04, 0.4)), 
                            help='Weight decay for optimizer (float or Schedule).')
        training.add_argument('--opt_mom', type=float, default=0.9, 
                            help='Momentum for SGD optimizer.')
        training.add_argument('--opt_beta1', type=float, default=0.9, 
                            help='Beta1 for Adam(W) optimizer.')
        training.add_argument('--opt_beta2', type=float, default=0.999, 
                            help='Beta2 for Adam(W) optimizer.')
        training.add_argument('--clip_grad', type=float, default=3, 
                            help='Value to clip gradient norm to.')
        training.add_argument('--wn_freeze_epochs', type=int, default=1,
                            help='Epochs to freeze WeightNormalizedLinear layer in DINOHead.')


        # Probing configurations
        addons = parser.add_argument_group('addons')
        addons.add_argument('--validation_freq', type=U.floatint_parser, default=1, 
                            help='Validation frequency, if (int) in epochs if (float) in ratio of total steps.')       
        addons.add_argument('--probe_every', type=int, default=1, 
                            help='Probe every so many epochs during training.')
        addons.add_argument('--probing_epochs', type=int, default=10, 
                            help='Number of epochs to train for linear probing.')
        addons.add_argument('--probing_k', type=int, default=20, 
                            help='Amount of neighbors for k-nearest neighbor probing.')
        addons.add_argument('--normalize_probe', type=U.bool_parser, default=True, 
                            help='Apply feature normalization (standardization) for probing.')
        addons.add_argument('--prober_seed', type=int, default=None,
                            help='The seed for reproducible probing, use numbers with good balance of 0 and 1 bits.')
        addons.add_argument('--track_feathist', type=U.bool_parser, default=False,
                            help='Track gradient variances of model, encoder and head.')
        addons.add_argument('--track_gradvar', type=U.bool_parser, default=False,
                            help='Track gradient variances of model, encoder and head.')
        # specify lists as: --argument choice1 choice2
        addons.add_argument('--save_ckpt', type=str, nargs='*', default=['probe_student'],
                            choices=['probe_student', 'loss_max', 'rank_min', 'none'],
                            help='Save checkpoints for specific types of metrics.')
        addons.add_argument('--save_features', type=str, nargs='*', default=[],
                            choices=['embeddings', 'projections', 'logits', 'all'],   
                            help='Save features for embeddings, projections and/or logits.')
        addons.add_argument('--save_paramstats', type=str, nargs='*', default=[],
                            choices=['student', 'teacher', 'logits', 'all'],   
                            help='Save layerwise parameter and gradient statistics for teacher and/or student.')
        
        return parser
    

    @staticmethod
    def get_default():
        parser = Configuration.parser()
        defaults = parser.parse_args([])
        return Configuration(vars(defaults))


    @staticmethod
    def from_json(json_path:str, default_config=None):
        '''Load configurations from a JSON file.'''

        # Get default configuration
        if default_config is None:
            default_config = Configuration.get_default()

        # Load configuration from json file
        with open(json_path, 'r') as f:
            json_config = json.load(f) 

        # Overwrite defaults
        default_config.update(json_config, allow_new_keys=True)

        return default_config

    @staticmethod
    def parse_cmd():
        '''Loading configuration according to priority:
        1. from commandline arguments
        2. from JSON configuration file
        3. from parser default values.'''
        
        # Initial parser.
        pre_parser = argparse.ArgumentParser(add_help=False)

        pre_parser.add_argument('--from_json', type=str,
                            help=Configuration.parse_cmd.__doc__)

        # Argument parser.
        parser = Configuration.parser(pre_parser)

        # 1. Get defaults from parser
        config = Configuration.get_default()

        # 2. Overwrite with defaults with JSON config
        pre_args, remaining_argv = pre_parser.parse_known_args()
        if pre_args.from_json is not None:
            json_path = pre_args.from_json 
            config = Configuration.from_json(json_path, config)
            config.from_json = pre_args.from_json

        # 3. Overwrite JSON config with remaining cmd args
        parser.parse_args(remaining_argv, config)

        return config


    def to_json(self, json_path:str):
        '''Dump configurations to a JSON file.'''
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2)
            f.write(s)

    def update(self, adict:dict, allow_new_keys=False):
        new_keys = adict.keys() - self.__dict__.keys()
        
        if not allow_new_keys and len(new_keys) > 0:
            raise RuntimeError(f'Cannot update configuration with new keys {new_keys}.')
        self.__dict__.update(adict)

def get_enc_norm_layer(config:Configuration) -> typing.Type[torch.nn.Module]:

    if config.enc_norm_layer == 'BatchNorm':
        return (lambda dim: torch.nn.BatchNorm2d(dim, affine=True, track_running_stats=True))

    if config.enc_norm_layer == 'InstanceNorm':
        return (lambda dim: torch.nn.GroupNorm(dim, dim, affine=True))

    if config.enc_norm_layer == 'GroupNorm8':
        return (lambda dim: torch.nn.GroupNorm(dim//8, dim, affine=True))

    if config.enc_norm_layer == 'LayerNorm':
        return (lambda dim: torch.nn.GroupNorm(1, dim, affine=True))

    if config.enc_norm_layer == 'Identity':
        return (lambda dim: torch.nn.Identity())

    raise RuntimeError('Unkown normalization layer name.')


def get_encoder(config:Configuration) -> typing.Type[models.Encoder]:
    '''
    This is a helper function that can be useful if you have several model definitions that you want to
    choose from via the command line.
    '''

    input_pixels = config.mc_spec[0]['out_size'] ** 2

    if config.enc in ['flatten', 'Flatten']:
        return (lambda : models.flatten(n_pixels=input_pixels, n_channels=3))

    if config.enc in models.__dict__.keys():
        # prepare keyword arguments
        kwargs = dict(num_classes=None)
        if 'mlp' in config.enc.lower():
            kwargs['in_numel'] = 3 * input_pixels

        if 'convnet' in config.enc.lower():
            kwargs['img_size'] = config.mc_spec[0]['out_size'] 

        if 'resnet' in config.enc.lower():
            kwargs['tiny_input'] = getattr(config, 'tiny_input', False)

        if 'vit' in config.enc.lower():
            kwargs['img_chans'] = 3
            kwargs['img_size'] = config.mc_spec[0]['out_size'] 
            if getattr(config, 'tiny_input', False):
                kwargs['patch_size'] = 8


        if getattr(config, 'enc_norm_layer', None) is not None:
            kwargs['norm_layer'] = get_enc_norm_layer(config)

        return (lambda : models.__dict__[config.enc](**kwargs))

    raise RuntimeError('Unkown model name.')


def init_student_teacher(config:Configuration, model:DINOModel) -> typing.Tuple[DINOModel, DINOModel]:
    
    t_generator = torch.Generator()
    s_generator = torch.Generator()
    if config.t_init_seed is None:
        config.t_init_seed = t_generator.seed()
    else:
        t_generator.manual_seed(config.t_init_seed)

    if config.s_init_seed is None:
        s_generator = t_generator
    else:
        s_generator.manual_seed(config.s_init_seed)


    # load checkpoint if required
    if config.t_init in ['s_ckpt', 't_ckpt'] or config.s_init in ['s_ckpt', 't_ckpt']:
        if getattr(config, 'ckpt_path', '') == '':
            raise RuntimeError('Student or teacher inititalization strategy requires \'--ckpt_path\' to be specified.')
        temp_student = copy.deepcopy(model) # required to load state dict into instanciated copy
        temp_teacher = copy.deepcopy(model) # required to load state dict into instanciated copy
        dino_ckpt = DINO.load_from_checkpoint(config.ckpt_path,  mc_spec=config.mc_spec, student=temp_student, teacher=temp_teacher)
    
    # Initialize teacher network
    if config.t_init == 's_ckpt':
        teacher = copy.deepcopy(dino_ckpt.student)     # make teacher from student checkpoint
    elif config.t_init == 't_ckpt':
        teacher = copy.deepcopy(dino_ckpt.teacher)     # make teacher from teacher checkpoint
    elif config.t_init == 'random':
        teacher = copy.deepcopy(model)  # make teacher with random params
        teacher.reset_parameters(generator=t_generator)
    else:
        raise RuntimeError(f'Teacher initialization strategy \'{config.t_init}\' not supported.')

    # Initialize student network
    if config.s_init == 'teacher':
        student = copy.deepcopy(teacher) # make student with same params as teacher
    elif config.s_init == 's_ckpt':
        student = copy.deepcopy(dino_ckpt.student)     # make student from student checkpoint
    elif config.s_init == 't_ckpt':
        student = copy.deepcopy(dino_ckpt.teacher)     # make student from teacher checkpoint
    elif config.s_init == 'random':     
        student = copy.deepcopy(model)
        student.reset_parameters(generator=s_generator)  # initialize student with random parameters
    elif config.s_init == 'interpolated':
        student = copy.deepcopy(model)
        student.reset_parameters(generator=s_generator)  # initialize student with random parameters
        for p_t, p_s in zip(teacher.parameters(), student.parameters()):
            alpha = config.s_init_alpha
            p_s.data = (1 - alpha) * p_t + alpha * p_s # interpolate between teacher and random
            if config.s_init_var_preserving:
                p_s.data /= math.sqrt(2*(alpha**2) - 2*alpha + 1)   # apply variance preserving correction
    elif config.s_init == 'neighborhood':
        student = copy.deepcopy(model)
        student.reset_parameters(generator=s_generator)  # initialize student with random parameters
        for p_t, p_s in zip(teacher.parameters(), student.parameters()):
            eps = config.s_init_eps
            p_s.data = p_t + eps * p_s # add eps neighborhood to teacher
            if config.s_init_var_preserving:
                p_s.data /= math.sqrt(eps**2 + 1)   # apply variance preserving correction
    else:
        raise RuntimeError(f'Student initialization strategy \'{config.s_init}\' not supported.')

    return student, teacher

def create_optimizer(config:Configuration) -> torch.optim.Optimizer:
    '''
    This is a helper function that can be useful if you have optimizers that you want to
    choose from via the command line.
    '''
    config.opt = config.opt.lower()

    if config.opt == 'adamw':
        return (lambda *args, **kwargs: torch.optim.AdamW(*args, betas=(config.opt_beta1, config.opt_beta2), foreach=False, **kwargs)) # TODO: pytorch 2.0 uses foreach to accelerate by default

    if config.opt == 'adam':
        return (lambda *args, **kwargs: torch.optim.Adam(*args, betas=(config.opt_beta1, config.opt_beta2), foreach=False, **kwargs)) # TODO: pytorch 2.0 uses foreach to accelerate by default

    if config.opt == 'sgd':
        return (lambda *args, **kwargs: torch.optim.SGD(*args, momentum=config.opt_mom, **kwargs))

    raise RuntimeError('Unkown optimizer name.')


def get_dataset(config:Configuration) -> typing.Type[datasets.BaseDataset]:
    '''
    This is a helper function that can be useful if you have several dataset definitions that you want to
    choose from via the command line.
    '''
    # backwards compatibility: if everything is lower make upper
    if all([c.islower() for c in config.dataset]):
        config.dataset = config.dataset.upper() 

    if config.dataset not in datasets.__all__:
        raise RuntimeError('Unkown dataset name.')
    
    DSet:datasets.BaseDataset = datasets.__dict__[config.dataset]
    config.img_size = DSet.img_size
    config.ds_pixels = DSet.ds_pixels
    config.ds_classes = DSet.ds_classes
    config.n_classes = config.ds_classes if config.n_classes is None else config.n_classes
    return DSet


def get_augmentations(config:Configuration, DSet, per_crop=False) -> typing.Callable:
    augnames:typing.List[str] = config.per_crop_augs if per_crop else config.augs

    # make list, if not already
    if not isinstance(augnames, list):
        augnames = [augnames]

    augnames = [augname.split('_') for augname in augnames]  # split augnames with '_'
    augnames = [elem for sublist in augnames for elem in sublist] # flatten nested list
        
    trfm = [] # Get list of transformation
    for augname in augnames:
        if augname not in datasets.augmentation.__all__:
            raise RuntimeError(f'Unkown augmentation name {augname}.')
        trfm.append(datasets.augmentation.__dict__[augname](DSet))
    
    return torchvision.transforms.Compose(trfm)


def create_mc_spec(config:Configuration):
    '''
    This is a helper function that can be useful if you have several multicrop definitions that you want to
    choose from via the command line.
    '''

    if config.mc == '2x128+4x96':
        return [
            {'name':'global1', 'out_size':128, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':128, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'local1', 'out_size':96, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local2', 'out_size':96, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local3', 'out_size':96, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local4', 'out_size':96, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
        ]

    if config.mc == '2x128':
        return [
            {'name':'global1', 'out_size':128, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':128, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
        ] 

    if config.mc == '1x128':
        return [
            {'name':'global1', 'out_size':128, 'min_scale':1.0, 'max_scale':1.0, 'teacher':True, 'student':True},
        ]
    
    if config.mc == '2x64+4x64':
        return [
            {'name':'global1', 'out_size':64, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':64, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'local1', 'out_size':64, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local2', 'out_size':64, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local3', 'out_size':64, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local4', 'out_size':64, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
        ]
    
    if config.mc == '2x64':
        return [
            {'name':'global1', 'out_size':64, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':64, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
        ] 
    
    if config.mc == '1x64':
        return [
            {'name':'global1', 'out_size':64, 'min_scale':1.0, 'max_scale':1.0, 'teacher':True, 'student':True},
        ] 
    
    if config.mc == '2x32+4x32':
        return [
            {'name':'global1', 'out_size':32, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':32, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'local1', 'out_size':32, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local2', 'out_size':32, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local3', 'out_size':32, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local4', 'out_size':32, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
        ]

    if config.mc == '2x32':
        return [
            {'name':'global1', 'out_size':32, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':32, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
        ]   

    if config.mc == '1x32':
        return [
            {'name':'global1', 'out_size':32, 'min_scale':1.0, 'max_scale':1.0, 'teacher':True, 'student':True},
        ]   

    if config.mc == '2x28+4x28':
        return [
            {'name':'global1', 'out_size':28, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':28, 'min_scale':0.4, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'local1', 'out_size':28, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local2', 'out_size':28, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local3', 'out_size':28, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
            {'name':'local4', 'out_size':28, 'min_scale':0.05, 'max_scale':0.4, 'teacher':False, 'student':True},
        ]

    if config.mc == '2x28':
        return [
            {'name':'global1', 'out_size':28, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
            {'name':'global2', 'out_size':28, 'min_scale':0.14, 'max_scale':1.0, 'teacher':True, 'student':True},
        ]   

    if config.mc == '1x28':
        return [
            {'name':'global1', 'out_size':28, 'min_scale':1.0, 'max_scale':1.0, 'teacher':True, 'student':True},
        ]   

    raise RuntimeError('Unkown multicrop name.')

def load_config(identifier:str) -> Configuration:
    if ':' in identifier:
        ckpt_path, name = identifier.split(':')
    else:
        ckpt_path, name = identifier, ''
    config =  Configuration.from_json(os.path.join(os.path.dirname(ckpt_path), 'config.json'))
    config.mc_spec = create_mc_spec(config)
    return config

def load_data(identifier:str, batchsize:int, num_workers:int, pin_memory:bool) -> typing.Tuple[DataLoader, DataLoader]:
    ckpt_path = identifier.split(':')[0] if ':' in identifier else identifier
    config = load_config(ckpt_path)
    
    DSet = get_dataset(config)
    trfm = torchvision.transforms.Compose([ # self-training
                    torchvision.transforms.Lambda(lambda img: img.convert('RGB')), 
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(DSet.mean, DSet.std),
                ])
    #trfm = MultiCrop(config.mc_spec, per_crop_transform=trfm)

    train_ds = DSet(root=os.environ['DINO_DATA'], train=True, transform=trfm, download=False)
    valid_ds = DSet(root=os.environ['DINO_DATA'], train=False, transform=trfm, download=False)
    train_dl = DataLoader(dataset=train_ds, batch_size=batchsize, num_workers=num_workers, pin_memory=pin_memory)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=batchsize, num_workers=num_workers, pin_memory=pin_memory)
    return train_dl, valid_dl

def load_model(identifier:str) -> typing.Union[DINO, DINOModel]:
    ''' Load the DINO model using the checkpoint path.
    Optionally access specific parts of the model by appending `:` followed by a combination of
        - `teacher` (to access only teacher)
        - `student` (to access only teacher)
        - `init`    (to revert model to its initialization)
        - `enc`     (to remove mlp and last_layer)
    '''

    if ':' in identifier:
        ckpt_path, name = identifier.split(':')
    else:
        ckpt_path, name = identifier, ''
    config = load_config(ckpt_path)

    # get configuration and prepare model
    enc = get_encoder(config)()
    config.embed_dim = enc.embed_dim
    head = DINOHead(config.embed_dim, config.out_dim, 
        hidden_dims=config.hid_dims, 
        l2bot_dim=config.l2bot_dim, 
        l2bot_cfg=config.l2bot_cfg,
        use_bn=config.mlp_bn,
        act_fn=config.mlp_act)
    student = DINOModel(enc, head)
    teacher = copy.deepcopy(student)

    # load DINO checkpoint
    dino = DINO.load_from_checkpoint(ckpt_path, map_location='cpu', mc_spec=config.mc_spec, student=student, teacher=teacher)
    
    # init if required by .init suffix
    if 'init' in name:
        student, teacher = init_student_teacher(config, student)
        dino.student = student
        dino.teacher = teacher

    if 'enc' in name:
        for model in [dino.student, dino.teacher]:
            model.head.cent.data = model.head.cent[:model.embed_dim]
            model.head.mlp = torch.nn.Identity()
            model.head.last_layer = torch.nn.Identity()
        
    if 'teacher' in name and 'student' not in name:
        return dino.teacher
    
    if 'student' in name and 'teacher' not in name:
        return dino.student

    return dino
