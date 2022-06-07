'''
Define common constants that are used globally.

Also define a configuration object whose parameters can be set via the command line or loaded from an existing
json file. Here you can add more configuration parameters that should be exposed via the command line. In the code,
you can access them via `config.your_parameter`. All parameters are automatically saved to disk in JSON format.

Copyright ETH Zurich, Manuel Kaufmann & Felix Sarnthein

'''
import argparse
import json
import os
import pprint
import warnings

import torch

from scheduling import *


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
                print(f'Torch running on {GPU}...' , flush=True)
            else:
                self.DEVICE = torch.device('cpu')
                print(f'Torch running on CPU...', flush=True)


            # Get directories from os.environ
            try: 
                self.DATA_DIR = os.environ['DINO_DATA']
                self.RESULTS_DIR = os.environ['DINO_RESULTS']
            except KeyError:
                warnings.warn(
                    '''Please configure the environment variables: 
                    - DINO_DATA: path to datasets
                    - DINO_RESULTS: path to store results''', stacklevel=4)

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

        # Data.
        data = parser.add_argument_group('Data')
        data.add_argument('--bs_train', type=int, default=256, 
                            help='Batch size for the training set.')
        data.add_argument('--bs_eval', type=int, default=256, 
                            help='Batch size for valid/test set.')

        # Model.
        model = parser.add_argument_group('Model')
        model.add_argument('--enc', type=str, default='ResNet18', 
                            help='Defines the model to train on.')
        model.add_argument('--mlp_act', type=str, choices={'GELU', 'ReLu'}, default='GELU',
                            help='Ativation function of DINOHead MLP.')
        model.add_argument('--use_bn', type=bool, action='store_true',
                            help='Use batchnorm in DINOHead MLP.')
        model.add_argument('--hid_dims', type=int, default=[2048, 2048], nargs='+',
                            help='Hidden dimensions of DINOHead MLP.')
        model.add_argument('--bot_dim', type=int, default=256,
                            help='L2-Bottleneck dimension of DINOHead MLP.')
        model.add_argument('--out_dim', type=int, default=256, 
                            help='Output dimension of the DINOHead MLP.')


        # Teacher Update, Temperature, Centering
        dino = parser.add_argument_group('DINO')
        dino.add_argument('--t_mode', type=str, choices={'ema', 'prev_epoch'}, default='ema',
                            help='Mode of teacher update.')
        dino.add_argument('--t_mom', type=Schedule.parse, default=CosSched(0.996, 1),
                            help='Teacher momentum for exponential moving average.')
        dino.add_argument('--t_cmom', type=Schedule.parse, default=ConstSched(0.9), 
                            help='Teacher centering momentum of DINOHead.')
        dino.add_argument('--t_temp', type=Schedule.parse, default=LinWarmup(0.04, 0.04, 0), 
                            help='Teacher temperature of DINOHead.')
        dino.add_argument('--s_temp', type=Schedule.parse, default=ConstSched(0.1), 
                            help='Teacher temperature of DINOHead.')


        
        # Training configurations.        
        training = parser.add_argument_group('Training')
        training.add_argument('--n_epochs', type=int, default=100, 
                            help='Number of epochs to train for.')
        training.add_argument('--opt', type=str, choices={'adam', 'sgd'}, default='adam', 
                            help='Optimizer to use for training.')                   
        training.add_argument('--clip_grad', type=float, default=3, 
                            help='Value to clip gradient norm to.')
        training.add_argument('--lr', type=Schedule.parse, default=CatSched(LinSched(0, 5e-4), CosSched(5e-4, 1e-6), 10), 
                            help='Learning rate for optimizer.')
        training.add_argument('--wd', type=Schedule.parse, default=CosSched(0.04, 0.4), 
                            help='Weight decay for optimizer.')
  

        # Probing configurations
        probing = parser.add_argument_group('Probing')
        probing.add_argument('--probe_every', type=int, default=5, 
                            help='Probe every so many epochs during training.')
        probing.add_argument('--probing_epochs', type=int, default=10, 
                            help='Number of epochs to train for linear probing.')

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
        default_config.update(json_config)

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

    def update(self, adict:dict):
        self.__dict__.update(adict)


def create_encoder(config:Configuration):
    '''
    This is a helper function that can be useful if you have several model definitions that you want to
    choose from via the command line.
    '''
    if config.enc == 'ResNet18':
        from encoders.ResNet import ResNet18
        return ResNet18(config)

    raise RuntimeError('Unkown model name.')


def create_optimizer(config:Configuration):
    '''
    This is a helper function that can be useful if you have optimizers that you want to
    choose from via the command line.
    '''
    if config.opt == 'adam':
        return torch.optim.Adam

    if config.opt == 'sgd':
        return torch.optim.SGD

    raise RuntimeError('Unkown optimizer name.')
