import argparse
import ast
from math import log, ceil
from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch
from typing_extensions import Self

__all__ = [
    'Schedule',
    'Scheduler',
    'ConstSched',
    'LinSched',
    'CosSched',
    'ExpSched',
    'CatSched',
    'LinWarmup',
    'ExpWarmup',
    'CosWarmup',
    'MultiStep',
]


###############################################################################
###################### Schedule and Scheduler Definitions #####################
###############################################################################

## Strings which are mapped to constants in the parser
__named_const__ = {'nan':torch.nan,
                    'NaN':torch.nan, 
                    'none':torch.nan, 
                    'None':torch.nan}

class Schedule():
    '''A template class for parameter schedules, which allows to implement
    new schedules easily. It features a simple parser to specify the schedule
    of a parameter directly in the commandline. The schedule is an abstract
    specification which materializes only after .prep() is called. 
    Concatenation of schedules is supported and everything can be plotted easily:
    ```
        warmup = CosSched(0.6, 0.8)
        sched = CatSched(warmup, 0.8, 10).prep(n_steps, n_epochs, steps_per_epoch)
        plt.plot(sched.xs(0, n_epochs), sched.ys)
    ```
    '''
    def __init__(self):      
        self.n_steps:int = None
        self.n_epochs:int = None
        self.steps_per_epoch:int = None
        self.ys:torch.Tensor = None

    def xs(self, lower:float=0, upper:float=1) -> torch.Tensor:
        'Map all steps to the range [lower, upper].'
        if self.n_steps is None:
            raise RuntimeError('Schedule needs to be prepared first: call .prep()')
        return torch.linspace(lower, upper, self.n_steps)
    
    def set_ys(self) -> torch.Tensor:
        'Compute and set ys, after that xs() is defined.'
        self.ys = torch.full((self.n_steps, ), torch.nan)
        raise NotImplementedError('Please compute ys here.')

    def prep(self, n_steps:int, n_epochs:int, steps_per_epoch:int) -> Self:
        'Materialize schedule with n_steps and n_epochs'
        if n_epochs < 0 and n_steps < 0:
            raise ValueError('Schedule can only be computed if one of n_epoch and n_steps is given.')
        elif n_epochs < 0 or n_epochs * steps_per_epoch < n_steps:
            n_epochs = ceil(n_steps / steps_per_epoch)
        elif n_steps < 0 or n_steps < n_epochs * steps_per_epoch:
            n_steps = n_epochs * steps_per_epoch

        self.n_steps = n_steps 
        self.n_epochs = n_epochs
        self.steps_per_epoch =  steps_per_epoch 
        self.set_ys()
        return self

    def unprep(self):
        'De-materialize schedule.'
        self.n_steps = None
        self.n_epochs = None
        self.steps_per_epoch = None
        self.ys = None

    def __call__(self, it:int, epoch_offset:int=0):
        'If offset = 0, indexes absolute iteration'
        if self.ys is None:
            raise RuntimeError('Schedule needs to be prepared first: call .prep()')
        return self.ys[it + epoch_offset * self.steps_per_epoch]

    @staticmethod
    def parse_const(expr: ast.Expression):
        if isinstance(expr, str): # just in case someone calls this with a string
            expr = ast.parse(expr, mode='eval')
            return Schedule.parse_const(expr)

        if isinstance(expr, ast.Constant): # pythonic constants 
            literal = ast.literal_eval(expr)    # resolve literal
            if literal is None:                     # NoneType
                return torch.nan
            if isinstance(literal, (int, float)):   # numeric constants
                return literal
            if isinstance(literal, str) and literal in __named_const__.keys():
                return __named_const__[literal]     # string as named constant 
            raise RuntimeError(f'Unknown constant \'{literal}\', needs to be numeric or one of {__named_const__}.')
        
        if isinstance(expr, ast.Name):     # pythonic defined names
            if expr.id in __named_const__.keys():
                    return __named_const__[expr.id]
            raise RuntimeError(f'Unknown name \'{expr.id}\', needs one of {__named_const__}.')

    @staticmethod
    def parse(expr:Union[str, ast.AST]):
        'Parse string or AST expression into a schedule.'
        # recursion start case for expression
        if isinstance(expr, str):
            expr = ast.parse(expr, mode='eval')
            if isinstance(expr.body, (ast.Constant, ast.Name)): # (named) constants
                return ConstSched(Schedule.parse_const(expr.body))
            if isinstance(expr.body, ast.Call):                      # call AST expression
                return Schedule.parse(expr.body)
            raise RuntimeError(f'Unkown expression {expr}.')

        # recursion base case for arguments
        if isinstance(expr, (ast.Constant, ast.Name)): # (named) constants
            return Schedule.parse_const(expr)
        
        # recursion case for schedule types
        if isinstance(expr, ast.Call): 
            args = list(map(Schedule.parse, expr.args))     # recursively parse args
            for sclass in Schedule.__subclasses__():        # find Schedule type from id
                if sclass.__name__ == expr.func.id:
                    return sclass(*args)
            raise RuntimeError(f'Unkown schedule \'{expr.func.id}\' with args {args}.')
        

        # fall through case: argument was not a parseable string but a float
        if isinstance(expr, float):
            return ConstSched(expr)

        # fall through case: argument was not a parseable string but an int
        if isinstance(expr, int):
            return ConstSched(expr)

        # fall through case: argument was not a parseable string but a Schedule
        if isinstance(expr, Schedule):
            return expr

        # fall through case: don't know what happened
        raise RuntimeError(f'Unkown expression {expr} of type {type(expr).__name__}.')

    def __repr__(self, args=[]) -> str:
        out = f'{self.__class__.__name__}('
        for i, arg in enumerate(args):
            sep = ', ' if i<len(args)-1 else ''
            out = f'{out}{arg}{sep}'
        return f'{out})'


class Scheduler(pl.Callback):
    '''A lightweight scheduler callback. It maintains a list of all
    scheduled parameters. Parameters are accessed by reference 
    using a loc dictionary and a corresponding key.'''
    def __init__(self) -> None:
        super().__init__()
        self.scheduled_params:List[Tuple[dict, str, Schedule]] = []

    def add(self, loc:Union[dict, object], key:str, sched:Schedule):
        if not isinstance(loc, dict):
            loc = loc.__dict__
        self.scheduled_params.append((loc, key, sched))

    def get(self, loc:Union[dict, object], key:str):
        if not isinstance(loc, dict):
            loc = loc.__dict__
        for curr_loc, curr_key, curr_sched in self.scheduled_params: # get first schedule
            if curr_loc is loc and curr_key == key: # check if dicts are same objects
                return curr_sched
        raise RuntimeError(f'Schedule {loc}[{key}] could not be retrieved.')
    
    def prep(self, n_steps:int, n_epochs:int, steps_per_epoch:int):
        for loc, key, sched in self.scheduled_params: # prepare all schedules
            sched.prep(n_steps, n_epochs, steps_per_epoch)
        return self
    
    def step(self, step:int):
        for loc, key, sched in self.scheduled_params: # update parameter
            loc[key] = sched(step)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
        trainer.fit_loop.setup_data() # load train dataloader -> get len(loader) == trainer.num_training_batches
        steps_per_epoch = len(trainer.train_dataloader) / trainer.accumulate_grad_batches
        if int(steps_per_epoch) != steps_per_epoch:
            raise ValueError('Currently, the batch accumulation factor must devide the number of batches.')
        self.prep(trainer.max_steps, trainer.max_epochs, int(steps_per_epoch))

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args):
        self.step(trainer.global_step)



###############################################################################
########################## Schedule Implementations ###########################
###############################################################################

class ConstSched(Schedule):
    def __init__(self, val):
        super().__init__()
        self.val = float(val)

    def set_ys(self) -> Self:
        self.ys = torch.full((self.n_steps,), self.val)

    def __repr__(self) -> str:
        return super().__repr__([self.val])


class LinSched(Schedule):
    def __init__(self, y_start, y_end):
        super().__init__()
        self.y_start = float(y_start)
        self.y_end = float(y_end)

    def set_ys(self) -> Self:
        self.ys = self.y_start + (self.y_end - self.y_start) * self.xs()
    
    def __repr__(self) -> str:
        return super().__repr__([self.y_start, self.y_end])


class CosSched(Schedule):
    def __init__(self, y_start, y_end):
        super().__init__()
        self.y_start = float(y_start)
        self.y_end = float(y_end)

    def set_ys(self) -> Self:
        cos = 0.5 + torch.cos(self.xs(-torch.pi,0)) / 2
        self.ys = self.y_start + (self.y_end - self.y_start) * cos

    def __repr__(self) -> str:
        return super().__repr__([self.y_start, self.y_end])


class ExpSched(Schedule):
    def __init__(self, y_start, y_end):
        super().__init__()
        self.y_start = float(y_start)
        self.y_end = float(y_end)

    def set_ys(self):
        self.ys = torch.exp(self.xs(log(self.y_start), log(self.y_end)))

    def __repr__(self) -> str:
        return super().__repr__([self.y_start, self.y_end])


class CatSched(Schedule):
    def __init__(self, sched_l:Schedule, sched_r:Schedule, where:Union[float, int]):
        super().__init__()
        self.sched_l = sched_l
        self.sched_r = sched_r
        self.where = where

        # convert number to ConstSched
        if not isinstance(sched_l, Schedule):
            self.sched_l = ConstSched(sched_l)
        if not isinstance(sched_r, Schedule):
            self.sched_r = ConstSched(sched_r)
            
    def set_ys(self) -> Self:
        if isinstance(self.where, float):
            frac = self.where # interprete as rounded fraction of epoch
            n_steps_l = round(frac * self.n_steps)
            n_steps_r = self.n_steps - n_steps_l
            n_epochs_l, n_epochs_r = -1, -1
        elif isinstance(self.where, int):
            n_epochs_l = self.where # interprete as epoch
            n_epochs_r = self.n_epochs - n_epochs_l
            n_steps_l, n_steps_r = -1, -1
        else:
            raise ValueError('Unkown type for \'where\'.')

        # prepare schedules of left and right
        ys_list = []
        if n_steps_l > 0 or n_epochs_l > 0: # don't prepare if length is 0
            self.sched_l.prep(n_steps_l, n_epochs_l, self.steps_per_epoch)
            ys_list.append(self.sched_l.ys)
        if n_steps_r > 0 or n_epochs_r > 0: # don't prepare if length is 0
            self.sched_r.prep(n_steps_r, n_epochs_r, self.steps_per_epoch)
            ys_list.append(self.sched_r.ys)

        # concatenate left and right schedules
        self.ys = torch.concat(ys_list)

        # de-materialize left and right schedules
        self.sched_r.unprep()
        self.sched_l.unprep()
        
    def __repr__(self) -> str:
        return super().__repr__([self.sched_l, self.sched_r, self.where])


class LinWarmup(CatSched, Schedule):
    def __init__(self, y_start:float, y_end:float, where:int):
        super().__init__(LinSched(y_start, y_end), y_end, where)
        self.y_start = float(y_start)
        self.y_end = float(y_end)
        self.where = where
    
    def __repr__(self) -> str:
        return Schedule.__repr__(self, [self.y_start, self.y_end, self.where])


class ExpWarmup(CatSched, Schedule):
    def __init__(self, y_start:float, y_end:float, where:int):
        super().__init__(ExpSched(y_start, y_end), y_end, where)
        self.y_start = float(y_start)
        self.y_end = float(y_end)
        self.where = where
    
    def __repr__(self) -> str:
        return Schedule.__repr__(self, [self.y_start, self.y_end, self.where])

class CosWarmup(CatSched, Schedule):
    def __init__(self, y_start:float, y_end:float, where:int):
        super().__init__(CosSched(y_start, y_end), y_end, where)
        self.y_start = float(y_start)
        self.y_end = float(y_end)
        self.where = where
    
    def __repr__(self) -> str:
        return Schedule.__repr__(self, [self.y_start, self.y_end, self.where])

class MultiStep(Schedule):
    def __init__(self, start:float, gamma:float, *steps:List[Union[int, float]]):
        self.start = float(start)
        self.gamma = float(gamma)
        self.steps = steps # ratio of steps if float, epochs if int

    def set_ys(self) -> Self:
        last_step = 0
        value = self.start
        self.ys = torch.full((self.n_steps, ), torch.nan)

        for step in self.steps:

            if isinstance(step, int):
                step = step * self.steps_per_epoch
            if isinstance(step, float):
                step = int(step * self.n_steps)

            self.ys[last_step:step] = value # fill ys
            value *= self.gamma # update value
            last_step = step # update last step

        self.ys[last_step:] = value # fill remaining

    def __repr__(self) -> str:
        return Schedule.__repr__(self, [self.start, self.gamma] + list(self.steps))


# run with: python -m dinopl.scheduling
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('schedule', type=Schedule.parse,
                        help='Schedule to parse, prep and print.')
    parser.add_argument('--n_epochs', type=int, default=-1,
                        help='Number of epochs to prepare the Schedule for.')
    parser.add_argument('--n_steps', type=int, default=-1,
                        help='Number of steps to prepare the Schedule for.')
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                        help='Number of steps per epoch.')

    args = parser.parse_args()
    sched:Schedule = args.schedule
    print(f'Parsed: {sched}')

    if args.steps_per_epoch and (args.n_steps >= 0 or args.n_epochs >= 0):
        sched.prep(args.n_steps, args.n_epochs, args.steps_per_epoch)
        print(f'Prepared Schedule: {sched.ys}')
    elif args.steps_per_epoch and args.n_steps < 0 and args.n_epochs < 0:
        raise RuntimeError('Please specify n_epochs or n_steps.')