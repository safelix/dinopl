import argparse
from typing import Any, Union, List
import re

class Schedule(object):
    def __init__(self):
        pass

    def __call__(slef):
        pass


    def step(self):
        pass

class Linear(Schedule):
    def __init__(self, start, end):
        self.start = start
        self.end = end

class Cosine(Schedule):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __repr__(self) -> str:
        return f'Cosine({self.start},{self.end})'


def SchedulerParser(argv:str) -> Cosine:
    m = re.match(r'(?P<ScheduleID>\w+)\((?P<args>.*)\)', argv)
    args = m.group('args')

    if m.group('ScheduleID') == 'Cosine':
        m = re.match(r'(?P<start>\d+[\.?][\d]*),[ ]*(?P<stop>\d+[\.?][\d]*)', args)
        return Cosine(m.group('start'), m.group('stop'))
        
    if m.group('ScheduleID') == 'Linear':
        m = re.match(r'(?P<start>\d+[\.?][\d]*),[ ]*(?P<stop>\d+[\.?][\d]*)', args)
        return Linear(m.group('start'), m.group('stop'))

    if m.group('ScheduleID') == 'Combine':
        m = re.match(r'(?P<first>\d+[\.?][\d]*),[ ]*(?P<second>\d+[\.?][\d]*)', args)
        return Linear(m.group('start'), m.group('stop'))

    raise argparse.ArgumentTypeError('Unkown ScheduleID!')


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

parser = argparse.ArgumentParser()


parser.add_argument('--float', type=float, default=0.7, nargs='+',
                    help='Float.')
parser.add_argument('--tuple', type=tuple_type, default=(1, 2),
                    help='Tuple.')
parser.add_argument('--sched', type=SchedulerParser, default=Cosine(1,1),
                    help='Scheduler.')

args = parser.parse_args()
print(args)
