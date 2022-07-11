#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='dinopl',
      version='0.0.1',
      description='Pytorch Lightning implementation of DINO',
      author='Felix Sarnthein',
      packages=find_packages(),
      install_requires=['torch', 'pytorch-lightning', 'torchmetrics']
     )