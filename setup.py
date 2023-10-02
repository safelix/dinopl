#!/usr/bin/env python

from setuptools import setup, find_packages

install_requires=['torch', 'torchvision', 'torchinfo', 'pytorch-lightning==2.0.6'],
extras_require=dict(tracking=['wandb', 'pandas', 'torchmetrics==1.0.3', 'faiss-gpu==1.7.2', 'scikit-learn', 'array_api_compat'],
                    notebook=['notebook', 'ipywidgets', 'matplotlib', 'seaborn', 'tueplots', 'scipy', 'pillow'])
extras_require=dict(extras_require, all= [dep for group in extras_require.values() for dep in group]) # add all extra

setup(name='dinopl',
      version='0.0.1',
      description='Pytorch Lightning implementation of DINO',
      author='Felix Sarnthein',
      packages=find_packages(),
      install_requires=install_requires,
      extras_require=extras_require
     )