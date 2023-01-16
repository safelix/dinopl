import os
from itertools import product
import argparse
from multiprocessing.pool import ThreadPool
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpus", type=str, default="")

args = parser.parse_args()
available_gpus = args.gpus.split(",")

if len(available_gpus) <= 0:
    print("More gpus required!")
    exit()

pools = [ThreadPool(1) for _ in range(len(available_gpus))]


def launch(call_args, i):
    gpu_id = available_gpus[i]
    print(" ".join(["CUDA_VISIBLE_DEVICES={}".format(gpu_id)] + call_args))
    os.system(" ".join(["CUDA_VISIBLE_DEVICES={}".format(gpu_id)] + call_args))


print("Starting")

i = -1
for dataset_size in [
    7500,
    15000,
    20000,
]:
    epochs = int(100 * 50000 / dataset_size)
    check_val_every_n_epoch = max(int(1 * 50000 / dataset_size), 1)
    i += 1
    call_args = [
        "DINO_RESULTS=/local/home/sanagnos/unsupervised/dinopl/results",
        "DINO_DATA=/local/home/sanagnos/unsupervised/dinopl/data",
        "python",
        "dino.py",
        "--from_json",
        "configs/cifar10_distillation_v2.json",
        "--enc",
        "vgg11",
        "--dino_dataset_size",
        str(dataset_size),
        "--probe_dataset_size",
        "50000",
        "--n_epochs",
        str(epochs),
        "--probe_every_n_steps",
        str(100),
        "--check_val_every_n_epoch",
        str(check_val_every_n_epoch),
    ]
    pool = pools[i % len(available_gpus)]
    pool.apply_async(launch, args=(call_args, i % len(available_gpus)))

for pool in pools:
    pool.close()
    pool.join()
