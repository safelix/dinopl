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
for network in ["vgg11"]:
    for init_loc in [
        "teacher",
        # "student_0_best",
        # "student_0_last",
        # "student_1_best",
        # "student_1_last",
    ]:
        for seed in range(50):
            i += 1

            call_args = [
                "python",
                "random_directions.py",
                "--network",
                network,
                "--init_loc",
                init_loc,
                "--seed",
                str(seed),
            ]
            pool = pools[i % len(available_gpus)]
            pool.apply_async(launch, args=(call_args, i % len(available_gpus)))

print("Total jobs:", i + 1)

for pool in pools:
    pool.close()
    pool.join()
