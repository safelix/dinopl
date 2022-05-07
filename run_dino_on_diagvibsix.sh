bsub -n 4 -W 4:00 -R "rusage[mem=4500]" -R "rusage[ngpus_excl_p=1]" -o bsub_std_out/dino_resnet18_on_diagvibsix python main_dino.py \
--arch resnet18 \
--use_diagvibsix \
--data_path /cluster/home/safelix/data/dino_on_diagvibsix/data/train \
--output_dir /cluster/home/safelix/data/dino_on_diagvibsix/results \
--saveckp_freq 40 \
--num_workers 4 \
--epochs 50