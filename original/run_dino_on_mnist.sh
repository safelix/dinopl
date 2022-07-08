bsub -n 4 -W 15:00 -R "rusage[mem=4500]" -R "rusage[ngpus_excl_p=1]" -o bsub_std_out/dino_resnet18_on_mnist python main_dino.py \
--arch resnet18 \
--use_mnist \
--data_path /cluster/home/lciernik/deeplearning/data \
--output_dir /cluster/home/lciernik/deeplearning/output \
--saveckp_freq 40 \
--num_workers 4 \
