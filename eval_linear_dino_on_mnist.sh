bsub -n 4 -W 15:00 -R "rusage[mem=4500]" -R "rusage[ngpus_excl_p=1]" -o bsub_std_out/eval_lineardino_resnet18_on_mnist python eval_linear.py \
--arch resnet18 \
--pretrained_weights /cluster/home/lciernik/deeplearning/output \
--use_mnist \
--data_path /cluster/home/lciernik/deeplearning/data \
--output_dir /cluster/home/lciernik/deeplearning/output2 \
--saveckp_freq 20 \
--num_workers 4 \
--num_labels 10 \
