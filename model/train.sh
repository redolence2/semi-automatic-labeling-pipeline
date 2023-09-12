#!/bin/bash

#stage 1
# python train.py -p bdd100k -c 3  -n 2 -b 16  --lr 1e-3  --optim adamw  --num_epochs 200 --warm_lr 1e-6 --poly_end_lr 1e-6 --poly_decay_power 0.9   --warmup_epoch 8 --freeze_instance True --lr_scheduler poly
python train.py -p bdd100k -c 3  -n 2 -b 16  --lr 5e-4  --optim adamw  --num_epochs 200 --warm_lr 1e-6 --poly_end_lr 1e-6 --poly_decay_power 0.9   --warmup_epoch 8 --freeze_instance True --lr_scheduler poly

#stage 2
# python train.py -p bdd100k -c 3  -n 2 -b 20  --lr 1e-3  --optim adamw  --num_epochs 50 --warm_lr 1e-6 --poly_end_lr 1e-6 --poly_decay_power 0.9   --warmup_epoch 2 --freeze_seg True --freeze_backbone True --lr_scheduler poly -w /home/wan/ZT_2T/work/git_hybridNets/HybridNets/checkpoints/bdd100k/hybridnets-d3_199_823500__test_poly_.pth

#stage 3
# python train.py -p bdd100k -c 3  -n 2 -b 6  --lr 1e-5  --optim adamw  --num_epochs 50 --warm_lr 1e-7 --poly_end_lr 1e-7 --poly_decay_power 0.9   --warmup_epoch 4  --lr_scheduler poly -w /home/wan/ZT_2T/work/git_hybridNets/HybridNets/checkpoints/bdd100k/hybridnets-d3_49_164500__stage2_fix_dataset_inst_.pth
# python train.py -p bdd100k -c 3  -n 2 -b 16  --lr 2.5e-7  --optim adamw  --num_epochs 100 --warm_lr 1e-8 --poly_end_lr 1e-8 --poly_decay_power 0.9   --warmup_epoch 4  --lr_scheduler poly -w /home/wan/ZT_2T/work/git_hybridNets/HybridNets/checkpoints/bdd100k/hybridnets-d3_49_102500__stage2_frz_bb_and_seg_.pth
# python train.py -p bdd100k -c 3  -n 1 -b 4  --lr 2.5e-7  --optim adamw  --num_epochs 50 --warm_lr 1e-8 --poly_end_lr 1e-8 --poly_decay_power 0.9   --warmup_epoch 4  --lr_scheduler poly -w /home/wan/ZT_2T/work/git_hybridNets/HybridNets/checkpoints/bdd100k/hybridnets-d3_49_164500__stage2_fix_dataset_inst_.pth



# debug
# python train.py -p bdd100k -c 3  -n 1 -b 1  --lr 2.5e-7 --vis_input True --optim adamw  --num_epochs 50 --warm_lr 1e-8 --poly_end_lr 1e-8 --poly_decay_power 0.9   --warmup_epoch 2  --lr_scheduler poly -w /home/wan/ZT_2T/work/git_hybridNets/HybridNets/checkpoints/bdd100k/hybridnets-d3_49_102500__stage2_frz_bb_and_seg_.pth




