#!/usr/bin/env sh
python train.py --batch_size 16 --domain_idxs 1,2,3 --save_path ../out/fundus/baseline/unet/target0/0 --gpu 0 --model unet --test_domain_idx 0
python train.py --batch_size 16 --domain_idxs 1,2,3 --save_path ../out/fundus/baseline/unet/target0/1 --gpu 0 --model unet --test_domain_idx 0
python train.py --batch_size 16 --domain_idxs 1,2,3 --save_path ../out/fundus/baseline/unet/target0/2 --gpu 0 --model unet --test_domain_idx 0

