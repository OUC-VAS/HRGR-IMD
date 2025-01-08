#!/usr/bin/env bash


# pre-train
CONFIG=configs/manipulation_detection/hssn_soft_internimage_t_512_160k_manipulation_focal.py

# fine-tune CASIA
#CONFIG=configs/casia_val/hssn_soft_internimage_freeze_kernel_t_512_160k_manipulation_casia_val_focal.py

# fine-tune Coverage
#CONFIG=configs/coverage/hssn_soft_internimage_freeze_kernel_t_512_160k_manipulation_coverage_val_focal.py

# fine-tune NIST16
#CONFIG=configs/nist16/hssn_soft_internimage_freeze_kernel_t_512_160k_manipulation_nist16_val_focal_16k.py

# shellcheck disable=SC2068
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch train.py $CONFIG --launcher pytorch
