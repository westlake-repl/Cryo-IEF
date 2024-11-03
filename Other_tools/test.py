import torch
# path='/yanyang2/projects/results/cryoem_pretrain/2024_0419_000755_moco3_vit_b14_bn_newdata/checkpoint/'
# path='/yanyang2/projects/results/cryoem_pretrain/2024_0421_193749_moco3_vit_b14_bn_newdata_3072//checkpoint/'
path='/yanyang2/projects/results/cryoem_pretrain/2024_0519_150128_moco3_vit_b16_3072_25e6_conv_f_class_shuffle/checkpoint/'
# path='/yanyang2/projects/results/cryoem_pretrain/2024_0520_145845_moco3_vit_b16_3072_2e5_conv_15r_class_shuffle/checkpoint/'
torch.save({'epoch': 6}, path + '/start_epoch.pt')