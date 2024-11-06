import time

t_import_start = time.time()
import torch
import os
import time
from torch.utils.tensorboard import SummaryWriter
from Cryoemdata.mrc_process import MyMrcData, raw_data_preprocess
from Cryoemdata.cryoemDataset import EMDataset_from_path
import yaml
from easydict import EasyDict
from torch.utils.data import random_split
from Cryo_IEF import get_transformers
from argparse import ArgumentParser
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
import Cryo_IEF.vits as vits
from Cryo_IEF.vits import  Classifier,Classifier_2linear
import sys
from tqdm import tqdm
import numpy as np
from safetensors.torch import load_file
import torch.nn.functional as F
from cryosparc.dataset import Dataset
import pickle
from collections import defaultdict
t_import = time.time() - t_import_start
print('import time:{}'.format(t_import))

def sort_labels_by_score(scores_score_list, label_list, uid):
    """
    Given a list of scores scores and a corresponding list of labels,
    return a dictionary where the keys are the unique labels and the values
    are lists of indices corresponding to the scores scores for that label,
    sorted in descending order by scores score.

    Parameters:
    scores_score_list (list): A list of scores scores.
    label_list (list): A list of labels corresponding to the scores scores.

    Returns:
    dict: A dictionary where the keys are the unique labels and the values
          are lists of indices corresponding to the scores scores for that label,
          sorted in descending order by scores score.
    """
    labels_set= list(set(label_list))
    if len(scores_score_list) != len(label_list):
        raise ValueError("scores score list and label list must have the same length.")

    # Create a defaultdict to store the indices for each label
    label_to_indices = defaultdict(dict)

    # Zip the scores scores and labels together and sort by scores score
    sorted_indices = sorted(range(len(scores_score_list)), key=lambda i: scores_score_list[i], reverse=True)

    # Add the sorted indices to the defaultdict
    for index in sorted_indices:
        label_to_indices[labels_set.index(label_list[index])][uid[index]] = scores_score_list[index]
    return dict(label_to_indices)



def features_inference_from_processed_data(model, valid_loader, accelerator,
                                           ):
    '''Get the particle features from the Cryo-IEF'''
    features_all = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(valid_loader, desc='evaluating', disable=not accelerator.is_local_main_process):
            x = data['aug1']
            features ,y_p=model(x)
            features = accelerator.gather_for_metrics(features)
            features_np = F.normalize(features, p=2, dim=1).cpu().numpy().astype(np.float16)
            features_list=[features_np[i] for i in range(len(features_np))]
            features_all.extend(features_list)

    return {'features_all':features_all}


def merge_and_sort_lists(list1, list2, max_num):
    # 合并两个列表
    merged_list = list1 + list2

    # 根据scores值从大到小排序
    sorted_list = sorted(merged_list, key=lambda x: x[1], reverse=True)

    # 如果排序后的列表长度小于max_num，返回全部；否则返回前max_num个元素
    return sorted_list[:max_num]

def model_inference(cfg, accelerator):
    '''model'''
    # model = vits.__dict__["vit_small"](num_classes=cfg['model']['num_classes'], dynamic_img_size=True)
    # model.patch_embed.proj = nn.Conv2d(1, 384, kernel_size=(16, 16), stride=(16, 16))
    model = vits.__dict__[cfg['model']['backbone_name']](num_classes=cfg['model']['num_classes'], dynamic_img_size=True,
                                                         stop_grad_conv1=cfg['model']['stop_grad_conv1'],
                                                         use_bn=cfg['model']['use_bn'],
                                                         patch_size=cfg['model']['patch_size'])

    in_channels = model.head.in_features

    if cfg['model']['classifier']=='2linear':
        model.head=Classifier_2linear(input_dim=in_channels, output_dim=cfg['model']['num_classes'])
    else:
        model.head = Classifier(input_dim=in_channels, output_dim=cfg['model']['num_classes'])

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    state_dict = load_file(cfg['path_model_proj'] + '/model.safetensors', device='cpu')
    my_state_dict = {}
    linear_keyword = 'head'
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('base_encoder') and not k.startswith('base_encoder.%s' % linear_keyword):
            # remove prefix
            my_state_dict[k[len("base_encoder."):]] = state_dict[k]
        if k.startswith('base_encoder.head.'):
            my_state_dict['projector' + k[len('base_encoder.head'):]] = state_dict[k]

    if len(my_state_dict)==0:
        msg=model.load_state_dict(state_dict, strict=False)
    else:
        msg=model.load_state_dict(my_state_dict, strict=False)

    '''prepare the dataset'''
    batch_size = cfg['batch_size']
    nw = min(os.cpu_count(), 8)  # number of workers
    accelerator.print('Using {} dataloader workers'.format(nw))

    transforms_list_val = get_transformers.get_val_transformations(cfg['augmentation_kwargs'],
                                                                   mean_std=(0.53786290141223, 0.11803331075935841))


    mrcdata_val = MyMrcData(mrc_path=cfg['raw_data_path'], emfile_path=None, tmp_data_save_path=cfg['path_result_dir'],
                            processed_data_path=cfg['processed_data_path'],
                            selected_emfile_path=None, cfg=cfg['preprocess_kwargs'], is_extra_valset=True,
                            accelerator=accelerator)

    valset = EMDataset_from_path(mrcdata=mrcdata_val,
                                 is_Normalize=cfg['augmentation_kwargs']['is_Normalize'], )
    valset.get_transforms(transforms_list_val)
    valset.local_crops_number = 0

    if len(cfg['valset_name']) > 0:
        _, valset_index, _, _, _ = mrcdata_val.preprocess_trainset_valset_index(cfg['valset_name'],
                                                                                is_balance=True,
                                                                                max_resample_num_val=cfg[
                                                                                    'max_resample_number'])
        valset = torch.utils.data.Subset(valset, valset_index)

    val_dataloader = torch.utils.data.DataLoader(valset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=nw,
                                                 pin_memory=True,
                                                 persistent_workers=True,
                                                 )
    accelerator.print('dataset len:{}'.format(len(valset)))

    model, val_dataloader = accelerator.prepare(model, val_dataloader)
    '''reference'''
    results = features_inference_from_processed_data(model, val_dataloader, accelerator,

                                                     )

    if accelerator.is_local_main_process:
        from Other_tools.select_particles import divide_selected_particles_id, get_particles_from_cs
        from Cryoemdata.cs_star_translate.cs2star import cs2star

        with open(os.path.join(cfg['path_result_dir'], 'features_all.data'), 'wb') as filehandle:
            pickle.dump(results['features_all'], filehandle)

        accelerator.print('features are saved in {}'.format(os.path.join(cfg['path_result_dir'], 'features_all.data')))



def cryo_features_main(cfg=None,job_path=None,cache_file_path=None,accelerator=None,features_max_num=1000000):

    '''distribution'''
    if accelerator is None:
        accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=96000))])
    # accelerator.print(cfg)


    accelerator.print('start inference')
    if cfg is None:
        cfg = EasyDict()

        with open(
                 os.path.dirname(os.path.abspath(__file__))+'/CryoRanker/classification_inference_settings.yml',
                'r') as stream:
            config = yaml.safe_load(stream)


        for k, v in config.items():
            cfg[k] = v

        if job_path is not None:
            cfg['raw_data_path']=job_path
        if cache_file_path is not None:
            cfg['path_result_dir']=cache_file_path


    if accelerator.is_local_main_process:
        if not os.path.exists(cfg['path_result_dir']):
            os.makedirs(cfg['path_result_dir'])
            # with open(cfg['path_result_dir'] + '/settings.txt', 'w', encoding='utf-8') as f:
            #     f.write('{')
            #     for key in config:
            #         f.write('\n')
            #         f.writelines('"' + str(key) + '": ' + str(config[key]))
            #     f.write('\n' + '}')

    ''' Set the running config of this experiment '''

    accelerator.print('Time of run: ' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    '''inference'''
    features_all=[]
    processed_data_path = os.path.join(cfg['path_result_dir'], 'processed_data')
    new_cs_data_path = processed_data_path + '/new_particles.cs'

    if cfg['processed_data_path'] is not None or (os.path.exists(new_cs_data_path) and os.path.exists(processed_data_path + '/output_tif_path.data')):
        # processed_data_path = cfg['processed_data_path']
        new_cs_data = Dataset.load(new_cs_data_path)
        if cfg['processed_data_path'] is None:
            cfg['processed_data_path'] = processed_data_path

    elif cfg['raw_data_path'] is not None:

        if accelerator.is_local_main_process:

            time_data_process_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            accelerator.print('Time of data process start: ' + time_data_process_start)
            _ = raw_data_preprocess(cfg['raw_data_path'], processed_data_path,
                                                           # csfile_path=cfg['particle_csfile_path'],
                                                           resize=cfg['preprocess_kwargs']['resize'],
                                                           is_to_int8=True)

            accelerator.print('Time of data process finish: ' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))


        accelerator.wait_for_everyone()
        cfg['processed_data_path'] = processed_data_path
        new_cs_data = Dataset.load(new_cs_data_path)
        cfg['is_calculate_acc'] = False

    else:
        print('We need processed_data_path or raw_data_path to generate the dataset!')
        sys.exit(0)
    accelerator.print('Time of start inference: '+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) )
    model_inference(cfg, accelerator)
    accelerator.print('Time of finish inference: '+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) )



if __name__ == '__main__':
    '''get config'''
    parser = ArgumentParser()
    parser.add_argument('--path_result_dir', default=None, type=str)
    parser.add_argument('--raw_data_path', default=None, type=str)
    parser.add_argument('--path_model_proj', default=None, type=str)

    # parser.add_argument('--path_result_dir',
    #                     default='/yanyang2/projects/results/cryo_select_inference/2024_0808_my_test16/', type=str)
    # parser.add_argument('--raw_data_path',
    #                     default='/yanyang2/dataset/classification/valset_real/galactosidase/raw_cs/J110/', type=str)
    # parser.add_argument('--path_model_proj', default='/storage/yanyang2/projects/model_weights/CryoIEF_old', type=str)

    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--model_name', default=None, type=str)


    args = parser.parse_args()

    cfg = EasyDict()
    # if args.path_proj_dir is None:
    args.path_proj_dir = os.path.dirname(os.path.abspath(__file__))

    with open(
            args.path_proj_dir + '/CryoRanker/classification_inference_settings.yml',
            'r') as stream:
        config = yaml.safe_load(stream)
    # else:



    for k, v in config.items():
        cfg[k] = v

    if args.path_result_dir is not None:
        cfg['path_result_dir'] = args.path_result_dir

    if args.raw_data_path is not None:
        if args.raw_data_path.lower() == 'none' or args.raw_data_path.lower() == 'null':
            cfg['raw_data_path'] = None
        else:
            cfg['raw_data_path'] = args.raw_data_path


    if args.path_model_proj is not None:
        cfg['path_model_proj'] = args.path_model_proj

    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    '''Main function'''
    cryo_features_main(cfg=cfg,job_path=cfg['raw_data_path'],cache_file_path=cfg['path_result_dir'],features_max_num=cfg['max_resample_number'])





