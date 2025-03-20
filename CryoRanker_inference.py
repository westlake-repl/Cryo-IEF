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
from Cryo_IEF.vits import  Classifier,Classifier_2linear,Classifier_new
import sys
from tqdm import tqdm
from CryoRanker.edl_loss import edl_digamma_loss, one_hot_embedding, relu_evidence
import numpy as np
from safetensors.torch import load_file
import pandas as pd
import torch.nn.functional as F
from cryosparc.dataset import Dataset
from sklearn.cluster import kmeans_plusplus, MiniBatchKMeans
import pickle
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
# from thop import profile
t_import = time.time() - t_import_start
print('import time:{}'.format(t_import))

def sampling_iter(labels_iter,centers_iter,num_class,sample_k,job=None):
    n = centers_iter.shape[0]
    new_num_classes=[]
    dis_list=[]

    if sample_k >= n:
        return np.arange(n)

    distances = euclidean_distances(centers_iter)

    selected_indices = [np.random.randint(n)]

    for _ in range(1, sample_k):

        min_distances = np.min(distances[selected_indices][:, np.setdiff1d(range(n), selected_indices)], axis=0)
        max_min_distance = np.max(min_distances)
        next_index = np.setdiff1d(range(n), selected_indices)[np.argmax(min_distances)]
        selected_indices.append(next_index)
        new_num_classes.append(num_class[next_index])
        dis_list.append(max_min_distance)
    print('distance list:'+str(dis_list))
    return selected_indices


def inference_processed_data(model, valid_loader, accelerator, is_calculate_acc=False, use_bnn_head=False,
                             use_edl_loss=False,use_features=False,features_max_num=50000):
    ''' Get the scores of the model on the dataset '''
    acc_correct_num = 0
    acc_num_all = 0
    recall_num_all = 0
    recall_correct_num = 0
    precision_num_all = 0
    precision_correct_num = 0

    recall_with_threshold = 0
    precision_with_threshold = 0
    acc_with_threshold = 0

    mean_uncertainty = 0
    mean_uncertainty_positive = 0
    mean_uncertainty_negative = 0

    total_loss, total_num = 0.0, 0

    class_p_all = []
    y_t_all = []
    features_all = []
    uncertainty_all = []
    scores_predicted_all = []
    scores_all = []
    # sorted_resampled_features=[]
    model.eval()
    with torch.no_grad():
        for data in tqdm(valid_loader, desc='evaluating', disable=not accelerator.is_local_main_process):
            # for data in valid_loader:
            x = data['aug1']
            y_t = accelerator.gather_for_metrics(data['label_for_classification'])
            y_t_all.extend(y_t.cpu().tolist())
            # Forward pass and record loss
            features ,y_p=model(x)
            # y_p = model.forward_head(features)
            y_p = accelerator.gather_for_metrics(y_p)

            features = accelerator.gather_for_metrics(features)

            if use_edl_loss:
                evidence = relu_evidence(y_p)
                alpha = evidence + 1
                uncertainty = 2 / torch.sum(alpha, dim=1, keepdim=True)
                uncertainty_all.extend(torch.squeeze(uncertainty).cpu().tolist())
            else:
                y_p_softmax = F.softmax(y_p, dim=-1)

                numerator_pos = y_p_softmax[:, 1]
                # denominator = y_p_softmax[:, 0] + y_p_softmax[:, 1]
                denominator = torch.sum(y_p_softmax, dim=1)
                scores_predicted = torch.div(numerator_pos, denominator).cpu().tolist()
                scores_predicted_all.extend(scores_predicted)
                numerator = torch.max(y_p_softmax, dim=1).values
                scores = torch.div(numerator, denominator)
                scores_all.extend(scores.cpu().tolist())
            if use_features:
                items = accelerator.gather_for_metrics(data['item']).cpu().numpy()
                features_np = F.normalize(features, p=2, dim=1).cpu().numpy().astype(np.float16)
                scores_tuple_list=[(items[i],scores_predicted[i],features_np[i]) for i in range(len(scores_predicted))]
                features_all=merge_and_sort_lists(features_all,scores_tuple_list,features_max_num)

                # features_all.extend(features_np)

            out_digit = y_p.argmax(axis=1)
            total_num += x.shape[0]
            acc_num_all += len(y_t)
            class_p_all.extend(out_digit.cpu().tolist())
            if is_calculate_acc:
                pass
                # loss_criterion = loss_function()
                # loss = loss_criterion(y_p, y_t)
                #
                # total_loss += loss.item() * x.shape[0]
                #
                # acc_correct_num += sum(y_t == out_digit).cpu().item()
                #
                # # m=(y_t == 1) & (out_digit == 1)
                # recall_correct_num += sum((y_t == 1) & (out_digit == 1)).cpu().item()
                # recall_num_all += sum(y_t == 1).cpu().item()
                #
                # precision_correct_num += sum((y_t == 1) & (out_digit == 1)).cpu().item()
                # precision_num_all += sum(out_digit == 1).cpu().item()

    new_features_all={}
    for id_scores_feature in features_all:
        new_features_all[id_scores_feature[0]]=id_scores_feature[2]

    if recall_num_all > 0:
        recall = recall_correct_num / recall_num_all
    else:
        recall = 0

    if precision_num_all > 0:
        precision = precision_correct_num / precision_num_all
    else:
        precision = 0

    return {'loss': total_loss / total_num, 'aug_img': x, 'img': data['mrcdata'], 'class_t': y_t_all,
            'acc': acc_correct_num / acc_num_all, 'recall': recall, 'precision': precision, 'class_p': class_p_all,
            'acc_with_threshold': acc_with_threshold,
            'recall_with_threshold': recall_with_threshold, 'precision_with_threshold': precision_with_threshold,
            'mean_uncertainty': mean_uncertainty,
            'mean_uncertainty_positive': mean_uncertainty_positive,
            'mean_uncertainty_negative': mean_uncertainty_negative,
            'positive ratio': sum(np.array(class_p_all) == 1) / len(y_t_all),
            'negative ratio': sum(np.array(class_p_all) == 0) / len(y_t_all), 'uncertainty_list': uncertainty_all,
            'scores_predicted_list': scores_predicted_all, 'scores_all_list': scores_all,'features_all':new_features_all}


def merge_and_sort_lists(list1, list2, max_num):
    # 合并两个列表
    merged_list = list1 + list2

    # 根据scores值从大到小排序
    sorted_list = sorted(merged_list, key=lambda x: x[1], reverse=True)

    # 如果排序后的列表长度小于max_num，返回全部；否则返回前max_num个元素
    return sorted_list[:max_num]

def model_inference(cfg, accelerator,use_features=False,features_max_num=50000):
    '''model'''
    # model = vits.__dict__["vit_small"](num_classes=cfg['model']['num_classes'], dynamic_img_size=True)
    # model.patch_embed.proj = nn.Conv2d(1, 384, kernel_size=(16, 16), stride=(16, 16))
    model = vits.__dict__[cfg['model']['backbone_name']](num_classes=cfg['model']['num_classes'], dynamic_img_size=True,
                                                         stop_grad_conv1=cfg['model']['stop_grad_conv1'],
                                                         use_bn=cfg['model']['use_bn'],
                                                         patch_size=cfg['model']['patch_size'])

    in_channels = model.head.in_features

    if cfg['model']['classifier']=='new':
        model.norm = torch.nn.Identity()
        model.head=Classifier_new(input_dim=in_channels, output_dim=cfg['model']['num_classes'])
    elif cfg['model']['classifier']=='2linear':
        model.head=Classifier_2linear(input_dim=in_channels, output_dim=cfg['model']['num_classes'])
    else:
        model.head = Classifier(input_dim=in_channels, output_dim=cfg['model']['num_classes'])

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    state_dict = load_file(cfg['path_model_proj'] + '/model.safetensors', device='cpu')
    model.load_state_dict(state_dict, strict=True)

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
    results = inference_processed_data(model, val_dataloader, accelerator, is_calculate_acc=cfg['is_calculate_acc'],
                                       use_edl_loss=cfg['model']['use_edl_loss'],
                                       use_bnn_head=cfg['model']['use_bnn_head'],use_features=use_features,features_max_num=features_max_num)

    if cfg['is_calculate_acc']:
        accelerator.print('valset acc:{}'.format(results['acc']))
        accelerator.print('valset recall:{}'.format(results['recall']))
        accelerator.print('valset precision:{}'.format(results['precision']))
        accelerator.print('valset loss:{}'.format(results['loss']))

    accelerator.print('valset positive ratio:{}'.format(results['positive ratio']))
    accelerator.print('valset negative ratio:{}'.format(results['negative ratio']))
    if accelerator.is_local_main_process:
        from Other_tools.select_particles import divide_selected_particles_id, get_particles_from_cs
        from Cryoemdata.cs_star_translate.cs2star import cs2star

        labels_predicted_pd = pd.DataFrame(data=results['class_p'], columns=['labels_predicted'])
        labels_predicted_pd.to_csv(os.path.join(cfg['path_result_dir'], 'labels_predicted.csv'), index=False)

        uncertainty_list_pd = pd.DataFrame(data=results['uncertainty_list'], columns=['uncertainty_list'])
        uncertainty_list_pd.to_csv(os.path.join(cfg['path_result_dir'], 'uncertainty_list.csv'), index=False)

        scores_list_pd = pd.DataFrame(data=results['scores_predicted_list'], columns=['scores_predicted_list'])
        scores_list_pd.to_csv(os.path.join(cfg['path_result_dir'], 'scores_predicted_list.csv'), index=False)

        scores_list_pd = pd.DataFrame(data=results['scores_all_list'], columns=['scores_all_list'])
        scores_list_pd.to_csv(os.path.join(cfg['path_result_dir'], 'scores_all_list.csv'), index=False)


        with open(os.path.join(cfg['path_result_dir'], 'features_all.data'), 'wb') as filehandle:
            pickle.dump(results['features_all'], filehandle)

        if cfg['is_calculate_acc']:
            labels_true_pd = pd.DataFrame(data=results['class_t'], columns=['labels_true'])
            labels_true_pd.to_csv(os.path.join(cfg['path_result_dir'], 'labels_true.csv'), index=False)


    return results['scores_predicted_list'],results['features_all']




def cryo_select_main(cfg=None,job_path=None,cache_file_path=None,accelerator=None,use_features=False,features_max_num=1000000):
    '''distribution'''
    if accelerator is None:
        accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=96000))])
    accelerator.print(cfg)


    accelerator.print('start inference')
    if cfg is None:
        cfg = EasyDict()

        with open(
                 os.path.dirname(os.path.abspath(__file__))+'/CryoRanker/cryoranker_inference_settings.yml',
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
    if not (os.path.exists(cfg['path_result_dir'] + '/labels_predicted.csv') and os.path.exists(
            cfg['path_result_dir'] + '/uncertainty_list.csv')):
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
        scores,features_all=model_inference(cfg, accelerator,use_features=use_features,features_max_num=features_max_num)
        accelerator.print('Time of finish inference: '+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) )
    else:
        accelerator.print('labels_predicted.csv and uncertainty_list.csv already exist!')

        new_cs_data = Dataset.load(cfg['path_result_dir'] +'/processed_data/new_particles.cs')
        scores= pd.read_csv(cfg['path_result_dir'] + '/scores_predicted_list.csv')[
                'scores_predicted_list'].values

        if os.path.exists(cfg['path_result_dir'] + '/features_all.data') and use_features:
            with open(cfg['path_result_dir'] + '/features_all.data',
                      'rb') as filehandle:
                features_all = pickle.load(filehandle)


    return new_cs_data,scores,features_all


if __name__ == '__main__':
    '''get config'''
    parser = ArgumentParser()

    parser.add_argument('--path_result_dir', default=None, type=str)
    parser.add_argument('--raw_data_path', default=None, type=str)
    parser.add_argument('--path_model_proj', default=None, type=str)
    parser.add_argument('--path_proj_dir', default=None, type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--model_name', default=None, type=str)


    args = parser.parse_args()

    cfg = EasyDict()
    if args.path_proj_dir is None:
        args.path_proj_dir = os.path.dirname(os.path.abspath(__file__))

    with open(
            args.path_proj_dir + '/CryoRanker/cryoranker_inference_settings.yml',
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

    new_cs_data,scores,features_all=cryo_select_main(cfg=cfg,job_path=cfg['raw_data_path'],cache_file_path=cfg['path_result_dir'],use_features=True)





