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
from CryoRanker.edl_loss import edl_digamma_loss, one_hot_embedding, relu_evidence
from CryoRanker.classification_finetune_train import loss_function
import numpy as np
from safetensors.torch import load_file
import pandas as pd
import torch.nn.functional as F
from cryosparc.dataset import Dataset
from sklearn.cluster import kmeans_plusplus, MiniBatchKMeans
from sklearn.metrics.pairwise import _euclidean_distances
import pickle
from collections import defaultdict
import math
from Cryoemdata.cs_star_translate.cs2star import cs2star
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# from thop import profile
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

def features_n_kmeans(mrcArray_features, k, cut_ratio, job=None, merge_threshold=0.0, centers_sampling=True, n_runs=10):
    '''Clustering the features of particles by kmeans++'''

    num_features = mrcArray_features.shape[0]

    if job is not None:
        job.log('particles num: ' + str(num_features))
    else:
        print('particles num: ' + str(num_features))

    scaler = StandardScaler()
    mrcArray_features_scaled = scaler.fit_transform(mrcArray_features)
    # mrcArray_features_scaled = mrcArray_features

    best_inertia = float('inf')
    best_labels = None
    best_centers = None

    for _ in range(n_runs):
        clf = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=3)
        clf.fit(mrcArray_features_scaled)
        if clf.inertia_ < best_inertia:
            best_inertia = clf.inertia_
            best_labels = clf.labels_
            best_centers = clf.cluster_centers_

    labels_iter = best_labels
    centers_iter = best_centers

    # 初始聚类结果
    unique_labels = np.unique(labels_iter)
    num_class = [np.sum(labels_iter == l) for l in unique_labels]

    if job is not None:
        job.log(f"Initial clusters: {num_class}")
    else:
        print(f"Initial clusters: {num_class}")

    # if merge_threshold > 0:
    #
    #     while max(num_class) < num_features * cut_ratio:
    #         distances = cdist(centers_iter, centers_iter, metric='cosine')  # 使用余弦距离
    #         np.fill_diagonal(distances, np.inf)
    #         closest_pair = np.unravel_index(distances.argmin(), distances.shape)
    #
    #         if distances[closest_pair] > merge_threshold:
    #             break
    #
    #         merge_from, merge_to = closest_pair
    #         labels_iter[labels_iter == unique_labels[merge_from]] = unique_labels[merge_to]
    #
    #         unique_labels = np.unique(labels_iter)
    #
    #         centers_iter = np.array([mrcArray_features_scaled[labels_iter == i].mean(axis=0) for i in unique_labels])
    #         num_class = [np.sum(labels_iter == l) for l in unique_labels]
    #
    #     label_map = {old: new for new, old in enumerate(unique_labels)}
    #     labels_iter = np.array([label_map[l] for l in labels_iter])
    #
    #     unique_labels = np.unique(labels_iter)
    #     centers_iter = np.array([mrcArray_features_scaled[labels_iter == i].mean(axis=0) for i in unique_labels])
    #     num_class = [np.sum(labels_iter == l) for l in unique_labels]
    #
    #     if centers_sampling:
    #         selected_indices=sampling_iter(labels_iter,centers_iter,num_class,sample_k=4,job=job)
    #         print('indexes with largest distance:'+str(selected_indices))
    #
    #     if job is not None:
    #         job.log(f"Final clusters: {num_class}")
    #     else:
    #         print(f"Final clusters: {num_class}")

    return labels_iter, centers_iter, np.array(num_class)

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
# def clustering_iter(labels_iter,centers_iter,num_features,num_class,k,iter_num=0,job=None):
#     num_class_centers = []
#     num_class_particles = []
#     new_k = int(math.sqrt(k))
#     expect_class_num_c =  new_k
#     particles_labels_set=list(set(labels_iter))
#     expect_particles_num = num_features / new_k
#     clf_c = MiniBatchKMeans(n_clusters=new_k, n_init='auto')
#     clf_c.fit(centers_iter)
#     centers_centers = clf_c.cluster_centers_
#     centers_labels = clf_c.labels_
#     centers_labels_set = list(set(centers_labels))
#     particles_num_for_classes_np = np.array(num_class)
#
#     class_to_be_combined = []
#     for l in centers_labels_set:
#         particles_n = np.sum(particles_num_for_classes_np[centers_labels == l])
#         class_n = np.sum(centers_labels == l)
#         num_class_centers.append(class_n)
#         num_class_particles.append(particles_n)
#         if class_n > 0.35 * k or particles_n > 0.35 * num_features:
#             class_to_be_combined.append(l)
#
#
#
#     if len(class_to_be_combined) == 0:
#         i_to_combined = {num_class_centers.index(max(num_class_centers))}
#         i_to_combined.add(num_class_particles.index(max(num_class_particles)))
#         for i_c in i_to_combined:
#             class_to_be_combined.append(i_c)
#     flag=0
#     labels_change_dict={}
#     new_centers = []
#     new_particles_num_of_diff_classes=[]
#     for i in particles_labels_set:
#         if not i in labels_change_dict.keys():
#             if not centers_labels[i] in class_to_be_combined:
#                 labels_change_dict[i]=flag
#                 new_centers.append(centers_iter[i])
#                 new_particles_num_of_diff_classes.append(num_class[i])
#             else:
#
#                 particles_labels_combined_set=[index for index, value in enumerate(centers_labels) if value == centers_labels[i]]
#                 for ii in particles_labels_combined_set:
#                     if not ii in labels_change_dict.keys():
#                         labels_change_dict[ii]=flag
#                 new_centers.append(centers_centers[centers_labels[i]])
#                 new_particles_num_of_diff_classes.append(np.sum(np.array(num_class)[centers_labels==centers_labels[i]]))
#             flag+=1
#
#
#     for i in range(labels_iter.shape[0]):
#         labels_iter[i]=labels_change_dict[labels_iter[i]]
#
#
#
#     if job is not None:
#         job.log('iter: '+str(iter_num))
#         job.log(str(num_class_centers))
#         job.log(str(new_particles_num_of_diff_classes))
#         job.log(str(num_class_particles))
#     else:
#         print('iter: '+str(iter_num))
#         print('number of class centers')
#         print(num_class_centers)
#         print('number of class particles')
#         print(num_class_particles)
#         print('number of particles after combination')
#         print(new_particles_num_of_diff_classes)
#     return labels_iter,new_centers,new_particles_num_of_diff_classes


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

    if cfg['model']['classifier']=='2linear':
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



# def results_score_cut(cfg, accelerator):
#     if accelerator.is_local_main_process:
#         from Other_tools.select_particles import divide_selected_particles_id, get_particles_from_cs
#         from Cryoemdata.cs_star_translate.cs2star import cs2star
#         tb_writer = SummaryWriter(log_dir=cfg['path_result_dir'] + "/tensorboard/")
#         labels_predicted = pd.read_csv(cfg['path_result_dir'] + '/labels_predicted.csv')['labels_predicted'].values
#
#         if cfg['is_calculate_acc']:
#             labels_true = pd.read_csv(cfg['path_result_dir'] + '/labels_true.csv')['labels_true'].values
#             recall_correct_num_original = sum((labels_true == 1) & (labels_predicted == 1))
#             recall_num_all_original = sum(labels_true == 1)
#             precision_num_all_original = sum(labels_predicted == 1)
#             if recall_num_all_original > 0:
#                 recall_original = recall_correct_num_original / recall_num_all_original
#             else:
#                 recall_original = 0
#             if precision_num_all_original > 0:
#                 precision_original = recall_correct_num_original / precision_num_all_original
#             else:
#                 precision_original = 0
#             accelerator.print('original acc: ' + str(sum(labels_predicted == labels_true) / len(labels_predicted)))
#             accelerator.print('original recall: ' + str(recall_original))
#             accelerator.print('original precision: ' + str(precision_original))
#         else:
#             labels_true = None
#
#         accelerator.print('original selected particles number: ' + str(sum(labels_predicted == 1)))
#         if cfg['model']['use_bnn_head'] or cfg['model']['use_edl_loss']:
#             uncertainty_list = pd.read_csv(cfg['path_result_dir'] + '/uncertainty_list.csv')['uncertainty_list'].values
#             selected_particles_id_all, unselected_particles_id, selected_particles_uncertainty, unselected_particles_uncertainty = divide_selected_particles_id(
#                 labels_predicted, [1], uncertainty_list)
#         else:
#             from Cryoemdata.utils import plot_scores_interval_inf
#             scores_predicted_list = pd.read_csv(cfg['path_result_dir'] + '/scores_predicted_list.csv')[
#                 'scores_predicted_list'].values
#             scores_all_list = pd.read_csv(cfg['path_result_dir'] + '/scores_all_list.csv')[
#                 'scores_all_list'].values
#             fig_num, fig_acc = plot_scores_interval_inf(scores_all_list, labels_predicted, labels_true,
#                                                             cfg['path_result_dir'])
#             tb_writer.add_figure('Evaluation figures/Particles num of different scores interval', fig_num)
#             if fig_acc is not None:
#                 tb_writer.add_figure('Evaluation figures/Classification acc of different scores interval', fig_acc)
#                 tb_writer.add_pr_curve('evaluation data/precision recall curve', np.array(labels_true),
#                                        np.array(scores_predicted_list))
#
#         select_ratio_step = (cfg['scores_cut']['ratio'][1] - cfg['scores_cut']['ratio'][0]) / \
#                             cfg['scores_cut']['step_num']
#         for i in range(cfg['scores_cut']['step_num'] + 1):
#             select_ratio = cfg['scores_cut']['ratio'][0] + i * select_ratio_step
#             final_selected_num = int(len(labels_predicted) * select_ratio)
#             accelerator.print()
#             accelerator.print('Ratio: ' + str(select_ratio))
#
#             new_selected_particles_id = score_cut_step(len(labels_predicted), scores_predicted_list, accelerator,
#                                                        final_selected_num)
#
#             accelerator.print('With select ratio ' + str(select_ratio) + ', final selected particles number: ' + str(
#                 len(new_selected_particles_id)))
#
#             if cfg['is_calculate_acc']:
#                 new_labels_predicted = np.zeros(len(labels_predicted))
#                 new_labels_predicted[new_selected_particles_id] = 1
#                 recall_correct_num = sum((labels_true == 1) & (new_labels_predicted == 1))
#                 recall_num_all = sum(labels_true == 1)
#                 precision_num_all = sum(new_labels_predicted == 1)
#                 if recall_num_all > 0:
#                     recall = recall_correct_num / recall_num_all
#                 else:
#                     recall = 0
#                 if precision_num_all > 0:
#                     precision = recall_correct_num / precision_num_all
#                 else:
#                     precision = 0
#                 acc = sum(new_labels_predicted == labels_true) / len(new_labels_predicted)
#                 accelerator.print(
#                     'final acc: ' + str(acc))
#                 accelerator.print('final recall: ' + str(recall))
#                 accelerator.print('final precision: ' + str(precision))
#                 tb_writer.add_scalar("evaluation data/Acc:", acc, i)
#                 tb_writer.add_scalar("evaluation data/Recall:", recall, i)
#                 tb_writer.add_scalar("evaluation data/Precision:", precision, i)
#
#             if cfg['particle_csfile_path'] is not None:
#                 if not os.path.exists(cfg['path_result_dir'] + '/selected_particles/'):
#                     os.makedirs(cfg['path_result_dir'] + '/selected_particles/')
#                 if not os.path.exists(cfg['path_result_dir'] + '/selected_particles/selected_particles.cs'):
#                     selected_particles_id_all = [index for index, j in enumerate(labels_predicted) if j == 1]
#                     selected_particles = get_particles_from_cs(cfg['particle_csfile_path'], selected_particles_id_all)
#                     selected_particles.save(
#                         cfg['path_result_dir'] + '/selected_particles/selected_particles.cs')
#                     if len(selected_particles_id_all) > 0:
#                         cs2star(
#                             cfg['path_result_dir'] + '/selected_particles/selected_particles.cs',
#                             cfg['path_result_dir'] + '/selected_particles/selected_particles.star')
#
#                 final_selected_particles = get_particles_from_cs(cfg['particle_csfile_path'], new_selected_particles_id)
#                 final_selected_particles.save(
#                     cfg['path_result_dir'] + '/selected_particles/selected_particles_' + str(
#                         select_ratio) + '.cs')
#                 if len(new_selected_particles_id) > 0:
#                     cs2star(
#                         cfg['path_result_dir'] + '/selected_particles/selected_particles_' + str(
#                             select_ratio) + '.cs',
#                         cfg['path_result_dir'] + '/selected_particles/selected_particles_' + str(
#                             select_ratio) + '.star')




# def score_cut_step(value_len, scores_all, accelerator, num_remain, reverse=True):
#     id_scores_dict = dict(zip(list(range(value_len)), scores_all))
#     sorted_id_scores_dict = sorted(id_scores_dict.items(), key=lambda x: x[1], reverse=reverse)
#     new_selected_id = [i[0] for i in sorted_id_scores_dict[:num_remain]]
#     # accelerator.print('selected particles number with high scores: ' + str(len(new_selected_id)))
#     # new_labels_p=np.zeros(value_len)
#     # new_labels_p[new_selected_id]=1
#     return new_selected_id


def cryo_select_main(cfg=None,job_path=None,cache_file_path=None,accelerator=None,use_features=False,features_max_num=1000000):
    '''distribution'''
    if accelerator is None:
        accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=96000))])
    accelerator.print(cfg)


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

# def save_csfile_different_classes(path,labels,csfile,selected_index):
#     labels_set=list(set(labels))
#     for label in labels_set:
#         selected_particles_id = [selected_index[index] for index, j in enumerate(labels) if j == label]
#         selected_particles = csfile.take(selected_particles_id)
#         if not os.path.exists(path + '/selected_particles_cs/'):
#             os.makedirs(path + '/selected_particles_cs/')
#         if not os.path.exists(path + '/selected_particles_star/'):
#             os.makedirs(path + '/selected_particles_star/')
#         selected_particles.save(
#             path + '/selected_particles_cs/selected_particles_'+str(label)+'.cs')
#         if len(selected_particles_id) > 0:
#             cs2star(
#                 path + '/selected_particles_cs/selected_particles_'+str(label)+'.cs',
#                 path + '/selected_particles_star/selected_particles_'+str(label)+'.star')

# def resample_and_save_cs(cs_whole_particles,sorted_indices_dict,path,particle_number_truncation_point,except_class=[]):
#     k_set = list(sorted_indices_dict.keys())
#     for class_remove in except_class:
#         k_set.remove(class_remove)
#
#     selected_uids = []
#     key_ptr = [0 for _ in range(len(sorted_indices_dict.keys()))]
#     sorted_indices_list = [sorted(sorted_indices_dict[i].items(), key=lambda x: x[1], reverse=True) for i in range(len(sorted_indices_dict.keys()))]
#
#     particles_sum=0
#     for i in range(len(sorted_indices_dict.keys())):
#         particles_sum+=len(sorted_indices_dict[i])
#
#     while (len(selected_uids) < particle_number_truncation_point):
#         for int_key in k_set:
#             if int_key not in except_class:
#                 if (key_ptr[int_key] < len(sorted_indices_list[int_key])):
#                     selected_uids.append(sorted_indices_list[int_key][key_ptr[int_key]][0])
#                     key_ptr[int_key] += 1
#                 if (len(selected_uids) >= particle_number_truncation_point):
#                     break
#     output_particles_dataset = cs_whole_particles.query({'uid': selected_uids})
#     if not os.path.exists(path + '/selected_particles_cs/'):
#         os.makedirs(path + '/selected_particles_cs/')
#     if not os.path.exists(path + '/selected_particles_star/'):
#         os.makedirs(path + '/selected_particles_star/')
#     output_particles_dataset.save(path + '/selected_particles_cs/whole_selected_particles_'+str(len(selected_uids))+'.cs')
#     if len(selected_uids) > 0:
#         cs2star(
#             path + '/selected_particles_cs/whole_selected_particles_' + str(len(selected_uids)) + '.cs',
#             path + '/selected_particles_star/whole_selected_particles_'+str(particles_sum)+'_' + str(len(selected_uids)) + '.star')

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

    new_cs_data,scores,features_all=cryo_select_main(cfg=cfg,job_path=cfg['raw_data_path'],cache_file_path=cfg['path_result_dir'],use_features=True)





