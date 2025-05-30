
import os
import yaml
from easydict import EasyDict
from argparse import ArgumentParser
import numpy as np
import pickle
from collections import defaultdict
from cryosparc.dataset import Dataset
from CryoIEF_inference import  cryo_features_main
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import kmeans_plusplus, MiniBatchKMeans
from Cryoemdata.cs_star_translate.cs2star import cs2star
from accelerate.utils import InitProcessGroupKwargs
from accelerate import Accelerator
from datetime import timedelta

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

def features_kmeans(mrcArray_features, k, cut_ratio=0.0, job=None, merge_threshold=0.0, centers_sampling=True, n_runs=10):
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

    return labels_iter, centers_iter, np.array(num_class)

def cryo_clustering_main(cfg,accelerator=None):
    if accelerator is None:
        accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=96000))])
    if cfg['path_result_dir'] is None:
        raise ValueError('path_result_dir is required to save the clustering results')
    if accelerator.is_main_process:
        if not os.path.exists(cfg['path_result_dir']):
            os.makedirs(cfg['path_result_dir'])
    accelerator.wait_for_everyone()
    if cfg['features_path'] is None:
        cryo_features_main(cfg, accelerator=accelerator)
        cfg['features_path'] = cfg['path_result_dir']
    if accelerator.is_main_process:
        results_path=cfg['path_result_dir']+'/clustering_'+str(cfg['k'])+'/'
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        cs_data=Dataset.load(cfg['features_path']+'/processed_data/new_particles.cs')
        features=pickle.load(open(cfg['features_path']+'/features_all.data','rb'))
        labels_predict, centers_predict, num_class = features_kmeans(np.array(features), cfg['k'])
        for i in range(cfg['k']):
            print(f"cluster {i} num: {num_class[i]}")
            cs_sub_data=cs_data.take(labels_predict==i)
            cs_save_path=results_path+'/cluster_'+str(i)+'.cs'
            cs_sub_data.save(cs_save_path)
            cs2star(cs_save_path,cs_save_path.replace('.cs','.star'))


if __name__ == '__main__':
    '''get config'''
    parser = ArgumentParser()
    parser.add_argument('--path_result_dir', default=None, type=str)
    parser.add_argument('--raw_data_path', default=None, type=str)
    parser.add_argument('--processed_data_path', default=None, type=str)
    parser.add_argument('--features_path', default=None, type=str)
    parser.add_argument('--path_model_proj', default=None, type=str)

    parser.add_argument('--batch_size', default=None, type=int)
    # parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--k', default=None, type=int)


    args = parser.parse_args()

    cfg = EasyDict()
    # if args.path_proj_dir is None:
    args.path_proj_dir = os.path.dirname(os.path.abspath(__file__))

    with open(
            args.path_proj_dir + '/CryoClustering/clustering_inference_settings.yml',
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

    if args.processed_data_path is not None:
        if args.processed_data_path.lower() == 'none' or args.processed_data_path.lower() == 'null':
            cfg['processed_data_path'] = None
        else:
            cfg['processed_data_path'] = args.processed_data_path

    if args.features_path is not None:
        if args.features_path.lower() == 'none' or args.features_path.lower() == 'null':
            cfg['features_path'] = None
        else:
            cfg['features_path'] = args.features_path

    if args.path_model_proj is not None:
        cfg['path_model_proj'] = args.path_model_proj

    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size

    if args.k is not None:
        cfg['k'] = args.k

    '''Main function'''
    cryo_clustering_main(cfg=cfg)
