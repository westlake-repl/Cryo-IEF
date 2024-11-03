from PIL import Image
import yaml
from easydict import EasyDict
# import math
import mrcfile
import numpy as np
# import random
import os
from tqdm import tqdm
import multiprocessing
from functools import partial
from argparse import ArgumentParser
from cryosparc.dataset import Dataset
import cryosparc.mrc as mrc
import math
import pickle
from collections import Counter
import pandas as pd
# from Cryoemdata.cs_star_translate.cs2star import cs2star


def divide_selected_particles_id(clustering_result, selected_classes,uncertainty_list=None):
    selected_particles = []
    selected_particles_uncertainty = []
    unselected_particles = []
    unselected_particles_uncertainty = []
    for i in selected_classes:
        # selected_particles += clustering_result[clustering_result['clustering_results'] == i].index.tolist()
        selected_particles += [index for index,j in enumerate(clustering_result) if j==i]
        unselected_particles += [index for index,j in enumerate(clustering_result) if j!=i]
    if uncertainty_list is not None:
        for i in selected_classes:
            selected_particles_uncertainty += [uncertainty_list[index] for index,j in enumerate(clustering_result) if j==i]
            unselected_particles_uncertainty += [uncertainty_list[index] for index,j in enumerate(clustering_result) if j!=i]
    return selected_particles, unselected_particles, selected_particles_uncertainty, unselected_particles_uncertainty



def get_particles_from_cs(cs_path, cs_id):
    cs_data = Dataset.load(cs_path)
    new_cs_data = cs_data.take(cs_id)
    return new_cs_data


def main(save_dir, target_cs_path, clustering_result_path, selected_classes):
    clustering_result = pd.read_csv(clustering_result_path)
    selected_particles_id = divide_selected_particles_id(clustering_result, selected_classes)
    selected_particles = get_particles_from_cs(target_cs_path, selected_particles_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('selected particles number:{}'.format(len(selected_particles)))
    selected_particles.save(save_dir + 'selected_particles.cs')
    cs2star(save_dir + 'selected_particles.cs', save_dir + 'selected_particles.star')
    print('selected particles cs file saved in {}'.format(save_dir + 'selected_particles.cs'))
    print('selected particles star file saved in {}'.format(save_dir + 'selected_particles.star'))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--target_cs_path', default=None, type=str)
    parser.add_argument('--clustering_result_path', default=None, type=str)
    parser.add_argument('--selected_classes', nargs='+', type=float, default=None)

    args = parser.parse_args()

    '''get config'''
    cfg = EasyDict()
    with open('/yanyang2/projects/particle_clustering/select_particles_settings.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    for k, v in config.items():
        cfg[k] = v

    if args.save_dir is not None:
        cfg['save_dir'] = args.save_dir

    if args.target_cs_path is not None:
        cfg['target_cs_path'] = args.target_cs_path

    if args.clustering_result_path is not None:
        cfg['clustering_result_path'] = args.clustering_result_path

    if args.selected_classes is not None:
        cfg['selected_classes'] = args.selected_classes

    print(cfg)

    main(cfg['save_dir'], cfg['target_cs_path'], cfg['clustering_result_path'], cfg['selected_classes'])
