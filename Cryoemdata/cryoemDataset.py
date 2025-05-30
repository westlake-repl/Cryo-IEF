# from models.get_transformers import to_int8
from Cryoemdata.data_preprocess.mrc_preprocess import to_int8
import numpy as np
# from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pickle
import random
# import mrcfile
# from PIL import Image
import torch
from io import BytesIO
import json
import lmdb
# from .utils import get_mean_error_distribution

# import time
# from torch.utils.data import DataLoader
# from memory_profiler import profile


class MyEmFile(object):
    def __init__(self, emfile_path=None, selected_emfile_path=None, filetype='star'):
        self.filetype = filetype
        if emfile_path:
            if emfile_path.endswith(".star"):
                self.particles_file_content, self.particles_star_title, self.particles_id = self.read_star(emfile_path)
                if selected_emfile_path is not None and selected_emfile_path.endswith(".star"):
                    self.selected_particles_file_content, self.selected_particles_star_file_title, self.selected_particles_id = self.read_star(
                        selected_emfile_path)
                    self.unselected_particles_file_content, self.unselected_particles_id = self.divide_selected_unselected_particles_star(
                        self.particles_file_content, self.particles_id, self.selected_particles_id)
                else:
                    self.selected_particles_id = None

            if emfile_path.endswith(".cs"):
                self.filetype = 'cs'
                self.particles_file_content, self.particles_id = self.read_cs(emfile_path)
                if selected_emfile_path is not None and selected_emfile_path.endswith(".cs"):
                    self.selected_particles_csfile_content, self.selected_particles_id = self.read_cs(
                        selected_emfile_path)
                    self.unselected_particles_csfile_content, self.unselected_particles_id = self.divide_selected_unselected_particles_cs(
                        self.particles_file_content, self.particles_id, self.selected_particles_id)
                    # pass
                else:
                    self.selected_particles_id = None
        else:
            self.particles_id = None
            self.filetype = None
            self.selected_particles_id = None
        # pass

    def read_star(self, star_path):
        with open(star_path, "r") as starfile:
            star_data = starfile.readlines()
        for index, x in enumerate(star_data):
            if x == 'data_particles\n':
                for index2, x2 in enumerate(star_data[index:]):

                    splited_x = x2.split()
                    next_splited_x = star_data[index + index2 + 1].split()
                    if splited_x:
                        item_num = splited_x[-1].replace("#", "")
                        if item_num.isdigit():
                            if int(item_num) == len(next_splited_x) and int(item_num) != len(splited_x):
                                start_site = index + index2 + 1
                                break
        content = star_data[start_site:]
        title = star_data[:start_site]
        img_ids = self.get_star_image_id(content)
        return content, title, img_ids

    def read_cs(self, cs_path):
        cs_data = Dataset.load(cs_path)
        img_ids = cs_data['uid'].tolist()
        # mm=cs_data['blob/path'].tolist()
        # dd=cs_data['blob/idx'].tolist()
        return cs_data, img_ids

    def get_star_image_id(self, star_content):
        image_id = []
        for x in star_content:
            if len(x.strip()) > 0:
                image_id.append(x.strip().split()[5])
        return image_id

    def divide_selected_unselected_particles_star(self, particles_star_content, particles_id, selected_particles_id):
        unselected_particles_star_content = [particles_star_content[index] for index, x in enumerate(particles_id) if
                                             len(x) > 0 and x not in selected_particles_id]
        unselected_particles_id = self.get_star_image_id(unselected_particles_star_content)
        return unselected_particles_star_content, unselected_particles_id

    def divide_selected_unselected_particles_cs(self, particles_cs_content, particles_id, selected_particles_id):
        unselected_list = []
        unselected_particles_id = []
        for i, id in enumerate(particles_id):
            if id not in selected_particles_id:
                unselected_list.append(i)
                unselected_particles_id.append(id)
        unselected_particles_cs_content = particles_cs_content.take(unselected_list)
        return unselected_particles_cs_content, unselected_particles_id


class MyMrcData(MyEmFile):
    def __init__(self, cfg, mrc_path=None, emfile_path=None, processed_data_path=None, selected_emfile_path=None,
                 tmp_data_save_path=None,
                 is_extra_valset=False, accelerator=None, ctf_correction_averages=False,
                 ctf_correction_inference=False):
        super(MyMrcData, self).__init__(emfile_path, selected_emfile_path)

        self.processed_data_path = processed_data_path
        if processed_data_path is not None:
            # with open(os.path.join(processed_data_path, 'path_divided_by_labels.data'), 'rb') as filehandle:
            #     self.path_divided_by_labels = pickle.load(filehandle)
            # with open(os.path.join(processed_data_path, 'ids_divided_by_labels.data'), 'rb') as filehandle:
            #     self.ids_divided_by_labels = pickle.load(filehandle)
            self.load_preprocessed_data_path(data_path=processed_data_path,
                                             ctf_correction_averages=ctf_correction_averages,
                                             ctf_correction_train=ctf_correction_inference)
        else:
            if is_extra_valset:
                tmp_data_save_path = tmp_data_save_path + '/tmp/preprocessed_data/extra_valset/'
            else:
                tmp_data_save_path = tmp_data_save_path + '/tmp/preprocessed_data/trainset/'
            self.mrc_path = mrc_path
            if not os.path.exists(tmp_data_save_path + '/output_tif_path.data') and accelerator.is_local_main_process:
                self.load_path()
                self.divided_into_single_mrc(tmp_data_save_path, resize=cfg['resize'], crop_ratio=cfg['crop_ratio'])
            accelerator.wait_for_everyone()
            self.load_preprocessed_data_path(data_path=tmp_data_save_path)

    def load_path(self):
        mrcs_path_list = []
        listdir(self.mrc_path, mrcs_path_list)
        self.mrcs_path_list = mrcs_path_list
        # if self.selected_particles_id is not None:
        # pass

    def load_preprocessed_data_path(self, data_path, ctf_correction_averages, ctf_correction_train):
        # path_out = path_result_dir + '/tmp/preprocessed_data/'
        if os.path.exists(data_path + '/output_tif_path.data'):
            with open(data_path + '/output_tif_path.data', 'rb') as filehandle:
                self.all_tif_path = pickle.load(filehandle)
        else:
            self.all_tif_path = None
        if ctf_correction_averages and os.path.exists(data_path + '/output_ctf_tif_path.data'):
            with open(data_path + '/output_ctf_tif_path.data', 'rb') as filehandle:
                self.all_tif_path_ctf_correction = pickle.load(filehandle)
        else:
            self.all_tif_path_ctf_correction = None

        if os.path.exists(data_path + '/lmdb_data'):
            lmdb_env = lmdb.open(
                data_path + '/lmdb_data/',
                readonly=True,
                lock=False,
                readahead=False
            )
            # self.lmdb_env = lmdb_env
            processed_tif_txn = lmdb_env.begin()
            self.length = processed_tif_txn.stat()['entries']
            lmdb_env.close()
            self.lmdb_path= data_path + '/lmdb_data/'
            self.all_processed_tif_path = None
        else:
            # self.processed_tif_txn = None
            self.lmdb_env = None
            self.lmdb_path = None

            with open(data_path + '/output_processed_tif_path.data',
                      'rb') as filehandle:
                self.all_processed_tif_path = pickle.load(filehandle)
            self.length = len(self.all_processed_tif_path)
        if ctf_correction_train and os.path.exists(data_path + '/output_ctf_processed_tif_path.data'):
            with open(data_path + '/output_ctf_processed_tif_path.data', 'rb') as filehandle:
                self.all_processed_tif_path_ctf_correction = pickle.load(filehandle)
        else:
            self.all_processed_tif_path_ctf_correction = None

        if os.path.exists(data_path + '/labels_for_clustering.data'):
            with open(data_path + '/labels_for_clustering.data', 'rb') as filehandle:
                self.labels_for_clustering = pickle.load(filehandle)
        else:
            self.labels_for_clustering = None

        if os.path.exists(data_path + '/labels_classification.data'):
            with open(data_path + '/labels_classification.data', 'rb') as filehandle:
                self.labels_classification = pickle.load(filehandle)
        else:
            self.labels_classification = [-1] * self.length

        # with open(path_out + 'output_tif_select_label.data', 'rb') as filehandle:
        #     self.tifs_selection_label = pickle.load(filehandle)

        if os.path.exists(data_path + '/means_stds.data'):
            with open(data_path + '/means_stds.data',
                      'rb') as filehandle:
                self.means_stds = pickle.load(filehandle)

        if os.path.exists(data_path + '/protein_id_list.data'):
            with open(data_path + '/protein_id_list.data',
                      'rb') as filehandle:
                self.protein_id_list = pickle.load(filehandle)
        else:
            self.protein_id_list = None

        if os.path.exists(data_path + '/protein_id_dict.data'):
            with open(data_path + '/protein_id_dict.data',
                      'rb') as filehandle:
                self.protein_id_dict = pickle.load(filehandle)
        else:
            self.protein_id_dict = None

        if os.path.exists(data_path + '/pretrain_data.json'):
            self.dataset_map = json.load(open(data_path + '/pretrain_data.json', 'r'))
        elif os.path.exists(data_path + '/finetune_data.json'):
            self.dataset_map = json.load(open(data_path + '/finetune_data.json', 'r'))
        else:
            self.dataset_map = None

        if os.path.exists(data_path + '/mean_error_dict.json'):
            mean_error_dict = json.load(open(data_path + '/mean_error_dict.json', 'r'))
            mean_error_dis_dict = get_mean_error_distribution(mean_error_dict)
            self.mean_error_dis_dict = {self.protein_id_dict[key]: value for key, value in mean_error_dis_dict.items()}
            # self.mean_error_dis_dict = {'good':{self.protein_id_dict[key]: value for key, value in mean_error_dis_dict['good'].items()},
            #                             'bad':{self.protein_id_dict[key]: value for key, value in mean_error_dis_dict['bad'].items()}}
        else:
            self.mean_error_dis_dict = None

        if os.path.exists(data_path + '/data_error_dict.json'):
            data_error_dict = json.load(open(data_path + '/data_error_dict.json', 'r'))
            self.data_error_dis_dict = {self.protein_id_dict[key]: np.array(value) for key, value in
                                        data_error_dict.items()}

        # with open(data_path + '/labels_for_training.data',
        #           'rb') as filehandle:
        #     self.labels_for_training = pickle.load(filehandle)
        # with open(data_path + '/probabilities_for_sampling.data',
        #           'rb') as filehandle:
        #     self.probabilities_for_sampling = pickle.load(filehandle)

    # def preprocess_trainset_valset_index(self, valset_name=[], dataset_except_names=[], is_balance=False,
    #                                      max_resample_num=None, max_resample_num_val=None, ratio_balance_train=[0.35,0.3,0.35],
    #                                      max_number_per_sample=None):
    #     valset_name_id = [self.protein_id_dict[name] for name in valset_name]
    #     dataset_except_names_id = [self.protein_id_dict[name] for name in dataset_except_names]
    #     id_sum_dict = {id: 0 for id in self.protein_id_dict.values()}
    #     dataset_index = []
    #     valset_index = []
    #     positive_index = []
    #     middle_index = []
    #     negative_index = []
    #     resample_num_p = 0
    #     resample_num_m=0
    #     resample_num_n = 0
    #     if len(valset_name_id) > 0 or len(dataset_except_names_id) > 0:
    #         for i, protein_id in enumerate(self.protein_id_list):
    #             if protein_id in valset_name_id:
    #                 valset_index.append(i)
    #             elif protein_id not in dataset_except_names_id:
    #                 dataset_index.append(i)
    #     else:
    #         # dataset_index = list(range(len(self.all_processed_tif_path)))
    #         dataset_index = list(range(self.length))
    #     if is_balance:
    #         for i in dataset_index:
    #             if self.labels_classification[i] == 1:
    #                 if max_number_per_sample is not None:
    #                     if id_sum_dict[self.protein_id_list[i]] < max_number_per_sample:
    #                         positive_index.append(i)
    #                         id_sum_dict[self.protein_id_list[i]] += 1
    #                 else:
    #                     positive_index.append(i)
    #             else:
    #                 if max_number_per_sample is not None:
    #                     if id_sum_dict[self.protein_id_list[i]] < max_number_per_sample:
    #                         negative_index.append(i)
    #                         id_sum_dict[self.protein_id_list[i]] += 1
    #                 else:
    #                     negative_index.append(i)
    #         if len(positive_index) > len(negative_index):
    #             resample_num = len(negative_index)
    #             # positive_index=random.sample(positive_index,len(negative_index))
    #         else:
    #             resample_num = len(positive_index)
    #             # negative_index=random.sample(negative_index,len(positive_index))
    #         if max_resample_num is not None:
    #             resample_num_p = int(max_resample_num * ratio_balance_train[1]) if max_resample_num * ratio_balance_train[1] < len(
    #                 positive_index) else resample_num
    #             resample_num_n = max_resample_num - resample_num_p
    #         else:
    #             resample_num_p = resample_num
    #             resample_num_n = resample_num
    #
    #         sub_positive_index = random.sample(positive_index, resample_num_p)
    #         sub_negative_index = random.sample(negative_index, resample_num_n)
    #         # if len(positive_index)>len(negative_index):
    #         #     positive_index=random.sample(positive_index,len(negative_index))
    #         # else:
    #         #     negative_index=random.sample(negative_index,len(positive_index))
    #         dataset_index = sub_positive_index + sub_negative_index
    #         if max_resample_num_val is not None:
    #             if len(valset_index) > max_resample_num_val:
    #                 valset_index = random.sample(valset_index, max_resample_num_val)
    #     elif max_resample_num is not None:
    #         if len(dataset_index) > max_resample_num:
    #             dataset_index = random.sample(dataset_index, max_resample_num)
    #     return dataset_index, valset_index, positive_index, negative_index, (resample_num_p, resample_num_n)

    # def preprocess_trainset_valset_index_finetune(self, valset_name=[], dataset_except_names=[],
    #                                               positive_ratio=0.5,
    #                                               max_number_per_sample=None, is_valset=False):
    #     if is_valset:
    #         id_index_dict_pos = {id: [] for name, id in self.protein_id_dict.items() if name.lower().endswith(
    #             'good') and name in valset_name and name not in dataset_except_names}
    #         id_index_dict_neg = {id: [] for name, id in self.protein_id_dict.items() if name.lower().endswith(
    #             'bad') and name in valset_name and name not in dataset_except_names}
    #     else:
    #         id_index_dict_pos = {id: [] for name, id in self.protein_id_dict.items() if name.lower().endswith(
    #             'good') and name not in valset_name and name not in dataset_except_names}
    #         id_index_dict_neg = {id: [] for name, id in self.protein_id_dict.items() if name.lower().endswith(
    #             'bad') and name not in valset_name and name not in dataset_except_names}
    #     protein_id_list_np = np.array(self.protein_id_list)
    #     for name, id in self.protein_id_dict.items():
    #         if name.lower().endswith('good'):
    #             id_index_dict_pos[id] = np.where(protein_id_list_np == id)[0].tolist()
    #         elif name.lower().endswith('bad'):
    #             id_index_dict_neg[id] = np.where(protein_id_list_np == id)[0].tolist()
    #     resample_num_p = int(max_number_per_sample * 4 * positive_ratio * len(id_index_dict_neg) / (
    #                 len(id_index_dict_pos) + len(id_index_dict_neg)))
    #     resample_num_n = int(max_number_per_sample * 2 - resample_num_p)
    #     return id_index_dict_pos, id_index_dict_neg, (resample_num_p, resample_num_n)

    def preprocess_trainset_valset_index_finetune(self, valset_name=[], dataset_except_names=[],
                                                  ratio_balance_train=[0.35, 0.3, 0.35],
                                                  max_number_per_sample=None, is_valset=False, is_balance=True,
                                                  middle_range_balance_train=[0.5, 0.85],
                                                  data_error_dis_dict=None):
        id_index_dict_pos = {}
        id_index_dict_neg = {}
        id_index_dict_mid = {}
        if data_error_dis_dict is not None:
            data_error_dis_dict_pos = {}
            data_error_dis_dict_neg = {}
            data_error_dis_dict_mid = {}

        protein_id_list_np = np.array(self.protein_id_list)
        labels_classification_np = np.array(self.labels_classification)
        for name, id in self.protein_id_dict.items():
            if name not in dataset_except_names:
                if name.lower().endswith('good'):
                    if name in valset_name and is_valset:
                        id_index_dict_pos[id] = np.where(protein_id_list_np == id)[0].tolist()
                    elif name not in valset_name and not is_valset:
                        id_index_dict_pos[id] = np.where(protein_id_list_np == id)[0].tolist()
                        if data_error_dis_dict is not None:
                            data_error_dis_dict_pos[id] = data_error_dis_dict[id] / np.sum(data_error_dis_dict[id])
                elif name.lower().endswith('bad'):
                    if name in valset_name and is_valset:
                        id_index_dict_neg[id] = np.where(protein_id_list_np == id)[0].tolist()
                    elif name not in valset_name and not is_valset:
                        id_index_dict_neg[id] = np.where(protein_id_list_np == id)[0].tolist()
                        if data_error_dis_dict is not None:
                            data_error_dis_dict_neg[id] = data_error_dis_dict[id] / np.sum(data_error_dis_dict[id])
                else:
                    protein_index = np.where(protein_id_list_np == id)[0]
                    pos_index = protein_index[labels_classification_np[protein_index] >= middle_range_balance_train[1]]
                    neg_index = protein_index[labels_classification_np[protein_index] < middle_range_balance_train[0]]
                    if data_error_dis_dict is not None:
                        pos_dis = data_error_dis_dict[id][
                            labels_classification_np[protein_index] >= middle_range_balance_train[1]]
                        neg_dis = data_error_dis_dict[id][
                            labels_classification_np[protein_index] < middle_range_balance_train[0]]
                    if middle_range_balance_train[0] != middle_range_balance_train[1]:
                        mid_index = protein_index[
                            (labels_classification_np[protein_index] >= middle_range_balance_train[0]) & (
                                    labels_classification_np[protein_index] < middle_range_balance_train[1])]
                        if len(mid_index) > 0:
                            id_index_dict_mid[id] = mid_index.tolist()
                            if data_error_dis_dict is not None:
                                mid_dis = data_error_dis_dict[id][
                                    (labels_classification_np[protein_index] >= middle_range_balance_train[0]) & (
                                            labels_classification_np[protein_index] < middle_range_balance_train[1])]
                                data_error_dis_dict_mid[id] = mid_dis / np.sum(mid_dis)
                        # id_index_dict_mid[id] = mid_index.tolist()
                    if name in valset_name and is_valset:
                        id_index_dict_pos[id] = pos_index.tolist()
                        id_index_dict_neg[id] = neg_index.tolist()
                    elif name not in valset_name and not is_valset:
                        if len(pos_index) > 0:
                            id_index_dict_pos[id] = pos_index.tolist()
                            if data_error_dis_dict is not None:
                                data_error_dis_dict_pos[id] = pos_dis / np.sum(pos_dis)
                        if len(neg_index) > 0:
                            id_index_dict_neg[id] = neg_index.tolist()
                            if data_error_dis_dict is not None:
                                data_error_dis_dict_neg[id] = neg_dis / np.sum(neg_dis)
        if is_balance:
            resample_num_p = int(max_number_per_sample * ratio_balance_train[2])
            resample_num_n = int(max_number_per_sample * ratio_balance_train[0])
            resample_num_m = int(max_number_per_sample * ratio_balance_train[1]) if len(id_index_dict_mid) > 0 else 0
            # if len(id_index_dict_mid)==0:
            #     resample_num_p = int(max_number_per_sample * 4 * ratio_balance_train[1] * len(id_index_dict_neg) / (
            #             len(id_index_dict_pos) + len(id_index_dict_neg)))
            #     resample_num_n = int(max_number_per_sample * 2 - resample_num_p)
            #     resample_num_m=0
            # else:
            #     # resample_num_p=int(6 * ratio_balance_train[1] * max_number_per_sample * (len(id_index_dict_neg) * len(id_index_dict_mid)) / (len(id_index_dict_pos) * len(id_index_dict_neg) + len(id_index_dict_neg) * len(id_index_dict_mid) + len(id_index_dict_pos) * len(id_index_dict_mid)))
            #     # resample_num_n=int(3*(1-positive_ratio)*max_number_per_sample*(len(id_index_dict_pos)*len(id_index_dict_mid))/(len(id_index_dict_pos)*len(id_index_dict_neg)+len(id_index_dict_neg)*len(id_index_dict_mid)+len(id_index_dict_pos)*len(id_index_dict_mid)))
            #     # resample_num_m=2*max_number_per_sample-resample_num_p-resample_num_n
            #     # resample_num_p = int(2 * positive_ratio * max_number_per_sample)
            #     # resample_num_n = int( max_number_per_sample-resample_num_p/2)
            #     # resample_num_m = 2*max_number_per_sample-resample_num_p-resample_num_n
            #     # ratio_multi=[1,1,1]
            #     resample_num_p = int(max_number_per_sample* ratio_balance_train[2])
            #     resample_num_n = int(max_number_per_sample* ratio_balance_train[0])
            #     resample_num_m = int(max_number_per_sample* ratio_balance_train[1])

        else:
            resample_num_p = max_number_per_sample
            resample_num_n = max_number_per_sample
            resample_num_m = max_number_per_sample
        data_error_dis_dict_all = {'good': data_error_dis_dict_pos, 'bad': data_error_dis_dict_neg,
                                   'mid': data_error_dis_dict_mid} if data_error_dis_dict is not None else {
            'good': None, 'bad': None, 'mid': None}
        return id_index_dict_pos, id_index_dict_neg, id_index_dict_mid, (
        resample_num_p, resample_num_n, resample_num_m), data_error_dis_dict_all

    def preprocess_trainset_index_pretrain(self, protein_id_dict=None, protein_id_list=None):
        if protein_id_dict is not None and protein_id_list is not None:
            target_protein_id_dict = protein_id_dict
            target_protein_id_list = protein_id_list
        else:
            target_protein_id_dict = self.protein_id_dict
            target_protein_id_list = self.protein_id_list
        bad_id_list_all = [target_protein_id_dict[name] for name in target_protein_id_dict.keys() if
                           name.lower().endswith('bad')]
        if self.dataset_map is None:

            # dataset_id_map=None
            id_map = None
            bad_id_list = None
        else:
            # id_index_dict = {target_protein_id_dict[name]: [] for name in self.dataset_map.keys()}
            id_map = {target_protein_id_dict[name]: target_protein_id_dict[name2] if name2 is not None else None for
                      name, name2 in self.dataset_map.items()}
            bad_id_list = [target_protein_id_dict[name] for name in self.dataset_map.keys() if
                           name.lower().endswith('bad')]

        dataset_id_map = {'id_map': id_map, 'bad_id_list': bad_id_list, 'bad_id_list_all': bad_id_list_all}
        # for i, id in enumerate(self.protein_id_list):
        #     id_index_dict[id].append(i)
        id_index_dict = {id: [] for id in target_protein_id_dict.values()}
        protein_id_list_np = np.array(target_protein_id_list)
        for id in target_protein_id_dict.values():
            # aaa = np.where(protein_id_list_np == id)
            id_index_dict[id] = np.where(protein_id_list_np == id)[0].tolist()
        return id_index_dict, dataset_id_map


class EMDataset_from_path(Dataset):
    """自定义数据集"""

    def __init__(self, mrcdata: MyMrcData, transform=None,
                 normal_scale=10, accelerator=None,
                 local_crops=None, slice_setting=None, weight_for_contrastive_classification_label=0.0,
                 use_triplex_labels=False, bar_score=0.4,
                 in_chans=1,needs_aug2=True
                 ):
        self.myindices = []
        self.tif_len = mrcdata.length
        self.lmdb_path=mrcdata.lmdb_path
        # if mrcdata.lmdb_path is not None:
        #     self.lmdb_env=lmdb.open(
        #         mrcdata.lmdb_path,
        #         readonly=True,
        #         # lock=False,
        #         # readahead=False
        #     )
        #     self.processed_tif_txn = self.lmdb_env.begin()
        # # if mrcdata.lmdb_env is not None:
        # #     self.processed_tif_txn = mrcdata.lmdb_env.begin()
        # else:
        #     self.processed_tif_txn = None
        # self.processed_tif_txn = mrcdata.processed_tif_txn
        self.tif_path_list = mrcdata.all_processed_tif_path
        self.tif_path_list_ctf_correction = mrcdata.all_processed_tif_path_ctf_correction
        self.tif_path_list_raw = mrcdata.all_tif_path
        self.tif_path_list_raw_ctf_correction = mrcdata.all_tif_path_ctf_correction
        self.labels_for_clustering = mrcdata.labels_for_clustering
        self.labels_classification = mrcdata.labels_classification
        self.slice_setting = slice_setting
        self.in_chans = in_chans
        self.needs_aug2=needs_aug2

        self.use_triplex_labels = use_triplex_labels
        if bar_score < 0.5:
            self.bar_score = 1 - bar_score
        else:
            self.bar_score = bar_score

        self.protein_id_list = mrcdata.protein_id_list
        self.protein_id_dict = mrcdata.protein_id_dict

        if mrcdata.particles_id is not None:
            self.particles_id = mrcdata.particles_id
        else:
            self.particles_id = range(self.tif_len)
        # self.isnorm = is_Normalize
        # self.mean_std = mrcdata.means_stds
        self.normal_scale = normal_scale
        self.transform = transform
        self.accelerator = accelerator
        # self.train=True

        # self.labels_for_training = mrcdata.labels_for_training
        # self.probabilities_for_sampling = mrcdata.probabilities_for_sampling
        self.processed_data_path = mrcdata.processed_data_path

        if local_crops is not None:
            self.local_crops_number = local_crops['number']
        else:
            self.local_crops_number = 0

        self.weight_for_contrastive_classification_label = weight_for_contrastive_classification_label
        if weight_for_contrastive_classification_label > 0:
            labels_classification_np = np.array(self.labels_classification)
            self.positive_items = np.where(labels_classification_np == 1)[0]
            self.negative_items = np.where(labels_classification_np == 0)[0]
        if slice_setting is not None and (slice_setting['p'] > 0 or slice_setting['align_p'] > 0) and slice_setting[
            'processed_path_slice'] is not None:
            with open(slice_setting['processed_path_slice'] + '/output_processed_tif_path.data', 'rb') as filehandle:
                self.tif_path_list_slice = pickle.load(filehandle)
        else:
            self.tif_path_list_slice = None

    def __len__(self):
        return self.tif_len
        # return len(self.tif_path_list)
    def open_lmdb(self):
        # if mrcdata.lmdb_path is not None:
        self.lmdb_env=lmdb.open(
            self.lmdb_path,
            readonly=True,
            meminit=False,
            max_readers=1,
            lock=False,
            readahead=False
        )
    # @profile(precision=4)
    def __getitem__(self, item):
        local_crops1 = []
        local_crops2 = []
        # end_git_item=time.time()
        item2 = None
        protein_id = self.protein_id_list[item]
        if self.weight_for_contrastive_classification_label > 0:
            if random.random() < self.weight_for_contrastive_classification_label:
                if self.labels_classification[item] == 1:
                    item2 = np.random.choice(self.positive_items)
                else:
                    item2 = np.random.choice(self.negative_items)

        # if self.processed_tif_txn is not None:
        #     key = f"{item}".encode()
        #     value = self.processed_tif_txn.get(key)
        #     data = torch.load(BytesIO(value))
        #     mrcdata = data['image_processed']
        #     tif_path = ''
        #     # raw_tif_path = ''
        #     del data,value
        if self.lmdb_path is not None:
            if not hasattr(self, 'lmdb_env'):
                self.open_lmdb()
            with self.lmdb_env.begin(write=False) as txn:
                key = f"{item}".encode()
                value = txn.get(key)
                data = torch.load(BytesIO(value))
                mrcdata = data['image_processed']
                tif_path = ''
                # raw_tif_path = ''
            del data, value
        else:
            tif_path = self.tif_path_list[item]
            if self.slice_setting is not None and (random.random() < self.slice_setting['p']):
                slice_path = random.choice(self.tif_path_list_slice)
                if os.path.exists(slice_path):
                    tif_path = slice_path





            try:
                with open(tif_path,
                          'rb') as filehandle:
                    mrcdata = pickle.load(filehandle)

            except EOFError:
                print('error for path: ' + tif_path)
            raw_tif_path = self.tif_path_list_raw[item]
        if mrcdata.mode != 'L':
            mrcdata = to_int8(mrcdata)
        # label_for_training=self.labels_for_training[item]
        # gaussian_probabilities=self.probabilities_for_sampling[item]
        if self.labels_for_clustering is not None and len(self.labels_for_clustering) > item:
            label_for_clustering = self.labels_for_clustering[item]
        else:
            label_for_clustering = -1
        label_for_classification = self.labels_classification[item]
        if self.use_triplex_labels:
            if label_for_classification > self.bar_score:
                label_for_classification = 1.0
            elif label_for_classification <= 1 - self.bar_score:
                label_for_classification = 0.0
            else:
                label_for_classification = 0.5

        particles_id = self.particles_id[item]

        if self.random_rotate_transform is not None:
            mrcdata_rotate1 = self.random_rotate_transform(mrcdata)
        else:
            mrcdata_rotate1 = mrcdata

        aug1 = self.transform(mrcdata_rotate1)
        aug2 = None

        if self.needs_aug2 and (item2 is not None or (self.slice_setting is not None and random.random() < self.slice_setting['p'])):
            if item2 is not None:
                tif_path2 = self.tif_path_list[item2] if item2 is not None else tif_path
            else:
                # tif_path2 = os.path.join(self.slice_path,'/'.join(tif_path.split('/')[-3:]))
                tif_path_split = tif_path.split('/')
                tif_path_split[-5] += '_slice'
                tif_path2 = '/'.join(tif_path_split)
            if os.path.exists(tif_path2):
                try:
                    with open(tif_path2,
                              'rb') as filehandle:
                        mrcdata2 = pickle.load(filehandle)

                except EOFError:
                    print('error for path: ' + tif_path)

                if mrcdata2.mode != 'L':
                    mrcdata2 = to_int8(mrcdata2)
                if self.random_rotate_transform is not None:
                    mrcdata_rotate2 = self.random_rotate_transform(mrcdata2)
                else:
                    mrcdata_rotate2 = mrcdata2
                aug2 = self.transform(mrcdata_rotate2)

        if aug2 is None and self.needs_aug2:
            if self.random_rotate_transform is not None:
                mrcdata_rotate2 = self.random_rotate_transform(mrcdata)
            else:
                mrcdata_rotate2 = mrcdata
            aug2 = self.transform(mrcdata_rotate2)

        for _ in range(self.local_crops_number):
            local_crops1.append(self.local_crops_transform(mrcdata_rotate1))
            if self.needs_aug2:
                local_crops2.append(self.local_crops_transform(mrcdata_rotate2))
        # imgs_all = [aug1, aug2] + local_crops1 + local_crops2

        if self.in_chans != 1:
            aug1 = aug1.repeat(self.in_chans, 1, 1)
            local_crops1 = [local_crop.repeat(self.in_chans, 1, 1) for local_crop in local_crops1]
            if self.needs_aug2:
                aug2 = aug2.repeat(self.in_chans, 1, 1)
                local_crops2 = [local_crop.repeat(self.in_chans, 1, 1) for local_crop in local_crops2]

        # img2tensor = transforms.ToTensor()
        # mrcdata = img2tensor(mrcdata)

        out = {
            # 'mrcdata': mrcdata,
            'aug1': aug1, 'aug2': aug2 if aug2 is not None else [],
            'label_for_clustering': label_for_clustering,
            'label_for_classification': label_for_classification,
            # 'path': tif_path,
            # 'raw_path': raw_tif_path,
            'particles_id': str(particles_id), 'item': item, 'local_crops1': local_crops1,
            'local_crops2': local_crops2, 'protein_id': protein_id}
        return out

    def get_transforms(self, transforms):
        self.transform = transforms[0]
        self.local_crops_transform = transforms[1]
        self.random_rotate_transform = transforms[2]


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file)[-1] == '.mrc' or os.path.splitext(file)[-1] == '.mrcs':
            list_name.append(file_path)


