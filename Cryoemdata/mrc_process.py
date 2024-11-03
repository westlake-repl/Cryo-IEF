from Cryoemdata.em_file_process import MyEmFile
import os
from tqdm import tqdm
import mrcfile
import random
from PIL import Image
import numpy as np
import multiprocessing
from functools import partial
import pickle
import math
import sys
from Cryoemdata.em_file_process import Dataset
import torch
from functools import partial
import multiprocessing
import json

class MyMrcData(MyEmFile):
    def __init__(self, mrc_path, cfg, emfile_path=None, processed_data_path=None, selected_emfile_path=None,
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

        with open(data_path + '/output_tif_path.data', 'rb') as filehandle:
            self.all_tif_path = pickle.load(filehandle)
        if ctf_correction_averages and os.path.exists(data_path + '/output_ctf_tif_path.data'):
            with open(data_path + '/output_ctf_tif_path.data', 'rb') as filehandle:
                self.all_tif_path_ctf_correction = pickle.load(filehandle)
        else:
            self.all_tif_path_ctf_correction = None

        with open(data_path + '/output_processed_tif_path.data',
                  'rb') as filehandle:
            self.all_processed_tif_path = pickle.load(filehandle)
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
            self.labels_classification = [-1] * len(self.all_processed_tif_path)

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


        # with open(data_path + '/labels_for_training.data',
        #           'rb') as filehandle:
        #     self.labels_for_training = pickle.load(filehandle)
        # with open(data_path + '/probabilities_for_sampling.data',
        #           'rb') as filehandle:
        #     self.probabilities_for_sampling = pickle.load(filehandle)

    def preprocess_trainset_valset_index(self, valset_name=[], dataset_except_names=[], is_balance=False,
                                         max_resample_num=None, max_resample_num_val=None, positive_ratio=0.5,
                                         max_number_per_sample=None):
        valset_name_id = [self.protein_id_dict[name] for name in valset_name]
        dataset_except_names_id = [self.protein_id_dict[name] for name in dataset_except_names]
        id_sum_dict = {id: 0 for id in self.protein_id_dict.values()}
        dataset_index = []
        valset_index = []
        positive_index = []
        negative_index = []
        resample_num_p = 0
        resample_num_n = 0
        if len(valset_name_id) > 0 or len(dataset_except_names_id) > 0:
            for i, protein_id in enumerate(self.protein_id_list):
                if protein_id in valset_name_id:
                    valset_index.append(i)
                elif protein_id not in dataset_except_names_id:
                    dataset_index.append(i)
        else:
            dataset_index = list(range(len(self.all_processed_tif_path)))
        if is_balance:
            for i in dataset_index:
                if self.labels_classification[i] == 1:
                    if max_number_per_sample is not None:
                        if id_sum_dict[self.protein_id_list[i]] < max_number_per_sample:
                            positive_index.append(i)
                            id_sum_dict[self.protein_id_list[i]] += 1
                    else:
                        positive_index.append(i)
                else:
                    if max_number_per_sample is not None:
                        if id_sum_dict[self.protein_id_list[i]] < max_number_per_sample:
                            negative_index.append(i)
                            id_sum_dict[self.protein_id_list[i]] += 1
                    else:
                        negative_index.append(i)
            if len(positive_index) > len(negative_index):
                resample_num = len(negative_index)
                # positive_index=random.sample(positive_index,len(negative_index))
            else:
                resample_num = len(positive_index)
                # negative_index=random.sample(negative_index,len(positive_index))
            if max_resample_num is not None:
                resample_num_p = int(max_resample_num * positive_ratio) if max_resample_num * positive_ratio < len(
                    positive_index) else resample_num
                resample_num_n = max_resample_num - resample_num_p
            else:
                resample_num_p = resample_num
                resample_num_n = resample_num

            sub_positive_index = random.sample(positive_index, resample_num_p)
            sub_negative_index = random.sample(negative_index, resample_num_n)
            # if len(positive_index)>len(negative_index):
            #     positive_index=random.sample(positive_index,len(negative_index))
            # else:
            #     negative_index=random.sample(negative_index,len(positive_index))
            dataset_index = sub_positive_index + sub_negative_index
            if max_resample_num_val is not None:
                if len(valset_index) > max_resample_num_val:
                    valset_index = random.sample(valset_index, max_resample_num_val)
        elif max_resample_num is not None:
            if len(dataset_index) > max_resample_num:
                dataset_index = random.sample(dataset_index, max_resample_num)
        return dataset_index, valset_index, positive_index, negative_index, (resample_num_p, resample_num_n)

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
                                                  positive_ratio=0.5,
                                                  max_number_per_sample=None, is_valset=False,is_balance=True):
        id_index_dict_pos ={}
        id_index_dict_neg = {}
        protein_id_list_np = np.array(self.protein_id_list)
        labels_classification_np = np.array(self.labels_classification)
        for name, id in self.protein_id_dict.items():
            if name not in dataset_except_names:
                if name.lower().endswith('good') :
                    if name in valset_name and is_valset:
                        id_index_dict_pos[id] = np.where(protein_id_list_np == id)[0].tolist()
                    elif name not in valset_name and not is_valset:
                        id_index_dict_pos[id] = np.where(protein_id_list_np == id)[0].tolist()
                elif name.lower().endswith('bad') :
                    if name in valset_name and is_valset:
                        id_index_dict_neg[id] = np.where(protein_id_list_np == id)[0].tolist()
                    elif name not in valset_name and not is_valset:
                        id_index_dict_neg[id] = np.where(protein_id_list_np == id)[0].tolist()
                else:
                    protein_index=np.where(protein_id_list_np == id)[0]
                    pos_index=protein_index[labels_classification_np[protein_index]==1]
                    neg_index=protein_index[labels_classification_np[protein_index]==0]
                    if name in valset_name and is_valset:
                        id_index_dict_pos[id] = pos_index.tolist()
                        id_index_dict_neg[id] = neg_index.tolist()
                    elif name not in valset_name and not is_valset:
                        id_index_dict_pos[id] = pos_index.tolist()
                        id_index_dict_neg[id] = neg_index.tolist()
        if is_balance:
            resample_num_p = int(max_number_per_sample * 4 * positive_ratio * len(id_index_dict_neg) / (
                    len(id_index_dict_pos) + len(id_index_dict_neg)))
            resample_num_n = int(max_number_per_sample * 2 - resample_num_p)
        else:
            resample_num_p = max_number_per_sample
            resample_num_n = max_number_per_sample
        return id_index_dict_pos, id_index_dict_neg, (resample_num_p, resample_num_n)
    def preprocess_trainset_index_pretrain(self,protein_id_dict=None,protein_id_list=None):
        if protein_id_dict is not None and protein_id_list is not None:
            target_protein_id_dict = protein_id_dict
            target_protein_id_list = protein_id_list
        else:
            target_protein_id_dict = self.protein_id_dict
            target_protein_id_list = self.protein_id_list
        if self.dataset_map is None:

            dataset_id_map=None
        else:
            # id_index_dict = {target_protein_id_dict[name]: [] for name in self.dataset_map.keys()}
            id_map = {target_protein_id_dict[name]: target_protein_id_dict[name2] if name2 is not None else None for name, name2 in self.dataset_map.items()}
            bad_id_list = [target_protein_id_dict[name] for name in self.dataset_map.keys() if name.lower().endswith('bad')]
            dataset_id_map = {'id_map': id_map, 'bad_id_list': bad_id_list}
        # for i, id in enumerate(self.protein_id_list):
        #     id_index_dict[id].append(i)
        id_index_dict = {id: [] for id in target_protein_id_dict.values()}
        protein_id_list_np = np.array(target_protein_id_list)
        for id in target_protein_id_dict.values():
            # aaa = np.where(protein_id_list_np == id)
            id_index_dict[id] = np.where(protein_id_list_np == id)[0].tolist()
        return id_index_dict,dataset_id_map

    def divided_into_single_mrc(self, path_result_dir, resize=None, is_norm=None, crop_ratio=None, filter_scale=None):

        path_out = path_result_dir
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        if crop_ratio:
            mask = get_crop_mask(resize, crop_ratio)
        value_error_mrcs = []
        self.output_tifs_path = []
        self.output_processed_tif_path = []
        self.labels_for_clustering = []
        raw_particles_dir = path_out + '/raw_particles/'
        processed_particles_dir = path_out + '/processed_particles/'

        pbar = tqdm(self.mrcs_path_list)
        pbar.set_description("mrcdata preprocessing")
        for file_id, file_path in enumerate(pbar):
            try:
                with mrcfile.open(file_path) as mrcdata:
                    p_list = file_path.split('/')
                    filename = p_list[-1]
                    mrcs = np.float32(mrcdata.data)
                    if resize and mrcs.shape[1] != resize:

                        processed_mrcs = mrcs_resize(mrcs, resize, resize)
                        # mrcs=processed_mrcs.copy()
                    else:
                        processed_mrcs = mrcs.copy()
                    if crop_ratio:
                        processed_mrcs = multi_process_crop(processed_mrcs, mask)

                    if self.filetype == 'cs':
                        self.process_mrcs_path_with_cs(file_id=file_id, file_path=file_path, filename=filename,
                                                       mrcs=mrcs,
                                                       processed_mrcs=processed_mrcs,
                                                       raw_particles_dir=raw_particles_dir,
                                                       processed_particles_dir=processed_particles_dir,
                                                       )
                    if self.filetype == None:
                        self.process_mrcs_path(file_id=file_id, p=file_path, filename=filename, mrcs=mrcs,
                                               processed_mrcs=processed_mrcs, raw_particles_dir=raw_particles_dir,
                                               processed_particles_dir=processed_particles_dir,
                                               )
                    # if self.filetype == 'star':
                    #     self.process_mrcs_path_with_star(file_id=file_id, p=file_path, filename=filename, mrcs=mrcs,
                    #                                      processed_mrcs=processed_mrcs,
                    #                                      raw_selected_dir=raw_particles_dir,
                    #
                    #                                      processed_selected_dir=processed_particles_dir,
                    #                                      )
            except ValueError:
                print('ValueError of ' + filename)
                value_error_mrcs.append(filename)
        # self.tifs_selection_label = output_tifs_select_label

        if value_error_mrcs:
            errorfilename = open(path_out + '/ValueError.txt', 'w')
            errorfilename.write(str(value_error_mrcs))
            errorfilename.close()
        with open(path_out + 'output_tif_path.data', 'wb') as filehandle:
            pickle.dump(self.output_tifs_path, filehandle)
        with open(path_out + 'output_processed_tif_path.data', 'wb') as filehandle:
            pickle.dump(self.output_processed_tif_path, filehandle)
        means_stds = get_mean_std(self.output_processed_tif_path)
        with open(path_out + 'means_stds.data', 'wb') as filehandle:
            pickle.dump(means_stds, filehandle)
        with open(path_out + 'output_tif_label.data', 'wb') as filehandle:
            pickle.dump(self.labels_for_clustering, filehandle)

    def process_mrcs_path(self, file_id, p, filename, mrcs, processed_mrcs, raw_particles_dir,
                          processed_particles_dir):
        mrcs_len = mrcs.shape[0]
        file_label = file_id
        label = 1
        for i in range(mrcs_len):
            n = str(i + 1).zfill(6)
            if not os.path.exists(raw_particles_dir + filename + '/'):
                os.makedirs(raw_particles_dir + filename + '/')
            if not os.path.exists(processed_particles_dir + filename + '/'):
                os.makedirs(processed_particles_dir + filename + '/')
            output_tif = raw_particles_dir + filename + '/' + n + '.data'
            processed_output_tif = processed_particles_dir + filename + '/' + n + '.data'

            # Image.fromarray(mrcs[i]).save(output_tif)
            # Image.fromarray(processed_mrcs[i]).save(processed_output_tif)

            with open(output_tif, 'wb') as filehandle:
                pickle.dump(Image.fromarray(mrcs[i]), filehandle)

            with open(processed_output_tif, 'wb') as filehandle:
                pickle.dump(Image.fromarray(processed_mrcs[i]), filehandle)

            self.output_tifs_path.append(output_tif)
            self.output_processed_tif_path.append(processed_output_tif)
            self.labels_for_clustering.append(file_label)

    def process_mrcs_path_with_cs(self, file_id, file_path, filename, mrcs, processed_mrcs, raw_particles_dir,
                                  processed_particles_dir):
        mrcs_path = os.path.join('/'.join(self.particles_file_content['blob/path'][0].split('/')[:-1]), filename)
        mrcs_id_list = self.particles_file_content.query({'blob/path': mrcs_path})['uid'].tolist()
        file_label = file_id
        for i, id in enumerate(mrcs_id_list):
            n = str(i + 1).zfill(6)
            label = 1
            if not os.path.exists(raw_particles_dir + filename + '/'):
                os.makedirs(raw_particles_dir + filename + '/')
            if not os.path.exists(processed_particles_dir + filename + '/'):
                os.makedirs(processed_particles_dir + filename + '/')
            output_tif = raw_particles_dir + filename + '/' + n + '.tif'
            processed_output_tif = processed_particles_dir + filename + '/' + n + '.tif'

            Image.fromarray(mrcs[i]).save(output_tif)
            Image.fromarray(processed_mrcs[i]).save(processed_output_tif)
            self.output_tifs_path.append(output_tif)
            self.output_processed_tif_path.append(processed_output_tif)

            self.labels_for_clustering.append(file_label)
    # def process_mrcs_path_with_star(self, file_id, p, filename, mrcs, processed_mrcs, raw_selected_dir,
    #                                 raw_unselected_dir, processed_selected_dir, processed_unselected_dir):
    #     p_list = p.split('/')
    #     mrcs_id = '/'.join(self.particles_id[0].split('@')[1].split('/')[0:3]) + '/' + p_list[-1]
    #     mrcs_len = mrcs.shape[0]
    #     file_label = file_id
    #     for i in range(mrcs_len):
    #         n = str(i + 1).zfill(6)
    #         label = 1
    #         mrc_id = n + '@' + mrcs_id
    #         if self.selected_particles_id:
    #
    #             if not os.path.exists(raw_selected_dir + filename + '/'):
    #                 os.makedirs(raw_selected_dir + filename + '/')
    #             if not os.path.exists(raw_unselected_dir + filename + '/'):
    #                 os.makedirs(raw_unselected_dir + filename + '/')
    #             if not os.path.exists(processed_selected_dir + filename + '/'):
    #                 os.makedirs(processed_selected_dir + filename + '/')
    #             if not os.path.exists(processed_unselected_dir + filename + '/'):
    #                 os.makedirs(processed_unselected_dir + filename + '/')
    #             if mrc_id in self.selected_particles_id:
    #
    #                 output_tif = raw_selected_dir + filename + '/' + n + '.tif'
    #                 processed_output_tif = processed_selected_dir + filename + '/' + n + '.tif'
    #
    #             else:
    #                 label = 0
    #
    #                 output_tif = raw_unselected_dir + filename + '/' + n + '.tif'
    #                 processed_output_tif = processed_unselected_dir + filename + '/' + n + '.tif'
    #
    #
    #         else:
    #             if not os.path.exists(raw_selected_dir + filename + '/'):
    #                 os.makedirs(raw_selected_dir + filename + '/')
    #             if not os.path.exists(processed_selected_dir + filename + '/'):
    #                 os.makedirs(processed_selected_dir + filename + '/')
    #             output_tif = raw_selected_dir + filename + '/' + n + '.tif'
    #             processed_output_tif = processed_selected_dir + filename + '/' + n + '.tif'
    #
    #         Image.fromarray(mrcs[i]).save(output_tif)
    #         Image.fromarray(processed_mrcs[i]).save(processed_output_tif)
    #         self.output_tifs_path.append(output_tif)
    #         self.output_processed_tif_path.append(processed_output_tif)
    #         self.tifs_label.append(file_label)


# def resample_from_id_index_dict(id_index_dict, max_number_per_sample):
#     resampled_index_list = []
#     for id, index_list in id_index_dict.items():
#         if len(index_list) > max_number_per_sample:
#             resampled_index_list.extend(random.sample(index_list, max_number_per_sample))
#         else:
#             resampled_index_list.extend(index_list)
#     return resampled_index_list
#
#
# def resample_from_id_index_dict_finetune(id_index_dict_pos, id_index_dict_neg, resample_num_p, resample_num_n):
#     resampled_index_list = []
#     resampled_index_list.extend(resample_from_id_index_dict(id_index_dict_pos, resample_num_p))
#     resampled_index_list.extend(resample_from_id_index_dict(id_index_dict_neg, resample_num_n))
#     return resampled_index_list


def raw_data_preprocess_one_mrcs(name, mrc_dir, raw_dataset_save_dir, processed_dataset_save_dir, resize=224,
                                 is_to_int8=True,indeices_per_mrcs_dict=None):
    mrcs_path = os.path.join(mrc_dir, name)
    mrcs = np.float32(mrcfile.open(mrcs_path).data)
    mrcs_len = mrcs.shape[0]
    raw_single_particle_path = []
    processed_single_particle_path = []
    if indeices_per_mrcs_dict is None:
        ids_list = range(mrcs_len)
    else:
        ids_list=indeices_per_mrcs_dict[name].tolist()

    if resize and mrcs.shape[1] != resize:

        processed_mrcs = mrcs_resize(mrcs, resize, resize)
    else:
        processed_mrcs = mrcs.copy()

    if is_to_int8:
        processed_mrcs = mrcs_to_int8(processed_mrcs)

    for j in ids_list:

        n = str(j + 1).zfill(6)


        if not os.path.exists(os.path.join(processed_dataset_save_dir, name)):
            os.makedirs(os.path.join(processed_dataset_save_dir, name))
        with open(os.path.join(processed_dataset_save_dir, name, n + '.data'), 'wb') as filehandle:
            pickle.dump(Image.fromarray(processed_mrcs[j]).convert('L'), filehandle)

        processed_single_particle_path.append(os.path.join(processed_dataset_save_dir, name, n + '.data'))

        if raw_dataset_save_dir is not None:
            if not os.path.exists(os.path.join(raw_dataset_save_dir, name)):
                os.makedirs(os.path.join(raw_dataset_save_dir, name))
            with open(os.path.join(raw_dataset_save_dir, name, n + '.data'), 'wb') as filehandle:
                pickle.dump(Image.fromarray(mrcs[j]), filehandle)
            raw_single_particle_path.append(os.path.join(raw_dataset_save_dir, name, n + '.data'))
        else:
            raw_single_particle_path.append('')

    return raw_single_particle_path, processed_single_particle_path

def combine_cs_files_column(cs_path1, cs_path2):
    cs_data1=Dataset.load(cs_path1)
    cs_data2=Dataset.load(cs_path2)
    cs_data=Dataset.innerjoin(cs_data1,cs_data2)
    # save_dir='/'.join(save_path.split('/')[:-1])
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # cs_data.save(save_path)
    # print('combined cs file saved in {}'.format(save_path))
    return cs_data
    # cs2star(save_path,save_path.replace('.cs','.star'))
def raw_csdata_process_from_cryosparc_dir(raw_data_path):
    passthrough_particles_path = None
    particles_cs_path = None
    for filename in os.listdir(raw_data_path):
        if filename.endswith('passthrough_particles.cs'):

            passthrough_particles_path = os.path.join(raw_data_path, filename)

        if filename.endswith('_split_0.cs'):

            passthrough_particles_path = os.path.join(raw_data_path, filename)

        if filename.endswith('extracted_particles.cs'):

            particles_cs_path = os.path.join(raw_data_path, filename)


        if filename.endswith('imported_particles.cs'):

            particles_cs_path = os.path.join(raw_data_path, filename)

        if filename.endswith('restacked_particles.cs'):

            particles_cs_path = os.path.join(raw_data_path, filename)

        if filename.endswith('split_0000.cs'):

            particles_cs_path = os.path.join(raw_data_path, filename)

    if passthrough_particles_path is not None:
        cs_data = combine_cs_files_column(particles_cs_path, passthrough_particles_path)
    elif particles_cs_path is not None:
        cs_data = Dataset.load(particles_cs_path)
    else:
        Exception(raw_data_path+': corresponding  not exists!')

    if os.path.exists(os.path.join(raw_data_path, 'restack')):
        mrc_dir = os.path.join(raw_data_path, 'restack')
    elif os.path.exists(os.path.join(raw_data_path, 'extract')):
        mrc_dir = os.path.join(raw_data_path, 'extract')
    elif os.path.exists(os.path.join(raw_data_path, 'imported')):
        mrc_dir = os.path.join(raw_data_path, 'imported')
    elif particles_cs_path.endswith('split_0000.cs'):
        raw_dir='/'.join(raw_data_path.split('/')[0:-2])
        mrc_dir = raw_dir+'/'+'/'.join(cs_data['blob/path'][0].split('/')[0:-1])+'/'
    return cs_data,mrc_dir
    # if not os.path.exists(processed_data_save_path):
    #     os.makedirs(processed_data_save_path)
    # new_cs_data_path=raw_data_preprocess(raw_data_path, processed_data_save_path,cs_data=cs_data)
    # print('new cs data saved in {}'.format(new_cs_data_path))
def raw_data_preprocess(raw_dataset_dir, dataset_save_dir,  resize=224, is_to_int8=True,save_raw_data=False):

    # if os.path.exists(os.path.join(raw_dataset_dir, 'restack')):
    #     mrc_dir = os.path.join(raw_dataset_dir, 'restack')
    # elif os.path.exists(os.path.join(raw_dataset_dir, 'extract')):
    #     mrc_dir = os.path.join(raw_dataset_dir, 'extract')
    # elif os.path.exists(os.path.join(raw_dataset_dir, 'imported')):
    #     mrc_dir = os.path.join(raw_dataset_dir, 'imported')
    # elif


    cs_data,mrc_dir=raw_csdata_process_from_cryosparc_dir(raw_dataset_dir)
    blob_path_list=cs_data['blob/path'].tolist()
    mrcs_names_list = [blob_path_list[i].split('/')[-1] for i in
                       range(len(blob_path_list))]

    mrcs_names_list_process = list(dict.fromkeys(mrcs_names_list))

    #Optimization speeds up this code
    # if cs_data is not None:
    print("Processing cs_data...")
    mrcs_names_np = np.array(mrcs_names_list)
    # aaa=np.where((mrcs_names_np == 'batch_17790_restacked.mrc'))[0]
    # sub_cs=cs_data.take(aaa)
    # mrcs_names_list_sub = sub_cs['blob/path'].tolist()
    # blob_idx_sub = sub_cs['blob/idx'].tolist()
    # Create a dictionary where the keys are names and the values are lists of indices
    blob_idx_np=_np = np.array(cs_data['blob/idx'].tolist())
    sorted_indices = np.argsort(mrcs_names_np)
    sorted_names = mrcs_names_np[sorted_indices]
    unique_names, counts = np.unique(sorted_names, return_counts=True)
    split_indices = np.split(sorted_indices, np.cumsum(counts)[:-1])
    indices_dict = dict(zip(unique_names, split_indices))
    indeices_per_mrcs_dict={}

    for name, indices in indices_dict.items():
        # Convert indices to numpy array
        indices_np = np.array(indices)

        # Get corresponding values in blob_idx_np
        values = blob_idx_np[indices_np]

        # Get the sorted indices based on the values
        sorted_indices = np.argsort(values)

        # Update indices in indices_dict in-place
        indices_dict[name] = indices_np[sorted_indices]
        indeices_per_mrcs_dict[name]=np.sort(values)

    # indices_dict = {name: np.where(mrcs_names_np == name)[0] for name in mrcs_names_list_process}
    # Use the dictionary to perform the operations
    # new_cs_data = cs_data.take(indices_dict[mrcs_names_list_process[0]])
    # for i, name in enumerate(mrcs_names_list_process[1:]):
    #     new_cs_data = Dataset.append(new_cs_data, cs_data.take(indices_dict[name]))
    func_append_data=partial(append_data,cs_data=cs_data,indices_dict=indices_dict)
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(func_append_data, mrcs_names_list_process)
    new_cs_data = Dataset.append(results[0], *results[1:])

    # new_blob_path_list = new_cs_data['blob/path'].tolist()
    # new_mrcs_names_list = [new_blob_path_list[i].split('/')[-1] for i in
    #                    range(len(new_blob_path_list))]
    # new_mrcs_names_np=np.array(new_mrcs_names_list)
    # aaa=np.where((new_mrcs_names_np == 'batch_17790_restacked.mrc'))[0]
    # sub_cs=new_cs_data.take(aaa)
    # mrcs_names_list_sub = sub_cs['blob/path'].tolist()
    # blob_idx_sub = sub_cs['blob/idx'].tolist()


    # new_mrcs_names_list = new_cs_data['blob/path'].tolist()
    # new_blob_idx=new_cs_data['blob/idx'].tolist()
    # blob_idx=cs_data['blob/idx'].tolist()

    if not os.path.exists(dataset_save_dir):
        os.makedirs(dataset_save_dir)
    new_csdata_path = os.path.join(dataset_save_dir, 'new_particles.cs')
    new_cs_data.save(new_csdata_path)
    particles_dir_name = raw_dataset_dir.split('/')[-1]

    if save_raw_data:
        raw_dataset_save_dir = os.path.join(dataset_save_dir, 'raw', particles_dir_name)
        if not os.path.exists(raw_dataset_save_dir):
            os.makedirs(raw_dataset_save_dir)
    else:
        raw_dataset_save_dir = None

    processed_dataset_save_dir = os.path.join(dataset_save_dir, 'processed', particles_dir_name)
    if not os.path.exists(processed_dataset_save_dir):
        os.makedirs(processed_dataset_save_dir)
    raw_path_list = []
    processed_path_list = []
    labels_classification_list = []
    labels_for_clustering_list = []
    particles_sum = 0

    phbar = tqdm(mrcs_names_list_process, desc='data preprocessing')
    func = partial(raw_data_preprocess_one_mrcs, mrc_dir=mrc_dir, raw_dataset_save_dir=raw_dataset_save_dir,
                   processed_dataset_save_dir=processed_dataset_save_dir, resize=resize, is_to_int8=is_to_int8,indeices_per_mrcs_dict=indeices_per_mrcs_dict)
    pool = multiprocessing.Pool(20)
    results = pool.map(func, phbar)
    pool.close()
    pool.join()

    for raw_single_particle_path, processed_single_particle_path in results:
        processed_path_list += processed_single_particle_path
        raw_path_list += raw_single_particle_path

        labels_classification_list += [-1] * len(raw_single_particle_path)
        labels_for_clustering_list += [-1] * len(raw_single_particle_path)

    processed_path_list_save_path = os.path.join(dataset_save_dir, 'processed_path_list.data')
    with open(os.path.join(dataset_save_dir, 'output_processed_tif_path.data'), 'wb') as filehandle:
        pickle.dump(processed_path_list, filehandle)

    with open(os.path.join(dataset_save_dir, 'output_tif_path.data'), 'wb') as filehandle:
        pickle.dump(raw_path_list, filehandle)

    # mean_std = sample_and_calculate_mean_std(processed_path_list)
    # with open(save_path + 'means_stds.data', 'wb') as filehandle:
    #     pickle.dump(mean_std, filehandle)
    with open(os.path.join(dataset_save_dir, 'labels_classification.data'), 'wb') as filehandle:
        pickle.dump(labels_classification_list, filehandle)
    with open(os.path.join(dataset_save_dir, 'labels_for_clustering.data'), 'wb') as filehandle:
        pickle.dump(labels_for_clustering_list, filehandle)
    print('raw data process all done')
    return new_cs_data


def append_data(name,cs_data,indices_dict):
    # mm=np.sort(indices_dict[name])
    return cs_data.take(indices_dict[name])

def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file)[-1] == '.mrc' or os.path.splitext(file)[-1] == '.mrcs':
            list_name.append(file_path)


def mrcs_resize(mrcs, width, height, is_norm=False):
    resized_mrcs = np.zeros((mrcs.shape[0], width, height))
    # pbar = tqdm(range(mrcs.shape[0]))
    # pbar.set_description("resize mrcs to width*height")
    for i in range(mrcs.shape[0]):
        mrc = mrcs[i]
        # if is_norm:
        #     mrc = (mrc - np.min(mrc)) * (30 / (np.max(mrc) - np.min(mrc)))
        #     mrc = mrc - np.mean(mrc)
        mrc = Image.fromarray(mrc)
        resized_mrcs[i] = np.asarray(mrc.resize((width, height), Image.BICUBIC))
    resized_mrcs = resized_mrcs.astype('float32')
    return resized_mrcs


# def norm(mrcs):
#     norm_mrcs = np.zeros((mrcs.shape[0], mrcs.shape[1], mrcs.shape[2]))
#     # pbar = tqdm(range(mrcs.shape[0]))
#     # pbar.set_description("resize mrcs to width*height")
#     for i in range(mrcs.shape[0]):
#         mrc = mrcs[i]
#         mrc = (mrc - np.min(mrc)) * (30 / (np.max(mrc) - np.min(mrc)))
#         mrc = mrc - np.mean(mrc)
#         norm_mrcs[i] = mrc
#     norm_mrcs = norm_mrcs.astype('float32')
#     return norm_mrcs


def get_mean_std(path_list, Cnum=10000):
    # calculate means and stds
    imgs = []
    random.shuffle(path_list)
    if len(path_list) < Cnum:
        Cnum = len(path_list)
    for i in range(Cnum):
        img = np.array(pickle.load(open(path_list[i], 'rb')))
        imgs.append(img)
    imgs_np = np.asarray(imgs)
    means = imgs_np.mean()
    stds = imgs_np.std()
    return means, stds


def multi_process_crop(mrcsArray, mask, is_norm=False):
    # print('mrcs array is cropping:')
    items = [mrcsArray[i] for i in range(mrcsArray.shape[0])]
    pool = multiprocessing.Pool(8)
    # pbar = tqdm(items)
    # pbar.set_description("mrcs array cropping")
    func = partial(image_cropping, mask=mask)
    mrcsArray = pool.map(func, items)
    pool.close()
    pool.join()
    return np.asarray(mrcsArray)


def get_crop_mask(mrc_length, crop_ratio):
    mask = np.zeros([mrc_length, mrc_length])
    center = np.array([mrc_length / 2, mrc_length / 2])
    for i in range(mrc_length):
        for j in range(mrc_length):
            if math.sqrt(
                    (i - int(center[0])) ** 2 + (j - int(center[1])) ** 2) < crop_ratio * mrc_length * 0.5:
                mask[i, j] = 1
    return mask


def image_cropping(img, mask):
    return img * mask


def write_list(l, path):
    f = open(path, "w")
    for line in l:
        f.write(line + '\n')
    f.close()


def to_int8(mrcdata):
    if torch.is_tensor(mrcdata):
        mrcdata = (mrcdata - torch.min(mrcdata)) / (torch.max(mrcdata) - torch.min(mrcdata))
        mrcdata = (mrcdata * 255).type(torch.uint8)
    else:
        mrcdata = (mrcdata - np.min(mrcdata)) / ((np.max(mrcdata) - np.min(mrcdata)))
        mrcdata = (mrcdata * 255).astype(np.uint8)
    return mrcdata


def mrcs_to_int8(mrcs):
    if torch.is_tensor(mrcs):
        new_mrcs = torch.zeros_like(mrcs, dtype=torch.uint8)
    else:
        new_mrcs = np.zeros_like(mrcs, dtype=np.uint8)
    for i in range(mrcs.shape[0]):
        new_mrcs[i] = to_int8(mrcs[i])
    return new_mrcs



