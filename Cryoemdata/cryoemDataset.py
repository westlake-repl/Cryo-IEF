from Cryo_IEF.get_transformers import to_int8
import numpy as np
# from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from .mrc_process import MyMrcData
import pickle
import random
import mrcfile
from PIL import Image
import torch


# import time
# from torch.utils.data import DataLoader
# from memory_profiler import profile
class EMDataset_from_path(Dataset):
    '''Self-defined cryoEM dataset'''

    def __init__(self, mrcdata: MyMrcData, transform=None,
                 is_Normalize=None, normal_scale=10, train_with_labels=None, device='cuda', accelerator=None,
                 local_crops=None, percent_with_ctf=0, use_weight_for_classification=False
                 ):
        self.myindices = []
        self.tif_path_list = mrcdata.all_processed_tif_path
        self.tif_path_list_ctf_correction = mrcdata.all_processed_tif_path_ctf_correction
        self.tif_path_list_raw = mrcdata.all_tif_path
        self.tif_path_list_raw_ctf_correction = mrcdata.all_tif_path_ctf_correction
        self.labels_for_clustering = mrcdata.labels_for_clustering
        self.labels_classification = mrcdata.labels_classification
        self.percent_with_ctf = percent_with_ctf

        self.protein_id_list = mrcdata.protein_id_list
        self.protein_id_dict = mrcdata.protein_id_dict

        if mrcdata.particles_id is not None:
            self.particles_id = mrcdata.particles_id
        else:
            self.particles_id = range(len(self.tif_path_list))
        # self.isnorm = is_Normalize
        # self.mean_std = mrcdata.means_stds
        self.normal_scale = normal_scale
        self.transform = transform
        if train_with_labels is not None:
            self.with_labels = train_with_labels['is_train_with_labels']
            self.with_labels_p = train_with_labels['p']
        else:
            self.with_labels = False
            self.with_labels_p = 0
        self.accelerator = accelerator
        # self.train=True

        # self.labels_for_training = mrcdata.labels_for_training
        # self.probabilities_for_sampling = mrcdata.probabilities_for_sampling
        self.processed_data_path = mrcdata.processed_data_path

        if local_crops is not None:
            self.local_crops_number = local_crops['number']
        else:
            self.local_crops_number = 0

        self.use_weight_for_classification = use_weight_for_classification
        if use_weight_for_classification:
            labels_classification_np = np.array(self.labels_classification)
            self.positive_items = np.where(labels_classification_np == 1)[0]
            self.negative_items = np.where(labels_classification_np == 0)[0]

    def __len__(self):
        return len(self.tif_path_list)

    # @profile(precision=4)
    def __getitem__(self, item):
        local_crops1 = []
        local_crops2 = []
        # end_git_item=time.time()

        if self.use_weight_for_classification:
            if random.random() > 0.5:
                item = np.random.choice(self.positive_items)
            else:
                item = np.random.choice(self.negative_items)

        if self.tif_path_list_ctf_correction is not None and random.random() > self.percent_with_ctf and \
                self.tif_path_list_ctf_correction[item] is not None:
            tif_path = self.tif_path_list_ctf_correction[item]
            labels_for_training_path = tif_path.replace('ctf_correction/processed', 'labels_for_training')
        else:
            tif_path = self.tif_path_list[item]
            labels_for_training_path = tif_path.replace('processed', 'labels_for_training')
        if self.tif_path_list_raw_ctf_correction is not None and self.tif_path_list_raw_ctf_correction[
            item] is not None:
            raw_tif_path = self.tif_path_list_raw_ctf_correction[item]
        else:
            raw_tif_path = self.tif_path_list_raw[item]

        try:
            with open(tif_path,
                      'rb') as filehandle:
                mrcdata = pickle.load(filehandle)

        except EOFError:
            print('error for path: ' + tif_path)

        if mrcdata.mode != 'L':
            mrcdata = to_int8(mrcdata)
        # label_for_training=self.labels_for_training[item]
        # gaussian_probabilities=self.probabilities_for_sampling[item]
        if self.labels_for_clustering is not None and len(self.labels_for_clustering) > 0:
            label_for_clustering = self.labels_for_clustering[item]
        else:
            label_for_clustering = -1
        label_for_classification = self.labels_classification[item]
        particles_id = self.particles_id[item]

        if self.random_rotate_transform is not None:
            mrcdata_rotate1 = self.random_rotate_transform(mrcdata)
        else:
            mrcdata_rotate1 = mrcdata

        aug1 = self.transform(mrcdata_rotate1)

        if self.with_labels and os.path.exists(labels_for_training_path) and random.random() < self.with_labels_p:
            # a=near_ids_and_p[0]
            # b=near_ids_and_p[1]
            with open(labels_for_training_path,
                      'rb') as filehandle:
                labels_for_training_item = pickle.load(filehandle)

            random_id = int(np.random.choice(labels_for_training_item[0], p=labels_for_training_item[1]))
            with open(self.tif_path_list[random_id],
                      'rb') as filehandle:
                mrcdata2 = pickle.load(filehandle)

            if mrcdata2.mode != 'L':
                mrcdata2 = to_int8(mrcdata2)
            if self.random_rotate_transform is not None:
                mrcdata_rotate2 = self.random_rotate_transform(mrcdata2)
            else:
                mrcdata_rotate2 = mrcdata2
            aug2 = self.transform(mrcdata_rotate2)

        else:
            if self.random_rotate_transform is not None:
                mrcdata_rotate2 = self.random_rotate_transform(mrcdata)
            else:
                mrcdata_rotate2 = mrcdata
            aug2 = self.transform(mrcdata_rotate2)

        for _ in range(self.local_crops_number):
            local_crops1.append(self.local_crops_transform(mrcdata_rotate1))
            local_crops2.append(self.local_crops_transform(mrcdata_rotate2))
        # imgs_all = [aug1, aug2] + local_crops1 + local_crops2

        img2tensor = transforms.ToTensor()
        mrcdata = img2tensor(mrcdata)

        out = {'mrcdata': mrcdata, 'aug1': aug1, 'aug2': aug2, 'label_for_clustering': label_for_clustering,
               'label_for_classification': label_for_classification, 'path': tif_path,
               'raw_path': raw_tif_path, 'particles_id': str(particles_id), 'item': item, 'local_crops1': local_crops1,
               'local_crops2': local_crops2}
        return out

    def get_transforms(self, transforms):
        self.transform = transforms[0]
        self.local_crops_transform = transforms[1]
        self.random_rotate_transform = transforms[2]


class EMDataset_from_rawdata(Dataset):
    '''Self-defined cryoEM dataset'''

    def __init__(self, mrc_dir, mrcs_names_list, max_batch_size, transform=None,
                 accelerator=None,
                 ):

        self.transform = transform
        self.accelerator = accelerator
        self.mrc_dir = mrc_dir
        self.mrcs_names_list = mrcs_names_list
        self.max_batch_size = max_batch_size

    def __len__(self):
        return len(self.mrcs_names_list)

    def __getitem__(self, item):
        mrcs_path = os.path.join(self.mrc_dir, self.mrcs_names_list[item])
        mrcs = np.float32(mrcfile.open(mrcs_path).data)
        mrcs_len = mrcs.shape[0]
        imgs_list = []
        imgs_sum = 0
        mrcs = [to_int8(mrcs[i]) for i in range(mrcs_len)]
        for i in range(mrcs_len // self.max_batch_size + 1):
            imgs_sum += self.max_batch_size
            if imgs_sum >= mrcs_len:
                # aaa=mrcs[i*self.max_batch_size:][0][0]
                # imgs=Image.fromarray(aaa)
                imgdata = torch.cat([self.transform(mrcs[i]) for i in range(i * self.max_batch_size, mrcs_len)])
            else:
                imgdata = torch.cat(
                    [self.transform(mrcs[i]) for i in range(i * self.max_batch_size, (i + 1) * self.max_batch_size)])
            imgs_list.append(imgdata)
        out = {'imgdata': imgs_list}
        return out

    def get_transforms(self, transforms):
        self.transform = transforms[0]
        self.local_crops_transform = transforms[1]
        self.random_rotate_transform = transforms[2]


def to_int8(mrcdata):
    if np.max(mrcdata) - np.min(mrcdata) != 0:
        mrcdata = (mrcdata - np.min(mrcdata)) / ((np.max(mrcdata) - np.min(mrcdata)))
        mrcdata = (mrcdata * 255).astype(np.uint8)
    else:
        mrcdata = mrcdata.astype(np.uint8)

    return Image.fromarray(mrcdata)


from torch.utils.data.sampler import Sampler


class MyResampleSampler(Sampler):
    def __init__(self, data, id_index_dict_pos, id_index_dict_neg, resample_num_pos, resample_num_neg,
                 batch_size_all=None, shuffle_type=True,dataset_id_map=None):
        self.data = data
        self.id_index_dict_pos = id_index_dict_pos
        self.id_index_dict_neg = id_index_dict_neg
        self.resample_num_pos = resample_num_pos
        self.resample_num_neg = resample_num_neg
        self.shuffle_type = shuffle_type
        self.batch_size_all = batch_size_all
        self.dataset_id_map = dataset_id_map
        self.my_seed = 0
        self.indices = resample_from_id_index_dict_finetune(self.id_index_dict_pos, self.id_index_dict_neg,
                                                            self.resample_num_pos,
                                                            self.resample_num_neg, batch_size_all=self.batch_size_all,
                                                            shuffle_type=self.shuffle_type, my_seed=self.my_seed,dataset_id_map=dataset_id_map)

    def __iter__(self):
        self.indices = resample_from_id_index_dict_finetune(self.id_index_dict_pos, self.id_index_dict_neg,
                                                            self.resample_num_pos,
                                                            self.resample_num_neg, batch_size_all=self.batch_size_all,
                                                            shuffle_type=self.shuffle_type, my_seed=self.my_seed,dataset_id_map=self.dataset_id_map)
        self.my_seed += 1
        return iter(self.indices)

    def __len__(self):
        # return len(self.data)
        return len(self.indices)


class MyResampleSampler_pretrain(Sampler):
    def __init__(self, id_index_dict, batch_size_all, max_number_per_sample=None, shuffle_type=True,
                 shuffle_mix_up_ratio=0.2, dataset_id_map=None):
        self.id_index_dict = id_index_dict
        self.batch_size_all = batch_size_all
        self.max_number_per_sample = max_number_per_sample
        self.shuffle_type = shuffle_type
        self.shuffle_mix_up_ratio = shuffle_mix_up_ratio
        self.my_seed = 0
        self.dataset_id_map = dataset_id_map
        self.indices = resample_from_id_index_dict(id_index_dict, max_number_per_sample, batch_size_all, shuffle_type,
                                                   shuffle_mix_up_ratio, self.my_seed, dataset_id_map)

    # @profile(precision=4)
    def __iter__(self):
        self.indices = resample_from_id_index_dict(self.id_index_dict, self.max_number_per_sample, self.batch_size_all,
                                                   self.shuffle_type, self.shuffle_mix_up_ratio, self.my_seed,
                                                   self.dataset_id_map)
        # print(self.indices[0:80])
        self.my_seed += 1
        # print(sorted(self.indices))
        return iter(self.indices)

    def __len__(self):
        # return len(self.data)
        return len(self.indices)


# # Optimize the resample_from_id_index_dict function
def resample_from_id_index_dict(id_index_dict, max_number_per_sample=None, batch_size_all=None, shuffle_type=None,
                                shuffle_mix_up_ratio=0.2, my_seed=0, dataset_id_map=None,positive_ratio=0.5):
    random.seed(my_seed)
    resampled_index_list = []
    final_resampled_index_list = []
    ids_list = list(id_index_dict.keys())
    mix_up_list = []

    if dataset_id_map is not None:
        id_map = dataset_id_map['id_map']
        bad_id_list = dataset_id_map['bad_id_list']
        ids_list = list(id_map.keys())
    else:
        id_map = None
        bad_id_list = []

    if shuffle_type == 'class':
        random.shuffle(ids_list)
    for my_id in ids_list:

        if dataset_id_map is not None and id_map[my_id] is not None:
            max_number_per_sample_pos = int(max_number_per_sample * positive_ratio)
            max_number_per_sample_neg = max_number_per_sample - max_number_per_sample_pos

            selected_index_list1, mix_up_list_added1 = get_index_per_class(id_index_dict, my_id,
                                                                             max_number_per_sample_pos, shuffle_type,
                                                                             shuffle_mix_up_ratio,
                                                                             is_bad_class=my_id in bad_id_list)



            selected_index_list2, mix_up_list_added2 = get_index_per_class(id_index_dict, id_map[my_id],
                                                                               max_number_per_sample_neg, shuffle_type,
                                                                               shuffle_mix_up_ratio)
            new_selected_index_list = selected_index_list1 + selected_index_list2
            # random.shuffle(new_selected_index_list)
            resampled_index_list.append(new_selected_index_list)
            mix_up_list_added = mix_up_list_added1 + mix_up_list_added2
            mix_up_list.extend(mix_up_list_added)
        else:
            max_number_per_sample_i = max_number_per_sample
            new_selected_index_list, mix_up_list_added = get_index_per_class(id_index_dict, my_id,
                                                                             max_number_per_sample_i, shuffle_type,
                                                                             shuffle_mix_up_ratio,
                                                                             is_bad_class=my_id in bad_id_list)
            if len(new_selected_index_list) > 0:
                resampled_index_list.append(new_selected_index_list)
            mix_up_list.extend(mix_up_list_added)

    if shuffle_type == 'batch':
        random.shuffle(mix_up_list)
        step = len(mix_up_list) // len(resampled_index_list)
        for i in range(len(resampled_index_list)):
            # step=batch_size_all-len(resampled_index_list[i])
            resampled_index_list[i].extend(mix_up_list[:step])
            random.shuffle(resampled_index_list[i])
            new_resampled_index_list_i = []
            for ii in range(len(resampled_index_list[i]) // batch_size_all + 1):
                if len(resampled_index_list[i]) >= batch_size_all:
                    new_resampled_index_list_i.append(resampled_index_list[i][:batch_size_all])
                    resampled_index_list[i] = resampled_index_list[i][batch_size_all:]
            final_resampled_index_list.extend(new_resampled_index_list_i)
            mix_up_list = mix_up_list[step:]
    if shuffle_type == 'class':
        random.shuffle(mix_up_list)
        step = len(mix_up_list) // len(resampled_index_list)
        for i in range(len(resampled_index_list)):
            # step=batch_size_all-len(resampled_index_list[i])
            resampled_index_list[i].extend(mix_up_list[:step])
            random.shuffle(resampled_index_list[i])
            mix_up_list = mix_up_list[step:]

    if shuffle_type == 'batch':
        random.shuffle(final_resampled_index_list)
        final_resampled_index_list = [item for sublist in final_resampled_index_list for item in sublist]
    else:
        final_resampled_index_list = [item for sublist in resampled_index_list for item in sublist]
        if shuffle_type == 'all':
            random.shuffle(final_resampled_index_list)
    return final_resampled_index_list


def get_index_per_class(id_index_dict, my_id, max_number_per_sample=None, shuffle_type=None, shuffle_mix_up_ratio=0.2,
                        is_bad_class=False):
    index_list = id_index_dict[my_id]
    len_index_list = len(index_list)
    if max_number_per_sample is not None and len_index_list > max_number_per_sample:
        selected_index_list = random.sample(index_list, max_number_per_sample)

        if shuffle_type == 'batch' or shuffle_type == 'class':
            if is_bad_class:
                mix_up_list_added = selected_index_list[:int(len(selected_index_list) * shuffle_mix_up_ratio)]
                new_selected_index_list = []
            else:
                new_selected_index_list = selected_index_list[
                                          :max_number_per_sample - int(len(selected_index_list) * shuffle_mix_up_ratio)]
                mix_up_list_added = (
                    selected_index_list[max_number_per_sample - int(len(selected_index_list) * shuffle_mix_up_ratio):])
        else:
            if is_bad_class:
                mix_up_list_added = selected_index_list
                new_selected_index_list = []
            else:
                new_selected_index_list = selected_index_list
                mix_up_list_added = []
    else:
        if is_bad_class:
            mix_up_list_added =index_list[:int(len(index_list) * shuffle_mix_up_ratio)]
            new_selected_index_list = []
        elif shuffle_type == 'batch' or shuffle_type == 'class':
            random.shuffle(index_list)
            new_selected_index_list = index_list[:len_index_list - int(len(index_list) * shuffle_mix_up_ratio)]
            mix_up_list_added = (index_list[len_index_list - int(len(index_list) * shuffle_mix_up_ratio):])
        else:
            new_selected_index_list = index_list
            mix_up_list_added = []
    return new_selected_index_list, mix_up_list_added


def resample_from_id_index_dict_finetune(id_index_dict_pos, id_index_dict_neg, resample_num_p, resample_num_n,
                                         batch_size_all=None, shuffle_type=None, shuffle_mix_up_ratio=0.2, my_seed=0,dataset_id_map=None):
    random.seed(my_seed)
    if shuffle_type== 'batch':
        id_index_dict_all={**id_index_dict_pos, **id_index_dict_neg}
        resampled_index_list =  resample_from_id_index_dict(id_index_dict_all, resample_num_p+resample_num_n, batch_size_all, shuffle_type,
                                        shuffle_mix_up_ratio, my_seed,dataset_id_map=dataset_id_map,positive_ratio=resample_num_p/(resample_num_p+resample_num_n))

    else:
        resampled_index_list = []
        resampled_index_list.extend(
            resample_from_id_index_dict(id_index_dict_pos, resample_num_p, batch_size_all, shuffle_type,
                                        shuffle_mix_up_ratio, my_seed))
        resampled_index_list.extend(
            resample_from_id_index_dict(id_index_dict_neg, resample_num_n, batch_size_all, shuffle_type,
                                        shuffle_mix_up_ratio, my_seed))
        if shuffle_type == 'all':
            random.shuffle(resampled_index_list)
    return resampled_index_list
