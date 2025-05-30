import numpy as np
from PIL import Image
import torch
import os
import pickle
import random
from cryosparc.dataset import Dataset
from functools import partial
import multiprocessing
from tqdm import tqdm
from . import mrc
from . import fft



def sample_and_evaluate(path_list,save_path,num_stacks=5,num_particles=2000):
    if num_stacks> len(path_list):
        num_stacks=len(path_list)
    path_sampled=random.sample(path_list, num_stacks)
    np_image_raw_all=[]
    np_image_FT_all=[]
    imgs_len_all=[]

    for i in range(num_stacks):
        arr,_= mrc.parse_mrc(path_sampled[i])
        # np_image_raw = np.float32(mrcfile.open(path_sampled[i]).data)
        np_image_raw = np.float32(arr)
        imgs_len = np_image_raw.shape[0]


        particles = []
        for i in range(imgs_len):
            img = np_image_raw[i]
            particles.append(fft.ht2_center(img))

        np_image_FT = np.asarray(particles, dtype=np.float32)
        np_image_FT = fft.symmetrize_ht(np_image_FT)

        np_image_raw_all.append(np_image_raw)
        np_image_FT_all.append(np_image_FT)
        imgs_len_all.append(imgs_len)
    np_image_raw_all=np.concatenate(np_image_raw_all,axis=0)
    np_image_FT_all=np.concatenate(np_image_FT_all,axis=0)
    if num_particles> len(np_image_raw_all):
        num_particles=len(np_image_raw_all)
    index = np.random.choice(np.arange(len(np_image_raw_all)), size=num_particles, replace=False)
    np_image_raw_all_sampled=np_image_raw_all[index]
    np_image_FT_all_sampled=np_image_FT_all[index]
    means_raw=np.mean(np_image_raw_all_sampled)
    std_raw=np.std(np_image_raw_all_sampled)
    # means_FT=np.mean(np_image_FT_all_sampled)
    std_FT=np.std(np_image_FT_all_sampled)
    mean_imgs_len=np.mean(imgs_len_all)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'means_stds_raw.data'), 'wb') as filehandle:
        pickle.dump((means_raw, std_raw), filehandle)
    with open(os.path.join(save_path, 'means_stds_FT.data'), 'wb') as filehandle:
        pickle.dump((0.0, std_FT), filehandle)
    img_dim=np_image_raw_all.shape[-1]
    with open(os.path.join(save_path, 'img_dim.data'), 'wb') as filehandle:
        pickle.dump(img_dim, filehandle)
    # return (means_raw, std_raw), (means_FT, std_FT), mean_imgs_len
    return mean_imgs_len


def get_mean_std(path_list, Cnum=10000,is_normalize=True):
    # calculate means and stds
    imgs = []
    random.shuffle(path_list)
    if len(path_list) < Cnum:
        Cnum = len(path_list)
    sample_path_list = random.sample(path_list, Cnum)
    for i in range(Cnum):
        # img = np.array(Image.open(path_list[i]))
        img = pickle.load(open(sample_path_list[i], 'rb'))
        if isinstance(img,np.ndarray):
            min_val= np.min(img)
            max_val= np.max(img)
        else:
            min_val, max_val = img.getextrema()
            img = np.array(img)
        if max_val - min_val != 0 and is_normalize:
            img = (img - min_val) / (max_val - min_val)
        img = img.astype(np.float32)
        imgs.append(img)
    imgs_np = np.asarray(imgs)
    means = imgs_np.mean()
    stds = imgs_np.std()
    return means, stds


def sample_and_calculate_mean_std(path_list, Cnum=1000, Ctimes=1,is_normalize=True):
    mean_list = []
    std_list = []
    Cnum_all = 0
    for i in range(Ctimes):
        if Cnum_all >= len(path_list):
            break
        means, stds = get_mean_std(path_list, Cnum,is_normalize=is_normalize)
        mean_list.append(means)
        std_list.append(stds)
        Cnum_all += Cnum
    mean = np.mean(mean_list)
    std = np.mean(std_list)
    return mean, std

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

def mrcs_to_int8(mrcs):
    if torch.is_tensor(mrcs):
        new_mrcs = torch.zeros_like(mrcs, dtype=torch.uint8)
    else:
        new_mrcs = np.zeros_like(mrcs, dtype=np.uint8)
    for i in range(mrcs.shape[0]):
        new_mrcs[i] = to_int8(mrcs[i])
    return new_mrcs

# def to_int8(mrcdata):
#     # mrcdata = np.array(mrcdata)
#     # a=np.max(mrcdata)
#     # b=np.min(mrcdata)
#     if torch.is_tensor(mrcdata):
#         mrcdata = mrcdata.cpu().numpy()
#     if np.max(mrcdata) - np.min(mrcdata) != 0:
#         mrcdata_processed = (mrcdata - np.min(mrcdata)) / ((np.max(mrcdata) - np.min(mrcdata)))
#         mrcdata_processed = (mrcdata_processed * 255).astype(np.uint8)
#     else:
#         mrcdata_processed = mrcdata.astype(np.uint8)
#
#     return Image.fromarray(mrcdata_processed)
    # return mrcdata

def to_int8(mrcdata):
    if torch.is_tensor(mrcdata):
        mrcdata = (mrcdata - torch.min(mrcdata)) / (torch.max(mrcdata) - torch.min(mrcdata))
        mrcdata = (mrcdata * 255).type(torch.uint8)
    else:
        if np.max(mrcdata) - np.min(mrcdata) != 0:
            mrcdata = (mrcdata - np.min(mrcdata)) / ((np.max(mrcdata) - np.min(mrcdata)))
            mrcdata= (mrcdata * 255).astype(np.uint8)
        else:
            mrcdata = mrcdata.astype(np.uint8)
        mrcdata = Image.fromarray(mrcdata)
    return mrcdata

    # return Image.fromarray(mrcdata_processed)
    # return mrcdata

def window_mask(resolution, in_rad, out_rad=.99):
    assert resolution % 2 == 0
    x0, x1 = np.meshgrid(np.linspace(-1, 1, resolution, endpoint=False, dtype=np.float32),
                         np.linspace(-1, 1, resolution, endpoint=False, dtype=np.float32))
    r = (x0 ** 2 + x1 ** 2) ** .5
    mask = np.minimum(1.0, np.maximum(0.0, 1 - (r - in_rad) / (out_rad - in_rad)))
    return mask

def raw_csdata_process_from_cryosparc_dir(raw_data_path):
    if raw_data_path.endswith('cs'):
        raw_data_path = os.path.dirname(raw_data_path)
    new_csdata_path = os.path.join(raw_data_path, 'new_particles.cs')
    mrc_dir = raw_data_path

    # if not os.path.exists(new_csdata_path):
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

        if filename.endswith('downsampled_particles.cs'):

            particles_cs_path = os.path.join(raw_data_path, filename)
    if not os.path.exists(new_csdata_path):
        if passthrough_particles_path is not None:
            cs_data = combine_cs_files_column(particles_cs_path, passthrough_particles_path)
        elif particles_cs_path is not None:
            cs_data = Dataset.load(particles_cs_path)
        else:
            Exception(raw_data_path+': corresponding  not exists!')

        cs_data.save(new_csdata_path)
    else:
        cs_data = Dataset.load(new_csdata_path)

    if os.path.exists(os.path.join(raw_data_path, 'restack')):
        mrc_dir = os.path.join(raw_data_path, 'restack')
    elif os.path.exists(os.path.join(raw_data_path, 'extract')):
        mrc_dir = os.path.join(raw_data_path, 'extract')
    elif os.path.exists(os.path.join(raw_data_path, 'imported')):
        mrc_dir = os.path.join(raw_data_path, 'imported')
    elif os.path.exists(os.path.join(raw_data_path, 'downsample')):
        mrc_dir = os.path.join(raw_data_path, 'downsample')
    elif particles_cs_path.endswith('split_0000.cs'):
        raw_dir='/'.join(raw_data_path.split('/')[0:-2])
        mrc_dir = raw_dir+'/'+'/'.join(cs_data['blob/path'][0].split('/')[0:-1])+'/'
    elif particles_cs_path.endswith('downsampled_particles.cs'):
        raw_dir='/'.join(raw_data_path.split('/')[0:-1])
        mrc_dir=raw_dir
        # mrc_dir = raw_dir+'/'+'/'.join(cs_data['blob/path'][0].split('/')[0:-1])+'/'
    return cs_data,mrc_dir

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

def raw_data_preprocess_one_mrcs(name, mrc_dir, raw_dataset_save_dir, processed_dataset_save_dir, FT_dataset_save_dir,resize=224,
                                 is_to_int8=True,indeices_per_mrcs_dict=None):
    mrcs_path = os.path.join(mrc_dir, name)
    arr,_= mrc.parse_mrc(mrcs_path)
    # mrcs = np.float32(mrcfile.open(mrcs_path).data)
    mrcs = np.float32(arr)
    mrcs_len = mrcs.shape[0]
    raw_single_particle_path = []
    processed_single_particle_path = []
    FT_single_particle_path=[]
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
        # processed_mrcs = to_int8(processed_mrcs)

    if FT_dataset_save_dir is not None:
        particles = []
        for i in range(mrcs_len):
            img= mrcs[i]
            particles.append(fft.ht2_center(img))

        FT_mrcs = np.asarray(particles, dtype=np.float32)
        FT_mrcs = fft.symmetrize_ht(FT_mrcs)

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
                # pickle.dump(Image.fromarray(mrcs[j]), filehandle)
                pickle.dump(mrcs[j], filehandle)
            raw_single_particle_path.append(os.path.join(raw_dataset_save_dir, name, n + '.data'))
        else:
            raw_single_particle_path.append('')

        if FT_dataset_save_dir is not None:
            if not os.path.exists(os.path.join(FT_dataset_save_dir, name)):
                os.makedirs(os.path.join(FT_dataset_save_dir, name))
            with open(os.path.join(FT_dataset_save_dir, name, n + '.data'), 'wb') as filehandle:
                # pickle.dump(Image.fromarray(mrcs[j]), filehandle)
                pickle.dump(FT_mrcs[j], filehandle)
            FT_single_particle_path.append(os.path.join(FT_dataset_save_dir, name, n + '.data'))
        else:
            FT_single_particle_path.append('')

    return raw_single_particle_path, processed_single_particle_path,FT_single_particle_path

def raw_data_preprocess(raw_dataset_dir, dataset_save_dir, resize=224, is_to_int8=True,save_raw_data=True,save_FT_data=True,use_lmdb=True):

    if not os.path.exists(dataset_save_dir):
        os.makedirs(dataset_save_dir)

    cs_data, mrc_dir = raw_csdata_process_from_cryosparc_dir(raw_dataset_dir)


    if cs_data is not None:
        blob_path_list = cs_data['blob/path'].tolist()
        mrcs_names_list = [blob_path_list[i].split('/')[-1] for i in
                           range(len(blob_path_list))]
        # mrc_dir= os.path.dirname(mrc_list)
        mrcs_names_list_process = list(dict.fromkeys(mrcs_names_list))
        # mrcs_names_list=mrc_list

        print("Processing cs_data...")
        mrcs_names_np = np.array(mrcs_names_list)
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


        func_append_data=partial(append_data,cs_data=cs_data,indices_dict=indices_dict)
        with multiprocessing.Pool(processes=8) as pool:
            results = pool.map(func_append_data, mrcs_names_list_process)
        new_cs_data = Dataset.append(results[0], *results[1:])




        new_csdata_path = os.path.join(dataset_save_dir, 'new_particles.cs')
        new_cs_data.save(new_csdata_path)
    else:
        indeices_per_mrcs_dict=None
        # mrcs_names_list_process=mrc_list
        new_cs_data=None

    if use_lmdb:
        tmp_data_lmdb_path = dataset_save_dir + '/lmdb_data'
        tmp_data_save_path= dataset_save_dir
        if not os.path.exists(tmp_data_lmdb_path):
            from Cryoemdata.data_preprocess.lmdb_preprocess import create_lmdb_dataset

            image_path_list = [os.path.join(mrc_dir, mrcs_name) for mrcs_name in mrcs_names_list_process]

            mean_len = sample_and_evaluate(image_path_list, tmp_data_save_path)

            # map_size = int(80 * 1024 * len(image_path_list) * 6)
            map_size = int(80 * 1024 * len(image_path_list) * mean_len * 4)
            # 创建 LMDB 数据库
            create_lmdb_dataset(image_path_list, tmp_data_lmdb_path, num_processes=8, chunksize=0,
                                map_size=map_size, window=False,  generate_ft_data=False,
                                save_raw_data=False)

    else:
        particles_dir_name = raw_dataset_dir.rstrip('/').split('/')[-1]

        if save_raw_data:
            raw_dataset_save_dir = os.path.join(dataset_save_dir, 'raw', particles_dir_name)
            if not os.path.exists(raw_dataset_save_dir):
                os.makedirs(raw_dataset_save_dir)
        else:
            raw_dataset_save_dir = None

        if save_FT_data:
            FT_dataset_save_dir = os.path.join(dataset_save_dir, 'FT', particles_dir_name)
            if not os.path.exists(FT_dataset_save_dir):
                os.makedirs(FT_dataset_save_dir)
        else:
            FT_dataset_save_dir = None

        processed_dataset_save_dir = os.path.join(dataset_save_dir, 'processed', particles_dir_name)
        if not os.path.exists(processed_dataset_save_dir):
            os.makedirs(processed_dataset_save_dir)
        raw_path_list = []
        processed_path_list = []
        FT_path_list = []


        phbar = tqdm(mrcs_names_list_process, desc='data preprocessing')
        func = partial(raw_data_preprocess_one_mrcs, mrc_dir=mrc_dir, raw_dataset_save_dir=raw_dataset_save_dir,FT_dataset_save_dir=FT_dataset_save_dir,
                       processed_dataset_save_dir=processed_dataset_save_dir, resize=resize, is_to_int8=is_to_int8,indeices_per_mrcs_dict=indeices_per_mrcs_dict)
        pool = multiprocessing.Pool(20)
        results = pool.map(func, phbar)
        pool.close()
        pool.join()

        for raw_single_particle_path, processed_single_particle_path,FT_single_particle_path in results:
            processed_path_list += processed_single_particle_path
            raw_path_list += raw_single_particle_path
            FT_path_list += FT_single_particle_path



        with open(os.path.join(dataset_save_dir, 'output_processed_tif_path.data'), 'wb') as filehandle:
            pickle.dump(processed_path_list, filehandle)

        with open(os.path.join(dataset_save_dir, 'output_tif_path.data'), 'wb') as filehandle:
            pickle.dump(raw_path_list, filehandle)

        mean_std_raw = sample_and_calculate_mean_std(raw_path_list,is_normalize=False)
        with open(dataset_save_dir + 'means_stds.data', 'wb') as filehandle:
            pickle.dump(mean_std_raw, filehandle)

        if FT_dataset_save_dir is not None:
            mean_std_FT = sample_and_calculate_mean_std(FT_path_list,is_normalize=False)
            with open(dataset_save_dir + 'means_stds_FT.data', 'wb') as filehandle:
                pickle.dump(mean_std_FT, filehandle)

    print('raw data process all done')
    return new_cs_data
def append_data(name,cs_data,indices_dict):
    # mm=np.sort(indices_dict[name])
    return cs_data.take(indices_dict[name])


