import os
import pickle
import numpy as np
import lmdb
import multiprocessing
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import torch
from .mrc_preprocess import mrcs_resize, mrcs_to_int8, window_mask
from . import fft
from . import mrc


def create_lmdb_dataset(image_path_list, lmdb_path,  map_size=1024 ** 4, num_processes=None,
                        chunksize=0, resize=224, is_to_int8=True, window=True, window_r=0.85, generate_ft_data=False,
                        save_raw_data=False):
    """
    并行化创建LMDB数据库。
    处理process_item返回的包含多个条目的列表，并在主进程中按序分配全局key。
    """

    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    path_id_data_list = []
    # 初始化LMDB环境
    env = lmdb.open(lmdb_path, map_size=map_size, readonly=False, create=True, max_readers=128)
    protein_id_dict = {}
    protein_id_list = []

    if num_processes is None:
        num_processes = 16
    if chunksize == 0:
        chunksize = 1

    with multiprocessing.Pool(processes=num_processes) as pool:

        protein_id = 0
        # 生成任务列表（索引+路径）
        tasks = [(idx, data_path, resize, is_to_int8, window, window_r, generate_ft_data, save_raw_data) for
                 idx, data_path in enumerate(image_path_list)]

        print(f"Starting parallel processing of {len(tasks)} files...")
        # 使用imap保证结果按任务顺序返回
        results_iterator = pool.imap(lmdb_process_item, tasks, chunksize=chunksize)

        # 使用事务进行写入
        with env.begin(write=True) as txn:
            global_item_index = 0
            # 创建两个进度条：处理文件和写入项目
            with tqdm(total=len(tasks), desc="Processing files") as pbar_files:
                # tqdm(desc="Writing items to LMDB") as pbar_items:
                for original_idx, path_id_data, item_list in results_iterator:
                    if item_list:  # 如果处理成功且有数据
                        for serialized_data in item_list:
                            key = f"{global_item_index}".encode()
                            txn.put(key, serialized_data)
                            global_item_index += 1
                            # pbar_items.update(1)  # 更新项目进度条
                    path_id_data_list.extend(path_id_data)
                    protein_name = tasks[original_idx][1].split('/')[-3]
                    if protein_name not in protein_id_dict:
                        protein_id_dict[protein_name] = protein_id
                        protein_id += 1
                    protein_id_list.extend([protein_id_dict[protein_name] for _ in range(len(path_id_data))])
                    pbar_files.update(1)  # 更新文件进度条

    env.close()
    save_data_path = os.path.dirname(lmdb_path)
    pickle.dump(protein_id_dict, open(os.path.join(save_data_path, 'protein_id_dict.data'), 'wb'))
    pickle.dump(protein_id_list, open(os.path.join(save_data_path, 'protein_id_list.data'), 'wb'))


    print("LMDB dataset creation finished.")



def lmdb_process_item(args):
    """
    处理单个pickle文件的加载、堆栈分割和序列化。
    支持 pil_image/np_image_raw/np_image_FT 为 [n, d, d] 形状的 images stack。
    不在此函数中生成最终的LMDB key。
    """
    idx, data_path, resize, is_to_int8, window, window_r, generate_ft_data, save_raw_data = args
    path_id_my_data = []
    processed_items_data = []  # 存储序列化后的单个图像数据
    data = {}
    try:
        pil_image = None
        np_image_raw = None
        np_image_FT = None
        n_total = 0  # Initialize to 0

        if data_path.endswith('.data'):
            # Load data from pickle files
            # Load pil_image (processed image)
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    pil_image = pickle.load(f)
                    if isinstance(pil_image, np.ndarray) and pil_image.ndim >= 2:
                        # Assuming pil_image loaded from .data is the primary processed data
                        processed_mrcs = pil_image
                        if processed_mrcs.ndim == 3:
                            n_total = processed_mrcs.shape[0]
                        elif processed_mrcs.ndim <= 2:  # Handle single image case
                            n_total = 1
                        else:
                            print(
                                f"Warning: Unexpected dimensions for data at {data_path}: {processed_mrcs.ndim}. Skipping.")
                            return (idx, [])  # Return empty list
                    elif isinstance(pil_image, Image.Image):
                        processed_mrcs = np.array(pil_image)
                        n_total = 1
                    else:
                        print(f"Warning: Unexpected data type in {data_path}: {type(pil_image)}. Skipping.")
                        return (idx, [])  # Return empty list


            else:
                print(f"Warning: Processed image file not found at {data_path}. Skipping.")
                return (idx, [])  # Return empty list if the primary file is missing

            # Construct paths for raw and FT data based on assumed directory structure
            pkl_path_parts = data_path.split('/')
            # Ensure path has enough parts to modify the directory
            if len(pkl_path_parts) > 3:
                # Assuming data_path is like .../processed/dataset_name/file_name.data
                # Raw/FT paths would be .../raw/dataset_name/file_name.data
                pkl_path_raw_parts = pkl_path_parts[:-4] + ['raw'] + pkl_path_parts[-3:]
                pkl_path_FT_parts = pkl_path_parts[:-4] + ['FT'] + pkl_path_parts[-3:]

                pkl_path_raw = '/'.join(pkl_path_raw_parts)
                pkl_path_FT = '/'.join(pkl_path_FT_parts)

                # Load raw image data
                if os.path.exists(pkl_path_raw):
                    with open(pkl_path_raw, 'rb') as f:
                        np_image_raw = pickle.load(f)

                # Load FT image data
                if os.path.exists(pkl_path_FT) and generate_ft_data:
                    with open(pkl_path_FT, 'rb') as f:
                        np_image_FT = pickle.load(f)
            else:
                print(f"Warning: Could not construct raw/FT paths for {data_path}. Assuming no raw/FT data.")


        else:  # Assuming MRC file
            try:
                # mrc_data = mrcfile.open(data_path).data
                mrc_data, _ = mrc.parse_mrc(data_path)
                np_image_raw = np.float32(mrc_data)
                imgs_len = np_image_raw.shape[0]  # Assuming [n, d, d]
                n_total = imgs_len  # Number of items is the number of images in stack

                processed_mrcs = np_image_raw.copy()
                if np_image_raw.shape[1] != resize:
                    processed_mrcs = mrcs_resize(processed_mrcs, resize, resize)

                if is_to_int8:
                    processed_mrcs = mrcs_to_int8(processed_mrcs)
                    # processed_mrcs = to_int8(processed_mrcs)

                if window:
                    np_image_raw *= window_mask(np_image_raw.shape[-1], window_r, .99)

                if generate_ft_data:
                    particles = []
                    for i in range(imgs_len):
                        img = np_image_raw[i]
                        particles.append(fft.ht2_center(img))

                    np_image_FT = np.asarray(particles, dtype=np.float32)
                    np_image_FT = fft.symmetrize_ht(np_image_FT)
            except Exception as mrc_e:
                print(f"Error processing MRC file {data_path}: {str(mrc_e)}. Skipping.")
                return (idx, [])

        # --- Process each image in the potential stack ---
        # If n_total is 0 after processing, return empty list
        if n_total == 0:
            print(f"Warning: No valid data found in {data_path}. Skipping.")
            return (idx, [])

        for i in range(n_total):
            current_pil = None
            current_raw = None
            current_FT = None

            path_id_my_data.append(os.path.join(data_path.split('/')[-1], str(i + 1).zfill(6)))

            # Extract processed image slice
            if processed_mrcs is not None:
                if processed_mrcs.ndim == 3 and i < n_total:
                    current_pil = Image.fromarray(processed_mrcs[i]).convert('L')
                elif processed_mrcs.ndim <= 2 and n_total == 1:  # Handle single image case
                    current_pil = Image.fromarray(processed_mrcs).convert('L')
                # else: index out of bounds or wrong dim
                data['image_processed'] = current_pil

            # Extract slice for raw data if it's a stack and index is valid
            if save_raw_data and isinstance(np_image_raw, np.ndarray):
                if np_image_raw.ndim == 3 and i < n_total:
                    current_raw = np_image_raw[i]
                elif np_image_raw.ndim <= 2 and n_total == 1:  # Handles single image case
                    current_raw = np_image_raw
                # else: index out of bounds for this stack type or wrong dim, current_raw remains None
                data['image_raw'] = current_raw

            # Extract slice for FT data if it's a stack and index is valid
            if generate_ft_data and isinstance(np_image_FT, np.ndarray):
                if np_image_FT.ndim == 3 and i < n_total:
                    current_FT = np_image_FT[i]
                elif np_image_FT.ndim <= 2 and n_total == 1:  # Handles single image case
                    current_FT = np_image_FT
                # else: index out of bounds for this stack type or wrong dim, current_FT remains None
                data['image_FT'] = current_FT

            if current_pil is not None or current_raw is not None or current_FT is not None:
                # data = {'image_processed': current_pil, 'image_raw': current_raw, 'image_FT': current_FT}
                buffer = BytesIO()
                torch.save(data, buffer)
                serialized_data = buffer.getvalue()
                # Append only the serialized data value
                processed_items_data.append(serialized_data)
            else:
                print(f"Warning: No data found for slice {i} in file {data_path}. Skipping this slice.")

        # Return the original file index and the list of serialized item data
        return (idx, path_id_my_data, processed_items_data)

    except Exception as e:
        print(f"Error processing {data_path}: {str(e)}")
        # Return original index and empty list to indicate failure for the whole file
        return (idx, [])


