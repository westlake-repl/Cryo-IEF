import pandas as pd
import os
from .generate_cs_file import generate_cs_file
from .cs2star import cs2star
from cryosparc.dataset import Dataset

def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file)[-1] == '.csv':
            list_name.append(file_path)


def read_record_data(path,target_mrcs_path= None):
    path_list = []
    blob_id_list = []
    record_data_path_list = []
    defocus_1_list = []
    defocus_2_list = []
    pixelsize_list = []
    size_list=[]
    listdir(path, record_data_path_list)
    for record_data_path in record_data_path_list:
        record_data = pd.read_csv(record_data_path)
        path_list.extend(record_data['path'].tolist())
        blob_id_list.extend(record_data.index.tolist())
        defocus_1_list.extend(record_data['defocus_1'].tolist())
        defocus_2_list.extend(record_data['defocus_2'].tolist())
        pixelsize_list.extend(record_data['pixelsize'].tolist())
        size_list.extend(record_data['image_size'].tolist())
    size_list=[[x,x] for x in size_list]
    if target_mrcs_path is not None:
        path_list=[os.path.join(target_mrcs_path,x.split('/')[-1]) for x in path_list]
    return {'path': path_list, 'blob_id': blob_id_list, 'defocus_1': defocus_1_list, 'defocus_2': defocus_2_list,
            'pixelsize': pixelsize_list,'image_size':size_list}


def write_record_data(path, record_data):
    cs_data = generate_cs_file(blob_path=record_data['path'], blob_id=record_data['blob_id'],
                               psize=record_data['pixelsize'], df1=record_data['defocus_1'],
                               df2=record_data['defocus_2'], image_size=record_data['image_size'])
    cs_data.save(path+'/simluated_particles.cs')
    cs2star(path+'/simluated_particles.cs', path+'/simluated_particles.star')

