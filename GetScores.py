#!/usr/bin/env python

import cryosparc.dataset as csd
import datetime
import shutil
import os

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

import CryoRanker_inference
import MyLib.CSLogin as CSLogin
import MyLib.MyJobAPIs as MyJobAPIs
import MyLib.mytoolbox as mytoolbox



accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=96000))])

if accelerator.is_main_process:
    wholetimebegin = datetime.datetime.now()

# 初始化需要的内容
globaldir = os.getcwd()
if not (globaldir[-1] == '/'):
    globaldir = globaldir + '/'
parameters = mytoolbox.readjson(globaldir + 'parameters/parameters.json')
if_initial_orientation_balance = parameters['initial_orientation_balance']
if_delete_cache = parameters['delete_cache']
cshandle = CSLogin.cshandleclass.GetCryoSPARCHandle(email=parameters['cryosparc_username'], password=parameters['cryosparc_password'])
dealjobs = MyJobAPIs.DealJobs(cshandle, parameters['project'], parameters['workspace'], parameters['lane'])
model_cache_file_path = globaldir + 'metadata/model_cache'
if accelerator.is_main_process:
    if not os.path.exists(model_cache_file_path):
        os.makedirs(model_cache_file_path)

# 识别输入的particle job的类别，支持的particle job有：Import Particle Stack, Extract From Micrographs(Multi-GPU), Restack Particles
source_particle_job = mytoolbox.readjson(globaldir + 'metadata/particle_job_uid.json')
files_in_job_dir = os.listdir((str)(dealjobs.cshandle.find_project(dealjobs.project).dir()) + '/' + source_particle_job)
if 'imported_particles.cs' in files_in_job_dir:
    source_particles_cs = csd.Dataset.load((str)(dealjobs.cshandle.find_project(dealjobs.project).dir()) + '/' + source_particle_job + '/imported_particles.cs')
elif 'extracted_particles.cs' in files_in_job_dir:
    source_particles_cs = csd.Dataset.load((str)(dealjobs.cshandle.find_project(dealjobs.project).dir()) + '/' + source_particle_job + '/extracted_particles.cs')
elif 'restacked_particles.cs' in files_in_job_dir:
    source_particles_cs = csd.Dataset.load((str)(dealjobs.cshandle.find_project(dealjobs.project).dir()) + '/' + source_particle_job + '/restacked_particles.cs')
elif 'split_0000.cs' in files_in_job_dir:
    source_particles_cs = csd.Dataset.load((str)(dealjobs.cshandle.find_project(dealjobs.project).dir()) + '/' + source_particle_job + '/split_0000.cs')
else:
    accelerator.print('Job type error...', flush=True)
    exit()

# '''This part is for testing'''
# source_particles_cs = csd.Dataset.load('/yanyang2/dataset/classification/valset_real/galactosidase/raw_cs/J110/imported_particles.cs')
# job_path='/yanyang2/dataset/classification/valset_real/galactosidase/raw_cs/J110/'
# cache_path='/yanyang2/projects/results/particle_classification_inference/2024_6_19_set_test2/'


# 使用用户指定的particle job，载入大模型生成对应的confidence_dict
# confidence_dict: 大模型对输入job的每个particle生成的置信度，字典类型，key为cs中每个particle的uid，value为置信度
accelerator.print('Creating confidence info from model...', flush=True)
new_particles_cs, confidence, features_all = CryoRanker_inference.cryo_select_main(job_path=(str)(dealjobs.cshandle.find_project(dealjobs.project).dir())+'/'+source_particle_job+'/', cache_file_path=model_cache_file_path, accelerator=accelerator, use_features=if_initial_orientation_balance, features_max_num=len(source_particles_cs))

if accelerator.is_main_process:

    new_info = [[] for _ in range(len(new_particles_cs))]
    for i in range(len(new_particles_cs)):
        new_mrcfile = new_particles_cs[i]['blob/path']
        new_idx = new_particles_cs[i]['blob/idx']
        new_info[i] = [new_mrcfile + (str)(new_idx), i]
    new_info = sorted(new_info, key=lambda x: x[0])
    source_info = [[] for _ in range(len(new_particles_cs))]
    for i in range(len(new_particles_cs)):
        source_mrcfile = source_particles_cs[i]['blob/path']
        source_idx = source_particles_cs[i]['blob/idx']
        source_uid = source_particles_cs[i]['uid']
        source_info[i] = [source_mrcfile + (str)(source_idx), (int)(source_uid)]
    source_info = sorted(source_info, key=lambda x: x[0])
    confidence_dict = {}
    features_dict = {}
    for i in range(len(new_particles_cs)):
        if (new_info[i][0] == source_info[i][0]):
            if if_initial_orientation_balance:
                if new_info[i][1] in features_all:
                    features_dict[source_info[i][1]] = features_all[new_info[i][1]]
            confidence_dict[source_info[i][1]] = (float)(confidence[new_info[i][1]])
        else:
            accelerator.print('Error! Source blob info does not match new blob info...', flush=True)
            exit()

    if os.path.exists(globaldir + 'metadata/confidence_dict.json'):
        os.remove(globaldir + 'metadata/confidence_dict.json')
    mytoolbox.savetojson(confidence_dict, globaldir + 'metadata/confidence_dict.json', False)
    if if_initial_orientation_balance:
        if os.path.exists(globaldir + 'metadata/features_dict.pickle'):
            os.remove(globaldir + 'metadata/features_dict.pickle')
        mytoolbox.savebypickle(features_dict, globaldir + 'metadata/features_dict.pickle', False)

    if if_delete_cache:
        shutil.rmtree(model_cache_file_path)

    wholetimeend = datetime.datetime.now()
    accelerator.print('Creating confidence info completed! Time spent:', wholetimeend - wholetimebegin, flush=True)


