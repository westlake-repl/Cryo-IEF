#!/usr/bin/env python

import numpy as np
import math
import sys
import os

import CryoRanker_inference
import MyLib.CSLogin as CSLogin
import MyLib.MyJobAPIs as MyJobAPIs
import MyLib.mytoolbox as mytoolbox





globaldir = os.getcwd()
if not (globaldir[-1] == '/'):
    globaldir = globaldir + '/'
parameters = mytoolbox.readjson(globaldir + 'parameters/parameters.json')
cshandle = CSLogin.cshandleclass.GetCryoSPARCHandle(email=parameters['cryosparc_username'], password=parameters['cryosparc_password'])
dealjobs = MyJobAPIs.DealJobs(cshandle, parameters['project'], parameters['workspace'], parameters['lane'])



def ClusterParticlesWithQuantity(particle_number, DealJobs_instance=dealjobs, particle_num_multiple_for_cluster=parameters['particle_num_multiple_for_cluster'], k=parameters['k']):
    globaldir = os.getcwd()
    if not (globaldir[-1] == '/'):
        globaldir = globaldir + '/'
    if (particle_number <= 0):
        print('Target particle number is illegal...', flush=True)
        return None, None

    # 判断是否存在get confidence job以及对应的confidence_dict文件
    if not os.path.exists(globaldir + 'metadata/external_get_confidence_jobuid.json'):
        print('external_get_confidence_jobuid.json do not exists...', flush=True)
        return None, None
    if not os.path.exists(globaldir + 'metadata/confidence_dict.json'):
        print('confidence_dict.json do not exists...', flush=True)
        return None, None
    source_particle_job = mytoolbox.readjson(globaldir + 'metadata/external_get_confidence_jobuid.json')['external_get_confidence_jobuid']
    confidence_dict = mytoolbox.readjson(globaldir + 'metadata/confidence_dict.json')
    features_dict = mytoolbox.readpicklefile(globaldir + 'metadata/features_dict.pickle')

    # 创建空白 external job
    project = DealJobs_instance.cshandle.find_project(DealJobs_instance.project)
    job = project.create_external_job(DealJobs_instance.workspace)

    # 创建和链接输入
    add_input_name = 'input_particles'
    job.add_input(type='particle', name=add_input_name, min=1, slots=['blob', 'ctf'], title='Input particles for selection')
    job.connect(target_input=add_input_name, source_job_uid=source_particle_job, source_output='particles_with_confidence')

    # 获取输入的cs信息，包含add_input()函数的slots参数指定的所有内容
    input_particles_dataset = job.load_input(name=add_input_name)

    with job.run():
        sorted_confidence_dict = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
        safe_unbalanced_particle_number = (int)(math.floor(particle_number * particle_num_multiple_for_cluster))
        safe_unbalanced_particle_number = safe_unbalanced_particle_number if (safe_unbalanced_particle_number <= len(sorted_confidence_dict)) else len(sorted_confidence_dict)

        temp_selected_uids = []
        for confidence_dict_item_ptr in range(safe_unbalanced_particle_number):
            temp_selected_uids.append(sorted_confidence_dict[confidence_dict_item_ptr][0])

        confidence_sorted = []
        features_sorted = []
        job.log('Start sorting confidence...')
        for i in range(safe_unbalanced_particle_number):
            select_uid = temp_selected_uids[i]
            confidence_sorted.append(confidence_dict[select_uid])
            features_sorted.append(features_dict[(int)(select_uid)])
        features_all_sorted = np.array(features_sorted)
        job.log('Sorting confidence succeed!')

        job.log('Start clustering...')
        labels_predict, centers_predict, num_class = classification_inference.features_n_kmeans(features_all_sorted, k, job, merge_threshold=0)
        sorted_indices_dict = classification_inference.sort_labels_by_score(confidence_sorted, labels_predict, temp_selected_uids)
        job.log('Cluster succeed!')

        sorted_indices_list = [sorted(sorted_indices_dict[i].items(), key=lambda x: x[1], reverse=True) for i in range(k)]
        for i in range(k):
            selected_uids = []
            for j in range(len(sorted_indices_list[i])):
                selected_uids.append(sorted_indices_list[i][j][0])
            output_particles_dataset = input_particles_dataset.query({'uid': selected_uids})
            add_output_name = 'particles_selected_' + (str)(i)
            job.add_output(type='particle', name=add_output_name, slots=['blob'], passthrough=add_input_name)
            job.save_output(add_output_name, output_particles_dataset)

    return job, safe_unbalanced_particle_number



new_cluster_particles_job, safe_unbalanced_particle_number = ClusterParticlesWithQuantity((int)(sys.argv[1]))
print(new_cluster_particles_job.uid, 'finished, particles used in clustering is', safe_unbalanced_particle_number, flush=True)