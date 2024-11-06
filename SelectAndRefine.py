#!/usr/bin/env python

import math
import scipy
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import datetime
import shutil
import time
import os

import CryoRanker_inference
import MyLib.CSLogin as CSLogin
import MyLib.MyJobAPIs as MyJobAPIs
import MyLib.mytoolbox as mytoolbox



print('Select and refine start...', flush=True)
wholetimebegin = datetime.datetime.now()



def excutecommand(command):
    # 执行命令行指令并实时获取stdout的打印信息
    '''Execute command line commands and get stdout print information in real time'''
    command_process = subprocess.Popen(command + '&&echo \"\nexcuting_command_completed\n\"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        line = ''
        single_character = command_process.stdout.read(1)
        while not ((single_character == '\r') or (single_character == '\n')):
            line += single_character
            single_character = command_process.stdout.read(1)
        line = line.rstrip()
        if (line == 'cryo_select_slurm_script_completed'):
            break
        if (line == 'excuting_command_completed'):
            break
        if (len(line) > 0):
            yield line


def wait_for_done_with_auto_restart(dealjobs_instance, jobuid, if_safe_mode, max_trying_time):
    # 监测job是否完成，且job配合multithreadpool.setthread(gpupools.AutoQueueJob, ...)使用失败时可以自动重启
    '''Monitor whether the job is completed, and the job can be automatically restarted when used with multithreadpool.setthread(gpupools.AutoQueueJob, ...) fails'''
    job = dealjobs_instance.cshandle.find_job(dealjobs_instance.project, jobuid)
    if if_safe_mode:
        job.wait_for_done(error_on_incomplete=True)
    else:
        trying_time = 0
        while (trying_time <= max_trying_time):
            job.wait_for_done()
            if (job.status == 'completed'):
                break
            else:
                time.sleep(30)
                trying_time += 1
        if not (job.status == 'completed'):
            print('Job status error...', flush=True)
            exit()


def CreateExternal_GetConfidence(DealJobs_instance, source_particle_job, source_particle_parameter_name, if_slurm, if_initial_orientation_balance):

    # DealJobs_instance: 实例化的DealJobs，使用自己的账号密码实例化
    '''DealJobs_instance: Instantiated DealJobs, instantiated with your own account password'''

    # whole_particles_list用于储存所有获得的particles job，list类型，每一个item均为一个字典：{'source_particle_job': source particle job uid, 'source_particle_parameter_name': source particle parameter name}
    '''whole_particles_list is used to store all the particles jobs obtained, list type, each item is a dictionary: {'source_particle_job': source particle job uid, 'source_particle_parameter_name': source particle parameter name}'''

    # 创建空白 external job
    '''Create a blank external job'''

    project = DealJobs_instance.cshandle.find_project(DealJobs_instance.project)
    job = project.create_external_job(DealJobs_instance.workspace)

    # 创建和链接输入
    '''Create and link input'''
    add_input_name = 'input_particles'
    job.add_input(type='particle', name=add_input_name, min=1, slots=['blob', 'ctf'], title='Input particles for getting confidence')
    job.connect(target_input=add_input_name, source_job_uid=source_particle_job, source_output=source_particle_parameter_name)

    # 获取输入的cs信息，包含add_input()函数的slots参数指定的所有内容
    '''Get the cs information of the input, including all the contents specified by the slots parameter of the add_input() function'''
    input_particles_dataset = job.load_input(name=add_input_name)

    with job.run():
        # 调用大模型生成confidence_dict
        '''Call the large model to generate confidence_dict'''
        globaldir = os.getcwd()
        filedir = os.path.dirname(__file__)
        if not (globaldir[-1] == '/'):
            globaldir = globaldir + '/'
        if not (filedir[-1] == '/'):
            filedir = filedir + '/'

        if_need_run = False
        if if_initial_orientation_balance:
            if not (os.path.exists(globaldir + 'metadata/confidence_dict.json') and os.path.exists(globaldir + 'metadata/features_dict.pickle')):
                if_need_run = True
        else:
            if not os.path.exists(globaldir + 'metadata/confidence_dict.json'):
                if_need_run = True

        if if_need_run:
            if if_slurm:
                if os.path.exists(globaldir + 'metadata/confidence_dict.json'):
                    os.remove(globaldir + 'metadata/confidence_dict.json')
                if os.path.exists(globaldir + 'metadata/features_dict.pickle'):
                    os.remove(globaldir + 'metadata/features_dict.pickle')
                if os.path.exists(globaldir + 'metadata/GetConfidence.sh'):
                    os.remove(globaldir + 'metadata/GetConfidence.sh')
                if os.path.exists(globaldir + 'metadata/GetConfidence.out'):
                    os.remove(globaldir + 'metadata/GetConfidence.out')
                shutil.copy(globaldir + 'parameters/GetConfidence.sh', globaldir + 'metadata/GetConfidence.sh')
                with open(globaldir + 'metadata/GetConfidence.sh', 'a') as f:
                    f.write('#SBATCH -o GetConfidence.out\n')
                    f.write('#SBATCH -e GetConfidence.out\n')
                    f.write('cd ' + globaldir + '\n')
                    f.write('srun accelerate launch  --mixed_precision=bf16 ' + filedir + 'GetScores.py\n')
                    f.write('echo \"\ncryo_select_slurm_script_completed\n\"\n')
                if not os.path.exists(globaldir + 'metadata/processpids'):
                    os.makedirs(globaldir + 'metadata/processpids')
                os.system('cd ' + globaldir + 'metadata&&sbatch GetConfidence.sh > ' + globaldir + 'metadata/processpids/slurmuid')
                while True:
                    if os.path.exists(globaldir + 'metadata/GetConfidence.out'):
                        break
                for line in excutecommand('tail -f ' + globaldir + 'metadata/GetConfidence.out'):
                    job.log(line)
                if os.path.exists(globaldir + 'metadata/processpids/slurmuid'):
                    os.remove(globaldir + 'metadata/processpids/slurmuid')
            else:
                # 建议使用slurm
                if os.path.exists(globaldir + 'metadata/confidence_dict.json'):
                    os.remove(globaldir + 'metadata/confidence_dict.json')
                if os.path.exists(globaldir + 'metadata/features_dict.pickle'):
                    os.remove(globaldir + 'metadata/features_dict.pickle')
                if os.path.exists(globaldir + 'metadata/GetConfidence.sh'):
                    os.remove(globaldir + 'metadata/GetConfidence.sh')
                with open(globaldir + 'metadata/GetConfidence.sh', 'w') as f:
                    f.write('#!/bin/bash\n')
                    f.write('pid=$$\n')
                    f.write('echo $pid > ' + globaldir + 'metadata/processpids/$pid\n')
                    f.write('cd ' + globaldir + '\n')
                    f.write('accelerate launch  --mixed_precision=bf16 ' + filedir + 'GetScores.py\n')
                if not os.path.exists(globaldir + 'metadata/processpids'):
                    os.makedirs(globaldir + 'metadata/processpids')
                for line in excutecommand('bash ' + globaldir + 'metadata/GetConfidence.sh'):
                    job.log(line)
        else:
            if if_initial_orientation_balance:
                job.log('confidence_dict.json and features_dict.pickle already exists!')
            else:
                job.log('confidence_dict.json already exists!')

        output_particles_dataset = input_particles_dataset

        # 创建输出，并保存本job的cs信息
        '''Create output and save the cs information of this job'''

        # cs里修改过的slot需要在add_output()函数的slots参数中全部指定来用于保存为新的，剩下的内容passthrough参数会自动从input中拉取并保存在xxx_passthrough.cs中
        '''The slot modified in cs needs to be specified in the slots parameter of the add_output() function to be saved as a new one, and the remaining content of the passthrough parameter will be automatically pulled from the input and saved in xxx_passthrough.cs'''

        add_output_name = 'particles_with_confidence'
        job.add_output(type='particle', name=add_output_name, slots=['blob'], passthrough=add_input_name)
        job.save_output(add_output_name, output_particles_dataset)

    return job


def CreateExternal_SelectParticles(DealJobs_instance, source_particle_job, source_particle_parameter_name, confidence_dict, particle_number_truncation_point, particle_num_multiple_for_cluster, features_dict, k):
    # DealJobs_instance: 实例化的DealJobs，使用自己的账号密码实例化
    # source_particle_job: 输入的job，例如'J1'
    # source_particle_parameter_name: 输入job的output name，例如'imported_particles'
    # confidence_dict: 大模型对输入job的每个particle生成的置信度，字典类型，key为cs中每个particle的uid，value为置信度
    # particle_number_truncation_point: 根据置信度排序，筛选颗粒数量的单个截断点
    # particle_num_multiple_for_cluster: 表示需要particle_number_truncation_point多少倍的数量来聚类particle以平衡朝向，如果为None则表示不需要做聚类平衡
    # features_dict, k: 分别表示大模型得到的feature和聚类数量，如果particle_num_multiple_for_cluster不为None，则需提供这两个值

    # 创建空白 external job
    '''Create a blank external job'''
    project = DealJobs_instance.cshandle.find_project(DealJobs_instance.project)
    job = project.create_external_job(DealJobs_instance.workspace)

    # 创建和链接输入
    '''Create and link input'''
    add_input_name = 'input_particles'
    job.add_input(type='particle', name=add_input_name, min=1, slots=['blob', 'ctf'], title='Input particles for selection')
    job.connect(target_input=add_input_name, source_job_uid=source_particle_job, source_output=source_particle_parameter_name)

    # 获取输入的cs信息，包含add_input()函数的slots参数指定的所有内容
    '''Get the cs information of the input, including all the contents specified by the slots parameter of the add_input() function'''
    input_particles_dataset = job.load_input(name=add_input_name)

    with job.run():
        # 根据confidence_truncation_point筛选particles
        '''Filter particles according to confidence_truncation_point'''
        if particle_num_multiple_for_cluster is not None:
            unbalanced_particle_number_truncation_point = (int)(math.floor(particle_number_truncation_point * particle_num_multiple_for_cluster))
            if (unbalanced_particle_number_truncation_point > len(confidence_dict.items())):
                unbalanced_particle_number_truncation_point = len(confidence_dict.items())

            temp_selected_uids = []
            sorted_confidence_dict = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
            for confidence_dict_item_ptr in range(unbalanced_particle_number_truncation_point):
                temp_selected_uids.append(sorted_confidence_dict[confidence_dict_item_ptr][0])

            confidence_sorted = []
            features_sorted = []
            job.log('Start sorting confidence...')
            for i in range(unbalanced_particle_number_truncation_point):
                select_uid = temp_selected_uids[i]
                confidence_sorted.append(confidence_dict[select_uid])
                features_sorted.append(features_dict[(int)(select_uid)])
            features_all_sorted = np.array(features_sorted)
            job.log('Sorting confidence succeed!')

            job.log('Start clustering...')
            labels_predict, centers_predict, num_class = CryoRanker_inference.features_n_kmeans(features_all_sorted, k, job, merge_threshold=0)
            sorted_indices_dict = CryoRanker_inference.sort_labels_by_score(confidence_sorted, labels_predict, temp_selected_uids)
            # k = len(sorted_indices_dict.keys())
            job.log('Cluster succeed!')

            # job.log('Start selecting particles...')
            # selected_uids = []
            # key_ptr = [0 for _ in range(k)]
            # sorted_indices_list = [sorted(sorted_indices_dict[i].items(), key=lambda x: x[1], reverse=True) for i in range(k)]
            # while (len(selected_uids) < particle_number_truncation_point):
            #     for int_key in range(k):
            #         if (key_ptr[int_key] < len(sorted_indices_list[int_key])):
            #             selected_uids.append(sorted_indices_list[int_key][key_ptr[int_key]][0])
            #             key_ptr[int_key] += 1
            #         if (len(selected_uids) >= particle_number_truncation_point):
            #             break
            # job.log('Selecting particles succeed!')

            sorted_indices_list = [sorted(sorted_indices_dict[i].items(), key=lambda x: x[1], reverse=True) for i in range(k)]
            for i in range(k):
                selected_uids = []
                for j in range(len(sorted_indices_list[i])):
                    selected_uids.append(sorted_indices_list[i][j][0])
                output_particles_dataset = input_particles_dataset.query({'uid': selected_uids})
                add_output_name = 'particles_selected_' + (str)(i)
                job.add_output(type='particle', name=add_output_name, slots=['blob'], passthrough=add_input_name)
                job.save_output(add_output_name, output_particles_dataset)
        else:
            selected_uids = []
            sorted_confidence_dict = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
            safe_particle_number = particle_number_truncation_point if (particle_number_truncation_point <= len(sorted_confidence_dict)) else len(sorted_confidence_dict)
            for confidence_dict_item_ptr in range(safe_particle_number):
                selected_uids.append(sorted_confidence_dict[confidence_dict_item_ptr][0])

            output_particles_dataset = input_particles_dataset.query({'uid': selected_uids})

            # 创建输出，并保存本job的cs信息
            '''Create output and save the cs information of this job'''

            # cs里修改过的slot需要在add_output()函数的slots参数中全部指定来用于保存为新的，剩下的内容passthrough参数会自动从input中拉取并保存在xxx_passthrough.cs中
            '''The slot modified in cs needs to be specified in the slots parameter of the add_output() function to be saved as a new one, and the remaining content of the passthrough parameter will be automatically pulled from the input and saved in xxx_passthrough.cs'''
            add_output_name = 'particles_selected'
            job.add_output(type='particle', name=add_output_name, slots=['blob'], passthrough=add_input_name)
            job.save_output(add_output_name, output_particles_dataset)

    return job


def CreateExternal_PlotConfidenceCurve(DealJobs_instance, source_nurefine_job, source_nurefine_parameter_name, curvedata, confidence_dict):
    # DealJobs_instance: 实例化的DealJobs，使用自己的账号密码实例化
    # source_nurefine_job: 输入的job，例如'J1'
    # source_nurefine_parameter_name: 输入job的output name，例如'imported_particles'
    # curvedata: 转为list后的second_nurefine_resolutions + final_nurefine_resolutions，例如[[100000, 3.2656377882776666], [200000, 3.1664877681943118], [300000, 3.1955818172963864], [400000, 3.3444307416793846]] + [[130000, 3.2656377882776666], [160000, 3.1664877681943118], [230000, 3.1955818172963864], [260000, 3.3444307416793846]]
    # confidence_dict: 大模型对输入job的每个particle生成的置信度，字典类型，key为cs中每个particle的uid，value为置信度

    # 创建空白 external job
    '''Create a blank external job'''
    project = DealJobs_instance.cshandle.find_project(DealJobs_instance.project)
    job = project.create_external_job(DealJobs_instance.workspace)

    # 创建和链接输入
    '''Create and link input'''
    add_input_name = 'final_particles'
    job.add_input(type='particle', name=add_input_name, min=1, slots=['blob', 'ctf', 'alignments3D'], title='Final particles')
    job.connect(target_input=add_input_name, source_job_uid=source_nurefine_job, source_output=source_nurefine_parameter_name)

    # 获取输入的cs信息，包含add_input()函数的slots参数指定的所有内容
    '''Get the cs information of the input, including all the contents specified by the slots parameter of the add_input() function'''
    input_particles_dataset = job.load_input(name=add_input_name)

    with job.run():
        if (len(curvedata) > 3):
            # 绘制curve，获取output_particles_dataset
            '''Draw curve, get output_particles_dataset'''
            globaldir = os.getcwd()
            if not (globaldir[-1] == '/'):
                globaldir = globaldir + '/'
            sorted_curvedata = sorted(curvedata, key=lambda x: x[0])

            sorted_particle_number_curvedatax = (np.asarray(sorted_curvedata).T)[0].tolist()
            sorted_particle_number_curvedatay = (np.asarray(sorted_curvedata).T)[1].tolist()
            interpolate_particle_number_function = scipy.interpolate.interp1d(sorted_particle_number_curvedatax, sorted_particle_number_curvedatay, kind='quadratic', bounds_error=False)
            interpolated_sorted_particle_number_curvedatax = np.arange(sorted_particle_number_curvedatax[0], sorted_particle_number_curvedatax[-1], (sorted_particle_number_curvedatax[-1] - sorted_particle_number_curvedatax[0]) / 100.0)
            interpolated_sorted_particle_number_curvedatay = interpolate_particle_number_function(interpolated_sorted_particle_number_curvedatax)
            side_blank = (interpolated_sorted_particle_number_curvedatax[-1] - interpolated_sorted_particle_number_curvedatax[0]) / 20.0
            plt.clf()
            plt.ioff()
            plt.plot(interpolated_sorted_particle_number_curvedatax, interpolated_sorted_particle_number_curvedatay)
            plt.xlim(interpolated_sorted_particle_number_curvedatax[0] - side_blank, interpolated_sorted_particle_number_curvedatax[-1] + side_blank)
            plt.xlabel('Particle number')
            plt.ylabel('Resolution (A)')
            plt.title('ParticleNumber-Resolution curve')
            plt.savefig(globaldir + 'metadata/ParticleNumber-Resolution.pdf')
            job.log_plot(plt, 'Plot ParticleNumber-Resolution curve')

            reverse_sorted_curvedata = sorted(curvedata, key=lambda x: x[0], reverse=True)
            sorted_confidence_dict = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_confidence_curvedatax = (np.asarray(reverse_sorted_curvedata).T)[0].tolist()
            for i in range(len(sorted_confidence_curvedatax)):
                sorted_confidence_curvedatax[i] = round(sorted_confidence_dict[(int)(math.floor(sorted_confidence_curvedatax[i] - 1))][1], 5)
            sorted_confidence_curvedatay = (np.asarray(reverse_sorted_curvedata).T)[1].tolist()
            interpolate_confidence_function = scipy.interpolate.interp1d(sorted_confidence_curvedatax, sorted_confidence_curvedatay, kind='quadratic', bounds_error=False)
            interpolated_sorted_confidence_curvedatax = np.arange(sorted_confidence_curvedatax[0], sorted_confidence_curvedatax[-1], (sorted_confidence_curvedatax[-1] - sorted_confidence_curvedatax[0]) / 100.0)
            interpolated_sorted_confidence_curvedatay = interpolate_confidence_function(interpolated_sorted_confidence_curvedatax)
            side_blank = (interpolated_sorted_confidence_curvedatax[-1] - interpolated_sorted_confidence_curvedatax[0]) / 20.0
            plt.clf()
            plt.ioff()
            plt.plot(interpolated_sorted_confidence_curvedatax, interpolated_sorted_confidence_curvedatay)
            plt.xlim(interpolated_sorted_confidence_curvedatax[-1] + side_blank, interpolated_sorted_confidence_curvedatax[0] - side_blank)
            plt.xlabel('Confidence')
            plt.ylabel('Resolution (A)')
            plt.title('Confidence-Resolution curve')
            plt.savefig(globaldir + 'metadata/Confidence-Resolution.pdf')
            job.log_plot(plt, 'Plot Confidence-Resolution curve')

            sorted_pertinence_curvedatax = (np.asarray(sorted_curvedata).T)[0].tolist()
            sorted_pertinence_curvedatay = (np.asarray(sorted_curvedata).T)[0].tolist()
            for i in range(len(sorted_pertinence_curvedatax)):
                sorted_pertinence_curvedatay[i] = round(sorted_confidence_dict[(int)(math.floor(sorted_pertinence_curvedatax[i] - 1))][1], 5)
            interpolate_pertinence_function = scipy.interpolate.interp1d(sorted_pertinence_curvedatax, sorted_pertinence_curvedatay, kind='quadratic', bounds_error=False)
            interpolated_sorted_pertinence_curvedatax = np.arange(sorted_pertinence_curvedatax[0], sorted_pertinence_curvedatax[-1], (sorted_pertinence_curvedatax[-1] - sorted_pertinence_curvedatax[0]) / 100.0)
            interpolated_sorted_pertinence_curvedatay = interpolate_pertinence_function(interpolated_sorted_pertinence_curvedatax)
            plt.clf()
            plt.ioff()
            plt.plot(interpolated_sorted_pertinence_curvedatax, interpolated_sorted_pertinence_curvedatay)
            plt.xlabel('Particle number')
            plt.ylabel('Confidence')
            plt.title('ParticleNumber-Confidence curve')
            plt.savefig(globaldir + 'metadata/ParticleNumber-Confidence.pdf')
            job.log_plot(plt, 'Plot ParticleNumber-Confidence curve')

            best_particle_number = sorted(curvedata, key=lambda x: x[1])[0][0]
            best_confidence = round(sorted_confidence_dict[(int)(math.floor(best_particle_number - 1))][1], 5)
            job.log('Best particle number: ' + (str)(best_particle_number))
            job.log('Best confidence: ' + (str)(best_confidence))

        else:
            job.log('Eligible job number is too small...')

        output_particles_dataset = input_particles_dataset

        # 创建输出，并保存本job的cs信息
        '''Create output and save the cs information of this job'''

        # cs里修改过的slot需要在add_output()函数的slots参数中全部指定来用于保存为新的，剩下的内容passthrough参数会自动从input中拉取并保存在xxx_passthrough.cs中
        '''The slot modified in cs needs to be specified in the slots parameter of the add_output() function to be saved as a new one, and the remaining content of the passthrough parameter will be automatically pulled from the input and saved in xxx_passthrough.cs'''

        add_output_name = 'best_particles'
        job.add_output(type='particle', name=add_output_name, slots=['blob'], passthrough=add_input_name)
        job.save_output(add_output_name, output_particles_dataset)

    return job





# 初始化需要的内容
'''Initialize the required content'''
globaldir = os.getcwd()
if not (globaldir[-1] == '/'):
    globaldir = globaldir + '/'
parameters = mytoolbox.readjson(globaldir + 'parameters/parameters.json')
if_safe_mode = parameters['safe_mode']
if_low_cache_mode = parameters['low_cache_mode']
if_slurm = parameters['if_slurm']
if_initial_orientation_balance = parameters['initial_orientation_balance']
max_trying_time = parameters['max_trying_time']
cluster_k = parameters['k']
cfar_lower_bound = parameters['cfar_lower_bound']
resolution_lower_bound = parameters['resolution_lower_bound']
particle_num_multiple_for_cluster = parameters['particle_num_multiple_for_cluster']
cshandle = CSLogin.cshandleclass.GetCryoSPARCHandle(email=parameters['cryosparc_username'], password=parameters['cryosparc_password'])
dealjobs = MyJobAPIs.DealJobs(cshandle, parameters['project'], parameters['workspace'], parameters['lane'])
if not if_safe_mode:
    hostname_list = []
    gpu_num_list = []
    max_num_of_jobs_list = []
    for i in range(len(parameters['hostname_gpus_jobnum_lists'])):
        hostname_list.append(parameters['hostname_gpus_jobnum_lists'][i][0])
        gpu_num_list.append(parameters['hostname_gpus_jobnum_lists'][i][1])
        max_num_of_jobs_list.append(parameters['hostname_gpus_jobnum_lists'][i][2])
    gpupools = MyJobAPIs.HostAndGPUPools(dealjobs, hostname_list, gpu_num_list, max_num_of_jobs_list)
    multithreadpool = mytoolbox.MultiThreadingRun(gpupools.max_job_num)
abinit_class = MyJobAPIs.CreateHomoAbinit(dealjobs)
abinit_parameters = mytoolbox.readjson(globaldir + 'parameters/abinit_parameters.json')
nurefine_class = MyJobAPIs.CreateNonuniformRefine(dealjobs)
nurefine_parameters = mytoolbox.readjson(globaldir + 'parameters/nurefine_parameters.json')
second_nurefine_parameters = mytoolbox.readjson(globaldir + 'parameters/final_nurefine_parameters.json')
final_nurefine_parameters = mytoolbox.readjson(globaldir + 'parameters/final_nurefine_parameters.json')
if if_low_cache_mode:
    restack_particles_class = MyJobAPIs.CreateRestackParticles(dealjobs)
    restack_particles_parameters = mytoolbox.readjson(globaldir + 'parameters/restack_particles_parameters.json')
if if_initial_orientation_balance:
    orientation_diagnostics_class = MyJobAPIs.CreateOrientationDiagnostics(dealjobs)
    orientation_diagnostics_parameters = mytoolbox.readjson(globaldir + 'parameters/orientation_diagnostics_parameters.json')





# 识别输入的particle job的类别，支持的particle job有：Import Particle Stack, Extract From Micrographs(Multi-GPU), Restack Particles
'''Identify the type of the input particle job, and the supported particle jobs are: Import Particle Stack, Extract From Micrographs(Multi-GPU), Restack Particles'''
source_particle_job = mytoolbox.readjson(globaldir + 'metadata/particle_job_uid.json')
files_in_job_dir = os.listdir((str)(dealjobs.cshandle.find_project(dealjobs.project).dir()) + '/' + source_particle_job)
if 'imported_particles.cs' in files_in_job_dir:
    source_particle_parameter_name = 'imported_particles'
elif 'extracted_particles.cs' in files_in_job_dir:
    source_particle_parameter_name = 'particles'
elif 'restacked_particles.cs' in files_in_job_dir:
    source_particle_parameter_name = 'particles'
elif 'split_0000.cs' in files_in_job_dir:
    source_particle_parameter_name = 'split_0'
else:
    print('Job type error...', flush=True)
    exit()



# 创建external_get_confidence job，调用大模型生成confidence_dict
'''Create external_get_confidence job, call the large model to generate confidence_dict'''
print('Get confidence...', flush=True)
new_external_get_confidence_job = CreateExternal_GetConfidence(dealjobs, source_particle_job, source_particle_parameter_name, if_slurm, if_initial_orientation_balance)
getting_confidence_job = new_external_get_confidence_job.uid
mytoolbox.savetojson({'external_get_confidence_jobuid': getting_confidence_job}, globaldir + 'metadata/external_get_confidence_jobuid.json', False)
print('Getting confidence completed!', flush=True)


# 设置abinit的置信度截断点
'''Set the confidence truncation point of abinit'''
# abinit_particle_number_truncation_points: 计算得到的截断点列表，列表类型
'''abinit_particle_number_truncation_points: List of calculated truncation points, list type'''
confidence_dict = mytoolbox.readjson(globaldir + 'metadata/confidence_dict.json')
sorted_confidence_dict = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
if ((parameters['base_abinit_particle_num'] * parameters['abinit_particle_num_step']) > len(sorted_confidence_dict)):
    base_abinit_particle_num = (int)(math.floor((float)(len(sorted_confidence_dict)) / parameters['abinit_particle_num_step']))
else:
    base_abinit_particle_num = parameters['base_abinit_particle_num']
abinit_particle_number_truncation_points = [(i + 1) * base_abinit_particle_num for i in range(parameters['abinit_particle_num_step'])]



############################################### get best initial volume ############################################################



# 通过external job筛选出几批particle做第一轮abinit+refine
'''Select several batches of particles through external job for the first round of abinit+refine'''
# external_select_particles_jobs_parents用于存储生成的external selection job的uid和对应的输入job uid，dict类型，key为job uid，value为{'particle': source_particle_job}
'''external_select_particles_jobs_parents is used to store the uid of the generated external selection job and the corresponding input job uid, dict type, key is job uid, value is {'particle': source_particle_job}'''
# external_select_particles_jobs_truncation_points用于存储生成的external selection job的uid和对应的truncation_point，dict类型，key为job uid，value为truncation_point
'''external_select_particles_jobs_truncation_points is used to store the uid of the generated external selection job and the corresponding truncation_point, dict type, key is job uid, value is truncation_point'''
external_select_particles_jobs_parents = {}
external_select_particles_jobs_truncation_points = {}
if if_low_cache_mode:
    # restack_particles_jobs_parents用于存储生成的restack particles job的uid和对应的输入job uid，dict类型，key为job uid，value为{'particle': source_particle_job}
    '''restack_particles_jobs_parents is used to store the uid of the generated restack particles job and the corresponding input job uid, dict type, key is job uid, value is {'particle': source_particle_job}'''
    # restack_particles_jobs_truncation_points用于存储生成的restack particles job的uid和对应的truncation_point，dict类型，key为job uid，value为truncation_point
    '''restack_particles_jobs_truncation_points is used to store the uid of the generated restack particles job and the corresponding truncation_point, dict type, key is job uid, value is truncation_point'''
    restack_particles_jobs_parents = {}
    restack_particles_jobs_truncation_points = {}
for i in range(len(abinit_particle_number_truncation_points)):
    truncation_point = abinit_particle_number_truncation_points[i]
    new_external_select_particles_job = CreateExternal_SelectParticles(dealjobs, getting_confidence_job, 'particles_with_confidence', confidence_dict, truncation_point, None, None, None)
    external_select_particles_jobs_parents[new_external_select_particles_job.uid] = {'particle': getting_confidence_job}
    external_select_particles_jobs_truncation_points[new_external_select_particles_job.uid] = truncation_point
    if if_low_cache_mode:
        new_restack_particles_job = restack_particles_class.QueueRestackJob(restack_particles_parameters, [new_external_select_particles_job.uid], ['particles_selected'])
        print('Restack particles job (', new_restack_particles_job.uid, ') has been queued, please wait for job to complete...', flush=True)
        restack_particles_jobs_parents[new_restack_particles_job.uid] = {'particle': [new_external_select_particles_job.uid]}
        restack_particles_jobs_truncation_points[new_restack_particles_job.uid] = truncation_point
if if_low_cache_mode:
    for key, value in restack_particles_jobs_parents.items():
        wait_for_done_with_auto_restart(dealjobs, key, True, max_trying_time)
mytoolbox.savetojson(external_select_particles_jobs_parents, globaldir + 'metadata/external_select_particles_jobs_parents.json', False)
mytoolbox.savetojson(external_select_particles_jobs_truncation_points, globaldir + 'metadata/external_select_particles_jobs_truncation_points.json', False)
if if_low_cache_mode:
    mytoolbox.savetojson(restack_particles_jobs_parents, globaldir + 'metadata/restack_particles_jobs_parents.json', False)
    mytoolbox.savetojson(restack_particles_jobs_truncation_points, globaldir + 'metadata/restack_particles_jobs_truncation_points.json', False)
print('All external selection jobs completed!', flush=True)


# 通过external job筛选出的几批particle创建abinit refine和一轮nu-refine
'''Create abinit refine and one round of nu-refine for several batches of particles selected through external job'''
# abinit_jobs_parents用于存储生成的abinit job的uid和对应的输入job uid，dict类型，key为abinit job uid，value为{'particle': [external select particles job uid]}
'''abinit_jobs_parents is used to store the uid of the generated abinit job and the corresponding input job uid, dict type, key is abinit job uid, value is {'particle': [external select particles job uid]}'''
abinit_jobs_parents = {}
if if_low_cache_mode:
    for key, value in restack_particles_jobs_parents.items():
        new_abinit_job = abinit_class.QueueHomoAbinitJob(abinit_parameters, [key], ['particles'])
        if not if_safe_mode:
            dealjobs.ClearJobSafely([new_abinit_job.uid])
            multithreadpool.setthread(gpupools.AutoQueueJob, job=new_abinit_job, gpus_num_needed=1)
        print('Abinit job (', new_abinit_job.uid, ') has been created, please wait for job to complete...', flush=True)
        abinit_jobs_parents[new_abinit_job.uid] = {'particle': [key]}
else:
    for key, value in external_select_particles_jobs_parents.items():
        new_abinit_job = abinit_class.QueueHomoAbinitJob(abinit_parameters, [key], ['particles_selected'])
        if not if_safe_mode:
            dealjobs.ClearJobSafely([new_abinit_job.uid])
            multithreadpool.setthread(gpupools.AutoQueueJob, job=new_abinit_job, gpus_num_needed=1, max_trying_time=max_trying_time)
        print('Abinit job (', new_abinit_job.uid, ') has been created, please wait for job to complete...', flush=True)
        abinit_jobs_parents[new_abinit_job.uid] = {'particle': [key]}
for key, _ in abinit_jobs_parents.items():
    wait_for_done_with_auto_restart(dealjobs, key, if_safe_mode, max_trying_time)
mytoolbox.savetojson(abinit_jobs_parents, globaldir + 'metadata/abinit_jobs_parents.json', False)
print('All abinit jobs completed!', flush=True)


# nurefine_jobs_parents用于存储生成的nu-refine job的uid和对应的输入job uid，dict类型，key为nu-refine job uid，value为{'particle': external select particles job uid, 'volume': abinit job uid}
'''nurefine_jobs_parents is used to store the uid of the generated nu-refine job and the corresponding input job uid, dict type, key is nu-refine job uid, value is {'particle': external select particles job uid, 'volume': abinit job uid}'''
# nurefine_jobs_resolutions用于存储生成的nu-refine job的uid和对应的final resolution，dict类型，key为nu-refine job uid，value为resolution
'''nurefine_jobs_resolutions is used to store the uid of the generated nu-refine job and the corresponding final resolution, dict type, key is nu-refine job uid, value is resolution'''
nurefine_jobs_parents = {}
nurefine_jobs_resolutions = {}
if if_initial_orientation_balance:
    # nurefine_jobs_orientation_info用于存储生成的nu-refine job的uid和对应的cfar和scf值，dict类型，key为nu-refine job uid，value为{'cfar': cfar, 'scf': scf}
    '''nurefine_jobs_orientation_info is used to store the uid of the generated nu-refine job and the corresponding cfar and scf values, dict type, key is nu-refine job uid, value is {'cfar': cfar, 'scf': scf}'''
    # orientation_jobs_parents用于存储生成的orientation diagnostics job的uid和对应的输入job uid，dict类型，key为orientation diagnostics job uid，value为{'volume': nu-refine job uid, 'particle': [nu-refine job uid]}
    '''orientation_jobs_parents is used to store the uid of the generated orientation diagnostics job and the corresponding input job uid, dict type, key is orientation diagnostics job uid, value is {'volume': nu-refine job uid, 'particle': [nu-refine job uid]}'''
    nurefine_jobs_orientation_info = {}
    orientation_jobs_parents = {}
if if_low_cache_mode:
    for key, value in abinit_jobs_parents.items():
        new_nurefine_job = nurefine_class.QueueNonuniformRefineJob(nurefine_parameters, [value['particle'][0]], ['particles'], key,  'volume_class_0')
        if not if_safe_mode:
            dealjobs.ClearJobSafely([new_nurefine_job.uid])
            multithreadpool.setthread(gpupools.AutoQueueJob, job=new_nurefine_job, gpus_num_needed=1)
        print('NU-refine job (', new_nurefine_job.uid, ') has been created, please wait for job to complete...', flush=True)
        nurefine_jobs_parents[new_nurefine_job.uid] = {'particle': [value['particle'][0]], 'volume': key}
else:
    for key, value in abinit_jobs_parents.items():
        new_nurefine_job = nurefine_class.QueueNonuniformRefineJob(nurefine_parameters, [value['particle'][0]], ['particles_selected'], key,  'volume_class_0')
        if not if_safe_mode:
            dealjobs.ClearJobSafely([new_nurefine_job.uid])
            multithreadpool.setthread(gpupools.AutoQueueJob, job=new_nurefine_job, gpus_num_needed=1)
        print('NU-refine job (', new_nurefine_job.uid, ') has been created, please wait for job to complete...', flush=True)
        nurefine_jobs_parents[new_nurefine_job.uid] = {'particle': [value['particle'][0]], 'volume': key}
for key, _ in nurefine_jobs_parents.items():
    wait_for_done_with_auto_restart(dealjobs, key, if_safe_mode, max_trying_time)
    if if_initial_orientation_balance:
        # 创建OrientationDiagnostics job
        '''Create OrientationDiagnostics job'''
        new_orientation_job = orientation_diagnostics_class.QueueOrientationDiagnosticsJob(orientation_diagnostics_parameters, key, 'volume', [key], ['particles'])
        if not if_safe_mode:
            dealjobs.ClearJobSafely([new_orientation_job.uid])
            multithreadpool.setthread(gpupools.AutoQueueJob, job=new_orientation_job, gpus_num_needed=1)
        wait_for_done_with_auto_restart(dealjobs, new_orientation_job.uid, if_safe_mode, max_trying_time)
        orientation_jobs_parents[new_orientation_job.uid] = {'volume': key, 'particle': [key]}
        # 获取信息
        cfar, scf = dealjobs.GetOrientationDiagnosticsCFARandSCF(new_orientation_job.uid)
        nurefine_jobs_orientation_info[key] = {'cfar': cfar, 'scf': scf}
    nurefine_jobs_resolutions[key] = dealjobs.GetNURefineFinalResolution(key)
mytoolbox.savetojson(nurefine_jobs_parents, globaldir + 'metadata/nurefine_jobs_parents.json', False)
mytoolbox.savetojson(nurefine_jobs_resolutions, globaldir + 'metadata/nurefine_jobs_resolutions.json', False)
if if_initial_orientation_balance:
    mytoolbox.savetojson(nurefine_jobs_orientation_info, globaldir + 'metadata/nurefine_jobs_orientation_info.json', False)
    mytoolbox.savetojson(orientation_jobs_parents, globaldir + 'metadata/orientation_jobs_parents.json', False)
print('All nu-refine jobs completed!', flush=True)


# 判断nurefine获得的volume作为后续的initial model是否合格，包括其分辨率和偏向性，如果合格从中挑选一个没有偏向性的情况下分辨率最高的job
'''Determine whether the volume obtained by nurefine is qualified as the subsequent initial model, including its resolution and orientation, if qualified, select the job with the highest resolution without orientation'''
sorted_resolution_info = sorted(nurefine_jobs_resolutions.items(), key=lambda x: x[1])
if not if_initial_orientation_balance:
    best_init_job_uid = sorted_resolution_info[0][0]
    best_init_job_resolution = sorted_resolution_info[0][1]
else:
    orientation_job_flag = False
    for i in range(len(sorted_resolution_info)):
        jobuid = sorted_resolution_info[i][0]
        resolution = sorted_resolution_info[i][1]
        cfar = nurefine_jobs_orientation_info[jobuid]['cfar']
        if (cfar >= cfar_lower_bound) and (resolution <= resolution_lower_bound):
            best_init_job_uid = jobuid
            best_init_job_resolution = resolution
            orientation_job_flag = True
            break
    # 如果没有符合条件的，则启动balance重新做initial流程
    '''If there is no job that meets the conditions, start the balance to redo the initial process'''
    if not orientation_job_flag:
        # 通过external job筛选出几批particle做第一轮abinit+refine
        '''Select several batches of particles through external job for the first round of abinit+refine'''
        # balance_external_select_particles_jobs_parents用于存储生成的balance external selection job的uid和对应的输入job uid，dict类型，key为job uid，value为{'particle': source_particle_job}
        # balance_external_select_particles_jobs_truncation_points用于存储生成的balance external selection job的uid和对应的truncation_point，dict类型，key为job uid，value为truncation_point
        balance_external_select_particles_jobs_parents = {}
        balance_external_select_particles_jobs_truncation_points = {}
        if if_low_cache_mode:
            # balance_restack_particles_jobs_parents用于存储生成的restack particles job的uid和对应的输入job uid，dict类型，key为job uid，value为{'particle': source_particle_job}
            # balance_restack_particles_jobs_truncation_points用于存储生成的restack particles job的uid和对应的truncation_point，dict类型，key为job uid，value为truncation_point
            balance_restack_particles_jobs_parents = {}
            balance_restack_particles_jobs_truncation_points = {}
        print('Start loading feature...', flush=True)
        features_dict = mytoolbox.readpicklefile(globaldir + 'metadata/features_dict.pickle')
        print('Feature loading succeed!', flush=True)
        for i in range(len(abinit_particle_number_truncation_points)):
            truncation_point = abinit_particle_number_truncation_points[i]
            new_external_select_particles_job = CreateExternal_SelectParticles(dealjobs, getting_confidence_job, 'particles_with_confidence', confidence_dict, truncation_point, particle_num_multiple_for_cluster, features_dict, cluster_k)
            balance_external_select_particles_jobs_parents[new_external_select_particles_job.uid] = {'particle': getting_confidence_job}
            balance_external_select_particles_jobs_truncation_points[new_external_select_particles_job.uid] = truncation_point
            if if_low_cache_mode:
                for i in range(cluster_k):
                    new_restack_particles_job = restack_particles_class.QueueRestackJob(restack_particles_parameters, [new_external_select_particles_job.uid], ['particles_selected_' + (str)(i)])
                    print('Restack particles job (', new_restack_particles_job.uid, ') has been queued, please wait for job to complete...', flush=True)
                    balance_restack_particles_jobs_parents[new_restack_particles_job.uid] = {'particle': [new_external_select_particles_job.uid]}
                    balance_restack_particles_jobs_truncation_points[new_restack_particles_job.uid] = truncation_point
        del features_dict
        if if_low_cache_mode:
            for key, value in balance_restack_particles_jobs_parents.items():
                wait_for_done_with_auto_restart(dealjobs, key, True, max_trying_time)
        mytoolbox.savetojson(balance_external_select_particles_jobs_parents, globaldir + 'metadata/balance_external_select_particles_jobs_parents.json', False)
        mytoolbox.savetojson(balance_external_select_particles_jobs_truncation_points, globaldir + 'metadata/balance_external_select_particles_jobs_truncation_points.json', False)
        if if_low_cache_mode:
            mytoolbox.savetojson(balance_restack_particles_jobs_parents, globaldir + 'metadata/balance_restack_particles_jobs_parents.json', False)
            mytoolbox.savetojson(balance_restack_particles_jobs_truncation_points, globaldir + 'metadata/balance_restack_particles_jobs_truncation_points.json', False)
        print('All balance external selection jobs completed!', flush=True)


        # 通过external job筛选出的几批particle创建abinit refine和一轮nu-refine
        '''Create abinit refine and one round of nu-refine for several batches of particles selected through external job'''
        # balance_abinit_jobs_parents用于存储生成的balance abinit job的uid和对应的输入job uid，dict类型，key为abinit job uid，value为{'particle': [external select particles job uid]}
        # balance_abinit_jobs_parents_name用于存储生成的balance abinit job的uid和对应的输入job parameter name，dict类型，key为abinit job uid，value为{'particle': [external select particles job parameter name]}
        balance_abinit_jobs_parents = {}
        balance_abinit_jobs_parents_name = {}
        if if_low_cache_mode:
            for key, value in balance_restack_particles_jobs_parents.items():
                new_abinit_job = abinit_class.QueueHomoAbinitJob(abinit_parameters, [key], ['particles'])
                if not if_safe_mode:
                    dealjobs.ClearJobSafely([new_abinit_job.uid])
                    multithreadpool.setthread(gpupools.AutoQueueJob, job=new_abinit_job, gpus_num_needed=1)
                print('Abinit job (', new_abinit_job.uid, ') has been created, please wait for job to complete...', flush=True)
                balance_abinit_jobs_parents[new_abinit_job.uid] = {'particle': [key]}
                balance_abinit_jobs_parents_name[new_abinit_job.uid] = {'particle_name': ['particles']}
        else:
            for key, value in balance_external_select_particles_jobs_parents.items():
                for i in range(cluster_k):
                    new_abinit_job = abinit_class.QueueHomoAbinitJob(abinit_parameters, [key], ['particles_selected_' + (str)(i)])
                    if not if_safe_mode:
                        dealjobs.ClearJobSafely([new_abinit_job.uid])
                        multithreadpool.setthread(gpupools.AutoQueueJob, job=new_abinit_job, gpus_num_needed=1, max_trying_time=max_trying_time)
                    print('Abinit job (', new_abinit_job.uid, ') has been created, please wait for job to complete...', flush=True)
                    balance_abinit_jobs_parents[new_abinit_job.uid] = {'particle': [key]}
                    balance_abinit_jobs_parents_name[new_abinit_job.uid] = {'particle_name': ['particles_selected_' + (str)(i)]}
        for key, _ in balance_abinit_jobs_parents.items():
            wait_for_done_with_auto_restart(dealjobs, key, if_safe_mode, max_trying_time)
        mytoolbox.savetojson(balance_abinit_jobs_parents, globaldir + 'metadata/balance_abinit_jobs_parents.json', False)
        print('All balance abinit jobs completed!', flush=True)


        # balance_nurefine_jobs_parents用于存储生成的balance nu-refine job的uid和对应的输入job uid，dict类型，key为nu-refine job uid，value为{'particle': external select particles job uid, 'volume': abinit job uid}
        # balance_nurefine_jobs_resolutions用于存储生成的balance nu-refine job的uid和对应的final resolution，dict类型，key为nu-refine job uid，value为resolution
        # balance_nurefine_jobs_orientation_info用于存储生成的balance nu-refine job的uid和对应的cfar和scf值，dict类型，key为nu-refine job uid，value为{'cfar': cfar, 'scf': scf}
        balance_nurefine_jobs_parents = {}
        balance_nurefine_jobs_resolutions = {}
        balance_nurefine_jobs_orientation_info = {}
        balance_orientation_jobs_parents = {}
        if if_low_cache_mode:
            for key, value in balance_abinit_jobs_parents.items():
                new_nurefine_job = nurefine_class.QueueNonuniformRefineJob(nurefine_parameters, [value['particle'][0]], [balance_abinit_jobs_parents_name[key]['particle_name'][0]], key,  'volume_class_0')
                if not if_safe_mode:
                    dealjobs.ClearJobSafely([new_nurefine_job.uid])
                    multithreadpool.setthread(gpupools.AutoQueueJob, job=new_nurefine_job, gpus_num_needed=1)
                print('NU-refine job (', new_nurefine_job.uid, ') has been created, please wait for job to complete...', flush=True)
                balance_nurefine_jobs_parents[new_nurefine_job.uid] = {'particle': [value['particle'][0]], 'volume': key}
        else:
            for key, value in balance_abinit_jobs_parents.items():
                new_nurefine_job = nurefine_class.QueueNonuniformRefineJob(nurefine_parameters, [value['particle'][0]], [balance_abinit_jobs_parents_name[key]['particle_name'][0]], key,  'volume_class_0')
                if not if_safe_mode:
                    dealjobs.ClearJobSafely([new_nurefine_job.uid])
                    multithreadpool.setthread(gpupools.AutoQueueJob, job=new_nurefine_job, gpus_num_needed=1)
                print('NU-refine job (', new_nurefine_job.uid, ') has been created, please wait for job to complete...', flush=True)
                balance_nurefine_jobs_parents[new_nurefine_job.uid] = {'particle': [value['particle'][0]], 'volume': key}
        for key, _ in balance_nurefine_jobs_parents.items():
            wait_for_done_with_auto_restart(dealjobs, key, if_safe_mode, max_trying_time)
            # 创建OrientationDiagnostics job
            '''Create OrientationDiagnostics job'''
            new_orientation_job = orientation_diagnostics_class.QueueOrientationDiagnosticsJob(orientation_diagnostics_parameters, key, 'volume', [key], ['particles'])
            if not if_safe_mode:
                dealjobs.ClearJobSafely([new_orientation_job.uid])
                multithreadpool.setthread(gpupools.AutoQueueJob, job=new_orientation_job, gpus_num_needed=1)
            wait_for_done_with_auto_restart(dealjobs, new_orientation_job.uid, if_safe_mode, max_trying_time)
            balance_orientation_jobs_parents[new_orientation_job.uid] = {'volume': key, 'particle': [key]}
            # 获取信息
            '''Get information'''
            cfar, scf = dealjobs.GetOrientationDiagnosticsCFARandSCF(new_orientation_job.uid)
            balance_nurefine_jobs_orientation_info[key] = {'cfar': cfar, 'scf': scf}
            balance_nurefine_jobs_resolutions[key] = dealjobs.GetNURefineFinalResolution(key)
        mytoolbox.savetojson(balance_nurefine_jobs_parents, globaldir + 'metadata/balance_nurefine_jobs_parents.json', False)
        mytoolbox.savetojson(balance_nurefine_jobs_resolutions, globaldir + 'metadata/balance_nurefine_jobs_resolutions.json', False)
        mytoolbox.savetojson(balance_nurefine_jobs_orientation_info, globaldir + 'metadata/balance_nurefine_jobs_orientation_info.json', False)
        print('All balance nu-refine jobs completed!', flush=True)

        balance_sorted_resolution_info = sorted(balance_nurefine_jobs_resolutions.items(), key=lambda x: x[1])
        best_init_job_uid = None
        for i in range(len(balance_sorted_resolution_info)):
            jobuid = balance_sorted_resolution_info[i][0]
            resolution = balance_sorted_resolution_info[i][1]
            cfar = balance_nurefine_jobs_orientation_info[jobuid]['cfar']
            if (cfar >= cfar_lower_bound) and (resolution <= resolution_lower_bound):
                best_init_job_uid = jobuid
                best_init_job_resolution = resolution
                break
        if best_init_job_uid is None:
            if (sorted_resolution_info[0][1] <= balance_sorted_resolution_info[0][1]):
                best_init_job_uid = sorted_resolution_info[0][0]
                best_init_job_resolution = sorted_resolution_info[0][1]
            else:
                best_init_job_uid = balance_sorted_resolution_info[0][0]
                best_init_job_resolution = balance_sorted_resolution_info[0][1]

print('Best initial job uid:', best_init_job_uid, ', resolution:', best_init_job_resolution, flush=True)



###################################### initial volume get, start refine #########################################



def get_iteration_flag(iteration, particle_number_step):
    if parameters['refine_iteration_min_particle_number_step'] is not None:
        if (particle_number_step >= parameters['refine_iteration_min_particle_number_step']):
            return True
        else:
            return False
    else:
        if (iteration < parameters['refine_iteration_num']):
            return True
        else:
            return False

# refine_external_select_particles_jobs_parents_list用于存储生成的external_select_particles_jobs_parents，顺序与iteration对应
# refine_external_select_particles_jobs_truncation_points_list用于存储生成的external_select_particles_jobs_truncation_points，顺序与iteration对应
# refine_restack_particles_jobs_parents_list用于存储生成的refine_restack_particles_jobs_parents，顺序与iteration对应
# refine_restack_particles_jobs_truncation_points_list用于存储生成的refine_restack_particles_jobs_truncation_points，顺序与iteration对应
# refine_nurefine_jobs_parents_list用于存储生成的refine_nurefine_jobs_parents，顺序与iteration对应
# refine_nurefine_jobs_resolutions_list用于存储生成的refine_nurefine_jobs_resolutions，顺序与iteration对应
# refine_nurefine_jobs_orientation_info_list用于存储生成的refine_nurefine_jobs_orientation_info，顺序与iteration对应
# refine_orientation_jobs_parents_list用于存储生成的refine_orientation_jobs_parents，顺序与iteration对应
refine_external_select_particles_jobs_parents_list = []
refine_external_select_particles_jobs_truncation_points_list = []
if if_low_cache_mode:
    refine_restack_particles_jobs_parents_list = []
    refine_restack_particles_jobs_truncation_points_list = []
refine_nurefine_jobs_parents_list = []
refine_nurefine_jobs_resolutions_list = []
if if_initial_orientation_balance:
    refine_nurefine_jobs_orientation_info_list = []
    refine_orientation_jobs_parents_list = []


# 根据confidence_dict设置第一轮refine的置信度截断点
'''Set the confidence truncation point of the first round of refine according to confidence_dict'''
# confidence_dict: 大模型对输入job的每个particle生成的置信度，字典类型，key为cs中每个particle的uid，value为置信度
# particle_number_truncation_points: 计算得到的截断点列表，列表类型
confidence_dict = mytoolbox.readjson(globaldir + 'metadata/confidence_dict.json')
sorted_confidence_dict = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
refine_grid_num_in_each_turn = parameters['refine_grid_num_in_each_turn']
min_truncation_confidence = parameters['min_refine_truncation_confidence']
max_truncation_confidence = parameters['max_refine_truncation_confidence']
min_truncation_particle_num = 0
max_truncation_particle_num = len(sorted_confidence_dict)
for confidence_dict_item_ptr in range(len(sorted_confidence_dict)):
    if (sorted_confidence_dict[confidence_dict_item_ptr][1] <= min_truncation_confidence):
        min_truncation_particle_num = confidence_dict_item_ptr + 1
        break
for confidence_dict_item_ptr in range(len(sorted_confidence_dict)):
    if (sorted_confidence_dict[confidence_dict_item_ptr][1] <= max_truncation_confidence):
        max_truncation_particle_num = confidence_dict_item_ptr
        break
if (max_truncation_particle_num < parameters['min_refine_particle_num']):
    if (len(sorted_confidence_dict) < parameters['min_refine_particle_num']):
        max_truncation_particle_num = len(sorted_confidence_dict)
    else:
        max_truncation_particle_num = parameters['min_refine_particle_num']
if (max_truncation_particle_num > parameters['max_refine_particle_num']):
    max_truncation_particle_num = parameters['max_refine_particle_num']
if (min_truncation_particle_num >= max_truncation_particle_num):
    min_truncation_particle_num = (int)(math.floor(max_truncation_particle_num / 2.0))

refine_iteration_count = 0
refine_particle_number_step = (int)(math.floor(1.0 * (max_truncation_particle_num - min_truncation_particle_num) / (refine_grid_num_in_each_turn + 1)))
refine_particle_number_truncation_points = [(i * refine_particle_number_step + min_truncation_particle_num) for i in range(1, refine_grid_num_in_each_turn + 1)]
last_turn_chosen_job = None
while (get_iteration_flag(refine_iteration_count, refine_particle_number_step)):
    # 通过external job筛选出几批particle做refine
    '''Select several batches of particles through external job for refine'''
    # refine_external_select_particles_jobs_parents用于存储生成的external selection job的uid和对应的输入job uid，dict类型，key为job uid，value为{'particle': source_particle_job}
    # refine_external_select_particles_jobs_truncation_points用于存储生成的external selection job的uid和对应的truncation_point，dict类型，key为job uid，value为truncation_point
    refine_external_select_particles_jobs_parents = {}
    refine_external_select_particles_jobs_truncation_points = {}
    if if_low_cache_mode:
        # refine_restack_particles_jobs_parents用于存储生成的restack particles job的uid和对应的输入job uid，dict类型，key为job uid，value为{'particle': source_particle_job}
        # refine_restack_particles_jobs_truncation_points用于存储生成的restack particles job的uid和对应的truncation_point，dict类型，key为job uid，value为truncation_point
        refine_restack_particles_jobs_parents = {}
        refine_restack_particles_jobs_truncation_points = {}
    for i in range(len(refine_particle_number_truncation_points)):
        truncation_point = refine_particle_number_truncation_points[i]
        new_external_select_particles_job = CreateExternal_SelectParticles(dealjobs, getting_confidence_job, 'particles_with_confidence', confidence_dict, truncation_point, None, None, None)
        refine_external_select_particles_jobs_parents[new_external_select_particles_job.uid] = {'particle': getting_confidence_job}
        refine_external_select_particles_jobs_truncation_points[new_external_select_particles_job.uid] = truncation_point
        if if_low_cache_mode:
            new_restack_particles_job = restack_particles_class.QueueRestackJob(restack_particles_parameters, [new_external_select_particles_job.uid], ['particles_selected'])
            print('Restack particles job (', new_restack_particles_job.uid, ') has been queued, please wait for job to complete...', flush=True)
            refine_restack_particles_jobs_parents[new_restack_particles_job.uid] = {'particle': [new_external_select_particles_job.uid]}
            refine_restack_particles_jobs_truncation_points[new_restack_particles_job.uid] = truncation_point
    if if_low_cache_mode:
        for key, value in refine_restack_particles_jobs_parents.items():
            wait_for_done_with_auto_restart(dealjobs, key, True, max_trying_time)
    refine_external_select_particles_jobs_parents_list.append(refine_external_select_particles_jobs_parents)
    refine_external_select_particles_jobs_truncation_points_list.append(refine_external_select_particles_jobs_truncation_points)
    if if_low_cache_mode:
        refine_restack_particles_jobs_parents_list.append(refine_restack_particles_jobs_parents)
        refine_restack_particles_jobs_truncation_points_list.append(refine_restack_particles_jobs_truncation_points)
    mytoolbox.savetojson(refine_external_select_particles_jobs_parents_list, globaldir + 'metadata/refine_external_select_particles_jobs_parents_list.json', False)
    mytoolbox.savetojson(refine_external_select_particles_jobs_truncation_points_list, globaldir + 'metadata/refine_external_select_particles_jobs_truncation_points_list.json', False)
    if if_low_cache_mode:
        mytoolbox.savetojson(refine_restack_particles_jobs_parents_list, globaldir + 'metadata/refine_restack_particles_jobs_parents_list.json', False)
        mytoolbox.savetojson(refine_restack_particles_jobs_truncation_points_list, globaldir + 'metadata/refine_restack_particles_jobs_truncation_points_list.json', False)
    print('All iteration', refine_iteration_count, 'external selection jobs completed!', flush=True)

    # refine_nurefine_jobs_parents用于存储第二轮生成的nu-refine job的uid和对应的输入job uid，dict类型，key为nu-refine job uid，value为{'particle': external select particles job uid, 'volume': best nu-refine job uid}
    # refine_nurefine_resolutions用于存储生成的nu-refine job的uid和对应的final resolution，dict类型，key为nu-refine job uid，value为resolution
    refine_nurefine_jobs_parents = {}
    refine_nurefine_jobs_resolutions = {}
    if if_initial_orientation_balance:
        # refine_nurefine_jobs_orientation_info用于存储生成的nu-refine job的uid和对应的cfar和scf值，dict类型，key为nu-refine job uid，value为{'cfar': cfar, 'scf': scf}
        # refine_orientation_jobs_parents用于存储生成的orientation diagnostics job的uid和对应的输入job uid，dict类型，key为orientation diagnostics job uid，value为{'volume': nu-refine job uid, 'particle': [nu-refine job uid]}
        refine_nurefine_jobs_orientation_info = {}
        refine_orientation_jobs_parents = {}
    if if_low_cache_mode:
        for key, value in refine_restack_particles_jobs_truncation_points.items():
            new_nurefine_job = nurefine_class.QueueNonuniformRefineJob(second_nurefine_parameters, [key], ['particles'], best_init_job_uid,  'volume')
            if not if_safe_mode:
                dealjobs.ClearJobSafely([new_nurefine_job.uid])
                multithreadpool.setthread(gpupools.AutoQueueJob, job=new_nurefine_job, gpus_num_needed=1)
            print('NU-refine job (', new_nurefine_job.uid, ') has been created, please wait for job to complete...', flush=True)
            refine_nurefine_jobs_parents[new_nurefine_job.uid] = {'particle': [key], 'volume': best_init_job_uid}
    else:
        for key, value in refine_external_select_particles_jobs_truncation_points.items():
            new_nurefine_job = nurefine_class.QueueNonuniformRefineJob(second_nurefine_parameters, [key], ['particles_selected'], best_init_job_uid,  'volume')
            if not if_safe_mode:
                dealjobs.ClearJobSafely([new_nurefine_job.uid])
                multithreadpool.setthread(gpupools.AutoQueueJob, job=new_nurefine_job, gpus_num_needed=1, max_trying_time=max_trying_time)
            print('NU-refine job (', new_nurefine_job.uid, ') has been created, please wait for job to complete...', flush=True)
            refine_nurefine_jobs_parents[new_nurefine_job.uid] = {'particle': [key], 'volume': best_init_job_uid}
    for key, _ in refine_nurefine_jobs_parents.items():
        wait_for_done_with_auto_restart(dealjobs, key, if_safe_mode, max_trying_time)
        if if_initial_orientation_balance:
            # 创建OrientationDiagnostics job
            '''Create OrientationDiagnostics job'''
            new_orientation_job = orientation_diagnostics_class.QueueOrientationDiagnosticsJob(orientation_diagnostics_parameters, key, 'volume', [key], ['particles'])
            if not if_safe_mode:
                dealjobs.ClearJobSafely([new_orientation_job.uid])
                multithreadpool.setthread(gpupools.AutoQueueJob, job=new_orientation_job, gpus_num_needed=1)
            wait_for_done_with_auto_restart(dealjobs, new_orientation_job.uid, if_safe_mode, max_trying_time)
            refine_orientation_jobs_parents[new_orientation_job.uid] = {'volume': key, 'particle': [key]}
            # 获取信息
            cfar, scf = dealjobs.GetOrientationDiagnosticsCFARandSCF(new_orientation_job.uid)
            refine_nurefine_jobs_orientation_info[key] = {'cfar': cfar, 'scf': scf}
        refine_nurefine_jobs_resolutions[key] = dealjobs.GetNURefineFinalResolution(key)
    refine_nurefine_jobs_parents_list.append(refine_nurefine_jobs_parents)
    refine_nurefine_jobs_resolutions_list.append(refine_nurefine_jobs_resolutions)
    if if_initial_orientation_balance:
        refine_nurefine_jobs_orientation_info_list.append(refine_nurefine_jobs_orientation_info)
        refine_orientation_jobs_parents_list.append(refine_orientation_jobs_parents)
    mytoolbox.savetojson(refine_nurefine_jobs_parents_list, globaldir + 'metadata/refine_nurefine_jobs_parents_list.json', False)
    mytoolbox.savetojson(refine_nurefine_jobs_resolutions_list, globaldir + 'metadata/refine_nurefine_jobs_resolutions_list.json', False)
    if if_initial_orientation_balance:
        mytoolbox.savetojson(refine_nurefine_jobs_orientation_info_list, globaldir + 'metadata/refine_nurefine_jobs_orientation_info_list.json', False)
        mytoolbox.savetojson(refine_orientation_jobs_parents_list, globaldir + 'metadata/refine_orientation_jobs_parents_list.json', False)
    print('All iteration', refine_iteration_count, 'nu-refine jobs completed!', flush=True)

    # 从本轮nu-refine jobs中选出分辨率最高的job，在这个nu-refine job的particle number附近重新采样
    # Select the job with the highest resolution from the current nu-refine jobs, and resample near the particle number of this nu-refine job


    sorted_nurefine_jobs_resolutions = sorted(refine_nurefine_jobs_resolutions.items(), key=lambda x: x[1])
    if ((last_turn_chosen_job is not None) and (last_turn_chosen_job['best_resolution'] < sorted_nurefine_jobs_resolutions[0][1])):
        best_resolution = last_turn_chosen_job['best_resolution']
        best_refine_nurefine_job_uid = last_turn_chosen_job['best_refine_nurefine_job_uid']
        best_refine_external_job_uid = last_turn_chosen_job['best_refine_external_job_uid']
        truncation_point = last_turn_chosen_job['truncation_point']
    else:
        best_resolution = sorted_nurefine_jobs_resolutions[0][1]
        best_refine_nurefine_job_uid = sorted_nurefine_jobs_resolutions[0][0]
        best_refine_external_job_uid = refine_nurefine_jobs_parents[best_refine_nurefine_job_uid]['particle'][0]
        if if_low_cache_mode:
            truncation_point = refine_restack_particles_jobs_truncation_points[best_refine_external_job_uid]
        else:
            truncation_point = refine_external_select_particles_jobs_truncation_points[best_refine_external_job_uid]
    refine_iteration_count += 1
    refine_particle_number_step = refine_particle_number_step / ((refine_grid_num_in_each_turn / 2.0) + 1.0)
    refine_particle_number_truncation_points = [(int)(math.floor(truncation_point - (refine_particle_number_step * (refine_grid_num_in_each_turn / 2.0)) + (i * refine_particle_number_step))) for i in range(refine_grid_num_in_each_turn + 1)]
    refine_particle_number_truncation_points = refine_particle_number_truncation_points[:(int)(math.floor(refine_grid_num_in_each_turn / 2.0))] + refine_particle_number_truncation_points[((int)(math.floor(refine_grid_num_in_each_turn / 2.0)) + 1):]

    last_turn_chosen_job = {'best_refine_nurefine_job_uid': best_refine_nurefine_job_uid, 'best_refine_external_job_uid': best_refine_external_job_uid, 'best_resolution': best_resolution, 'truncation_point': truncation_point}



############################################## plot curve, find best volume #################################################



# 通过最佳finel refine job绘制confidence-resolution curve
'''Draw the confidence-resolution curve through the best finel refine job'''
# sorted_final_nurefine_jobs_resolutions = sorted(final_nurefine_jobs_resolutions.items(), key=lambda x: x[1])
# best_final_nurefine_job_uid = sorted_final_nurefine_jobs_resolutions[0][0] if (sorted_final_nurefine_jobs_resolutions[0][1] <= sorted_second_nurefine_jobs_resolutions[0][1]) else sorted_second_nurefine_jobs_resolutions[0][0]
best_final_nurefine_job_uid = best_init_job_uid
best_final_nurefine_resolution = best_init_job_resolution
whole_nurefine_jobs_resolutions_curvedata = []
for i in range(len(refine_nurefine_jobs_parents_list)):
    for key, value in refine_nurefine_jobs_resolutions_list[i].items():
        if ((if_initial_orientation_balance) and (refine_nurefine_jobs_orientation_info_list[i][key]['cfar'] < cfar_lower_bound)):
            continue
        if if_low_cache_mode:
            restack_particles_job_uid = refine_nurefine_jobs_parents_list[i][key]['particle'][0]
            particle_number_point = refine_restack_particles_jobs_truncation_points_list[i][restack_particles_job_uid]
        else:
            external_select_particles_job_uid = refine_nurefine_jobs_parents_list[i][key]['particle'][0]
            particle_number_point = refine_external_select_particles_jobs_truncation_points_list[i][external_select_particles_job_uid]
        whole_nurefine_jobs_resolutions_curvedata.append([particle_number_point, value])
        if (value < best_final_nurefine_resolution):
            best_final_nurefine_job_uid = key
            best_final_nurefine_resolution = value
print('Curve data length:', len(whole_nurefine_jobs_resolutions_curvedata), flush=True)
new_external_plot_confidence_curve_job = CreateExternal_PlotConfidenceCurve(dealjobs, best_final_nurefine_job_uid, 'particles', whole_nurefine_jobs_resolutions_curvedata, confidence_dict)
print('Best refine job uid:', best_final_nurefine_job_uid, ', resolution:', best_final_nurefine_resolution, flush=True)
curve_job_parents = {'best_final_nurefine_job_uid': best_final_nurefine_job_uid}
mytoolbox.savetojson(curve_job_parents, globaldir + 'metadata/curve_job_parents.json', False)



wholetimeend = datetime.datetime.now()
print('Select and refine finished! Time spent:', (wholetimeend - wholetimebegin), flush=True)