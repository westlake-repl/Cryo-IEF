#!/usr/bin/env python

import datetime
import numpy as np
import time
import os

import sys
sys.path.append(os.path.dirname(__file__))

import cryowizardlib.CSLogin as CSLogin
import cryowizardlib.MyJobAPIs as MyJobAPIs
import cryowizardlib.mytoolbox as mytoolbox





def ImportAndExtract(project_dir):
    print('Import and extract start...', flush=True)
    wholetimebegin = datetime.datetime.now()

    # 初始化需要的内容
    '''Initialize the necessary content'''
    if not (project_dir[-1] == '/'):
        globaldir = project_dir + '/'
    else:
        globaldir = project_dir
    if not os.path.exists(globaldir + 'metadata/'):
        os.makedirs(globaldir + 'metadata/')
    parameters = mytoolbox.readjson(globaldir + 'parameters/parameters.json')
    if_safe_mode = parameters['safe_mode']
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

    # 先从movie或者micrograph中extract particles，并与输入的particles一起整合获取输入particle文件
    '''First extract particles from movie or micrograph, and integrate with the input particles to obtain the input particle file'''
    # whole_particles_list用于储存所有获得的particles job，list类型，每一个item均为一个字典：{'source_particle_job': source particle job uid, 'source_particle_parameter_name': source particle parameter name}
    '''whole_particles_list is used to store all the particles jobs obtained, list type, each item is a dictionary: {'source_particle_job': source particle job uid, 'source_particle_parameter_name': source particle parameter name}'''
    whole_particles_list = []
    # import_movies_jobs_parents用于存储生成的import movies job的uid和对应的parameters folder，dict类型，key为import movies job uid，value为{'folder': import movies parameters folder}
    '''import_movies_jobs_parents is used to store the uid of the generated import movies job and the corresponding parameters folder, dict type, key is the import movies job uid, value is {'folder': import movies parameters folder}'''
    import_movies_jobs_parents = {}
    import_movies_class = MyJobAPIs.CreateImportMovies(dealjobs)
    # import_micrographs_jobs_parents用于存储生成的import_micrographs job的uid和对应的parameters folder，dict类型，key为import_micrographs job uid，value为{'folder': import micrographs parameters folder}
    '''import_micrographs_jobs_parents is used to store the uid of the generated import micrographs job and the corresponding parameters folder, dict type, key is the import micrographs job uid, value is {'folder': import micrographs parameters folder}'''
    import_micrographs_jobs_parents = {}
    import_micrographs_class = MyJobAPIs.CreateImportMicrographs(dealjobs)
    # motion_correction_jobs_parents用于存储生成的motion_correction job的uid和对应的输入job uid以及对应的parameters folder，dict类型，key为motion_correction job uid，value为{'movies': import movies job uid, 'folder': import movies parameters folder}
    '''motion_correction_jobs_parents is used to store the uid of the generated motion_correction job and the corresponding input job uid and the corresponding parameters folder, dict type, key is the motion_correction job uid, value is {'movies': import movies job uid, 'folder': import movies parameters folder}'''
    motion_correction_jobs_parents = {}
    motion_correction_class = MyJobAPIs.CreatePatchMotionCorrection(dealjobs)
    # ctf_estimation_jobs_parents用于存储生成的ctf_estimation job的uid和对应的输入job uid以及对应的parameters folder，dict类型，key为ctf_estimation job uid，value为{'micrographs': job uid, 'folder': parameters folder}
    '''ctf_estimation_jobs_parents is used to store the uid of the generated ctf_estimation job and the corresponding input job uid and the corresponding parameters folder, dict type, key is the ctf_estimation job uid, value is {'micrographs': job uid, 'folder': parameters folder}'''
    ctf_estimation_jobs_parents = {}
    ctf_estimation_class = MyJobAPIs.CreatePatchCtfEstimation(dealjobs)
    # blob_pick_jobs_parents用于存储生成的blob_pick job的uid和对应的输入job uid以及对应的parameters folder，dict类型，key为blob_pick job uid，value为{'micrographs': job uid, 'folder': parameters folder}
    '''blob_pick_jobs_parents is used to store the uid of the generated blob_pick job and the corresponding input job uid and the corresponding parameters folder, dict type, key is the blob_pick job uid, value is {'micrographs': job uid, 'folder': parameters folder}'''
    blob_pick_jobs_parents = {}
    blob_pick_class = MyJobAPIs.CreateBlobPicker(dealjobs)
    # extract_jobs_parents用于存储生成的extract job的uid和对应的输入job uid以及对应的parameters folder，dict类型，key为extract job uid，value为{'particles': job uid', micrographs': job uid, 'folder': parameters folder}
    '''extract_jobs_parents is used to store the uid of the generated extract job and the corresponding input job uid and the corresponding parameters folder, dict type, key is the extract job uid, value is {'particles': job uid', micrographs': job uid, 'folder': parameters folder}'''
    extract_jobs_parents = {}
    extract_class = MyJobAPIs.CreateExtractMicrographs(dealjobs)
    parameters_files_list = os.listdir(globaldir + 'parameters/')
    jobtype_num_count = [0, 0, 0]
    for parameter_item in parameters_files_list:
        if ((len(parameter_item) < 18) or os.path.isfile(globaldir + 'parameters/' + parameter_item)):
            continue
        if not (parameter_item[:18] == 'import_parameters_'):
            continue
        source_input_parameters_folder = globaldir + 'parameters/' + parameter_item + '/'
        source_input_type = mytoolbox.readjson(source_input_parameters_folder + 'input_type.json')
        if (source_input_type == 'Movies'):
            # import movie，motion correction，ctf estimation，blob pick，extract，获得particles
            '''import movie, motion correction, ctf estimation, blob pick, extract, get particles'''
            jobtype_num_count[0] += 1

            # 创建import movies job
            '''Create import movies job'''
            import_movies_parameters = mytoolbox.readjson(source_input_parameters_folder + 'import_movies_parameters.json')
            new_import_movies_job = import_movies_class.QueueImportMoviesJob(import_movies_parameters)
            print('Import movies job (', new_import_movies_job.uid, ') has been queued, please wait for job to complete...', flush=True)
            import_movies_jobs_parents[new_import_movies_job.uid] = {'parameterpath': source_input_parameters_folder + 'import_movies_parameters.json'}

            # 通过import movies job创建motion correction job
            '''Create motion correction job through import movies job'''
            motion_correction_parameters = mytoolbox.readjson(source_input_parameters_folder + 'patch_motion_correction_parameters.json')
            new_motion_correction_job = motion_correction_class.QueuePatchMotionCorrectionJob(motion_correction_parameters, [new_import_movies_job.uid], ['imported_movies'])
            motion_correction_jobs_parents[new_motion_correction_job.uid] = {'movies': [new_import_movies_job.uid], 'parameterpath': source_input_parameters_folder + 'patch_motion_correction_parameters.json'}
            if not if_safe_mode:
                print('Motion correction job (', new_motion_correction_job.uid, ') has been created, please wait for job to complete...', flush=True)
                new_motion_correction_job.clear()
            else:
                print('Motion correction job (', new_motion_correction_job.uid, ') has been queued, please wait for job to complete...', flush=True)

            # 通过motion correction job创建ctf estimation job
            '''Create ctf estimation job through motion correction job'''
            ctf_estimation_parameters = mytoolbox.readjson(source_input_parameters_folder + 'patch_ctf_estimation_parameters.json')
            new_ctf_estimation_job = ctf_estimation_class.QueuePatchCtfEstimationJob(ctf_estimation_parameters, [new_motion_correction_job.uid], ['micrographs'])
            ctf_estimation_jobs_parents[new_ctf_estimation_job.uid] = {'micrographs': [new_motion_correction_job.uid], 'parameterpath': source_input_parameters_folder + 'patch_ctf_estimation_parameters.json'}
            if not if_safe_mode:
                print('CTF estimation job (', new_ctf_estimation_job.uid, ') has been created, please wait for job to complete...', flush=True)
                new_ctf_estimation_job.clear()
            else:
                print('CTF estimation job (', new_ctf_estimation_job.uid, ') has been queued, please wait for job to complete...', flush=True)

            # 通过ctf estimation job创建blob pick job
            '''Create blob pick job through ctf estimation job'''
            blob_pick_parameters = mytoolbox.readjson(source_input_parameters_folder + 'blob_picker_parameters.json')
            new_blob_pick_job = blob_pick_class.QueueBlobPickerJob(blob_pick_parameters, [new_ctf_estimation_job.uid], ['exposures'])
            blob_pick_jobs_parents[new_blob_pick_job.uid] = {'micrographs': [new_ctf_estimation_job.uid], 'parameterpath': source_input_parameters_folder + 'blob_picker_parameters.json'}
            if not if_safe_mode:
                print('Blob pick job (', new_blob_pick_job.uid, ') has been created, please wait for job to complete...', flush=True)
                new_blob_pick_job.clear()
            else:
                print('Blob pick job (', new_blob_pick_job.uid, ') has been queued, please wait for job to complete...', flush=True)

            # 通过blob pick job创建extract job
            '''Create extract job through blob pick job'''
            extract_parameters = mytoolbox.readjson(source_input_parameters_folder + 'extract_micrographs_parameters.json')
            new_extract_job = extract_class.QueueExtractMicrographsJob(extract_parameters, [new_blob_pick_job.uid], ['micrographs'], [new_blob_pick_job.uid], ['particles'])
            extract_jobs_parents[new_extract_job.uid] = {'particles': [new_blob_pick_job.uid], 'micrographs': [new_blob_pick_job.uid], 'parameterpath': source_input_parameters_folder + 'extract_micrographs_parameters.json'}
            if not if_safe_mode:
                print('Extract job (', new_extract_job.uid, ') has been created, please wait for job to complete...', flush=True)
                new_extract_job.clear()
            else:
                print('Extract job (', new_extract_job.uid, ') has been queued, please wait for job to complete...', flush=True)

            whole_particles_list.append({'source_particle_job': new_extract_job.uid, 'source_particle_parameter_name': 'particles'})

        elif (source_input_type == 'Micrographs'):
            # import micrographs，ctf estimation，blob pick，extract，获得particles
            '''import micrographs, ctf estimation, blob pick, extract, get particles'''
            jobtype_num_count[1] += 1

            # 创建import micrographs job
            '''Create import micrographs job'''
            import_micrographs_parameters = mytoolbox.readjson(source_input_parameters_folder + 'import_micrographs_parameters.json')
            new_import_micrographs_job = import_micrographs_class.QueueImportMicrographsJob(import_micrographs_parameters)
            print('Import micrographs job (', new_import_micrographs_job.uid, ') has been queued, please wait for job to complete...', flush=True)
            import_micrographs_jobs_parents[new_import_micrographs_job.uid] = {'parameterpath': source_input_parameters_folder + 'import_micrographs_parameters.json'}

            # 通过import micrographs job创建ctf estimation job
            '''Create ctf estimation job through import micrographs job'''
            ctf_estimation_parameters = mytoolbox.readjson(source_input_parameters_folder + 'patch_ctf_estimation_parameters.json')
            new_ctf_estimation_job = ctf_estimation_class.QueuePatchCtfEstimationJob(ctf_estimation_parameters, [new_import_micrographs_job.uid], ['imported_micrographs'])
            ctf_estimation_jobs_parents[new_ctf_estimation_job.uid] = {'micrographs': [new_import_micrographs_job.uid], 'parameterpath': source_input_parameters_folder + 'patch_ctf_estimation_parameters.json'}
            if not if_safe_mode:
                print('CTF estimation job (', new_ctf_estimation_job.uid, ') has been created, please wait for job to complete...', flush=True)
                new_ctf_estimation_job.clear()
            else:
                print('CTF estimation job (', new_ctf_estimation_job.uid, ') has been queued, please wait for job to complete...', flush=True)

            # 通过ctf estimation job创建blob pick job
            '''Create blob pick job through ctf estimation job'''
            blob_pick_parameters = mytoolbox.readjson(source_input_parameters_folder + 'blob_picker_parameters.json')
            new_blob_pick_job = blob_pick_class.QueueBlobPickerJob(blob_pick_parameters, [new_ctf_estimation_job.uid], ['exposures'])
            blob_pick_jobs_parents[new_blob_pick_job.uid] = {'micrographs': [new_ctf_estimation_job.uid], 'parameterpath': source_input_parameters_folder + 'blob_picker_parameters.json'}
            if not if_safe_mode:
                print('Blob pick job (', new_blob_pick_job.uid, ') has been created, please wait for job to complete...', flush=True)
                new_blob_pick_job.clear()
            else:
                print('Blob pick job (', new_blob_pick_job.uid, ') has been queued, please wait for job to complete...', flush=True)

            # 通过blob pick job创建extract job
            '''Create extract job through blob pick job'''
            extract_parameters = mytoolbox.readjson(source_input_parameters_folder + 'extract_micrographs_parameters.json')
            new_extract_job = extract_class.QueueExtractMicrographsJob(extract_parameters, [new_blob_pick_job.uid], ['micrographs'], [new_blob_pick_job.uid], ['particles'])
            extract_jobs_parents[new_extract_job.uid] = {'particles': [new_blob_pick_job.uid], 'micrographs': [new_blob_pick_job.uid], 'parameterpath': source_input_parameters_folder + 'extract_micrographs_parameters.json'}
            if not if_safe_mode:
                print('Extract job (', new_extract_job.uid, ') has been created, please wait for job to complete...', flush=True)
                new_extract_job.clear()
            else:
                print('Extract job (', new_extract_job.uid, ') has been queued, please wait for job to complete...', flush=True)

            whole_particles_list.append({'source_particle_job': new_extract_job.uid, 'source_particle_parameter_name': 'particles'})

        elif (source_input_type == 'Particles'):
            # 识别输入的particle job的类别，支持的particle job有：Import Particle Stack, Extract From Micrographs(Multi-GPU), Restack Particles
            '''Identify the category of the input particle job, supported particle jobs are: Import Particle Stack, Extract From Micrographs(Multi-GPU), Restack Particles'''
            jobtype_num_count[2] += 1
            source_particle_jobuid = mytoolbox.readjson(source_input_parameters_folder + 'particle_job_uid.json')['source_particle_job_uid']
            files_in_job_dir = os.listdir((str)(dealjobs.cshandle.find_project(dealjobs.project).dir()) + '/' + source_particle_jobuid)
            if 'imported_particles.cs' in files_in_job_dir:
                source_particle_parameter_name = 'imported_particles'
            elif 'extracted_particles.cs' in files_in_job_dir:
                source_particle_parameter_name = 'particles'
            elif 'restacked_particles.cs' in files_in_job_dir:
                source_particle_parameter_name = 'particles'
            else:
                print('Job type error...', flush=True)
                exit()

            whole_particles_list.append({'source_particle_job': source_particle_jobuid, 'source_particle_parameter_name': source_particle_parameter_name})

        else:
            print('Input type error...', flush=True)
            exit()
    mytoolbox.savetojson(whole_particles_list, globaldir + 'metadata/whole_particles_list.json', False)
    mytoolbox.savetojson(import_movies_jobs_parents, globaldir + 'metadata/import_movies_jobs_parents.json', False)
    mytoolbox.savetojson(import_micrographs_jobs_parents, globaldir + 'metadata/import_micrographs_jobs_parents.json', False)
    mytoolbox.savetojson(motion_correction_jobs_parents, globaldir + 'metadata/motion_correction_jobs_parents.json', False)
    mytoolbox.savetojson(ctf_estimation_jobs_parents, globaldir + 'metadata/ctf_estimation_jobs_parents.json', False)
    mytoolbox.savetojson(blob_pick_jobs_parents, globaldir + 'metadata/blob_pick_jobs_parents.json', False)
    mytoolbox.savetojson(extract_jobs_parents, globaldir + 'metadata/extract_jobs_parents.json', False)

    if not if_safe_mode:
        whole_gpu_job_info = []
        for item in motion_correction_jobs_parents.items():
            whole_gpu_job_info.append(item)
        for item in ctf_estimation_jobs_parents.items():
            whole_gpu_job_info.append(item)
        for item in blob_pick_jobs_parents.items():
            whole_gpu_job_info.append(item)
        for item in extract_jobs_parents.items():
            whole_gpu_job_info.append(item)

        while True:
            whole_job_all_complete_flag = True

            for jobuid, parents_info in whole_gpu_job_info:
                job = dealjobs.cshandle.find_job(dealjobs.project, jobuid)
                if job.status in ['completed', 'killed', 'failed']:
                    continue
                elif job.status in ['queued', 'launched', 'started', 'running', 'waiting']:
                    whole_job_all_complete_flag = False
                    continue
                else:
                    whole_job_all_complete_flag = False

                    parent_jobuid_list = []
                    for key, value in parents_info.items():
                        if key == 'parameterpath':
                            source_input_parameter_path = value
                        else:
                            if (type(value) == type([])):
                                parent_jobuid_list += value
                            elif (type(value) == type('')):
                                parent_jobuid_list.append(value)
                    parent_jobuid_list = list(set(parent_jobuid_list))
                    parent_job_all_complete_flag = True
                    for parent_jobuid in parent_jobuid_list:
                        parent_job = dealjobs.cshandle.find_job(dealjobs.project, parent_jobuid)
                        if not (parent_job.status == 'completed'):
                            parent_job_all_complete_flag = False
                            break

                    if parent_job_all_complete_flag:
                        parameter_filename = os.path.basename(parents_info['parameterpath'])
                        if (parameter_filename == 'patch_motion_correction_parameters.json'):
                            gpunum = mytoolbox.readjson(parents_info['parameterpath'])['Number of GPUs to parallelize']
                            jobtype = 'Motion correction'
                        elif (parameter_filename == 'patch_ctf_estimation_parameters.json'):
                            gpunum = mytoolbox.readjson(parents_info['parameterpath'])['Number of GPUs to parallelize']
                            jobtype = 'CTF estimation'
                        elif (parameter_filename == 'blob_picker_parameters.json'):
                            gpunum = 1
                            jobtype = 'Blob pick'
                        elif (parameter_filename == 'extract_micrographs_parameters.json'):
                            gpunum = mytoolbox.readjson(parents_info['parameterpath'])[
                                'Number of GPUs to parallelize (0 for CPU-only)']
                            jobtype = 'Extract'
                        multithreadpool.setthread(gpupools.AutoQueueJob, job=job, gpus_num_needed=gpunum)
                        print(jobtype + ' job (', job.uid, ') has been queued, please wait for job to complete...',
                              flush=True)

            if whole_job_all_complete_flag:
                break
            else:
                time.sleep(30)

        print('All particles prepared!', flush=True)

    if (jobtype_num_count == [0, 0, 1]):
        mytoolbox.savetojson(whole_particles_list[0]['source_particle_job'], globaldir + 'metadata/particle_job_uid.json', False)
    else:
        particle_sets_tool_source_particle_job_list = []
        particle_sets_tool_source_particle_parameter_name_list = []
        for item in whole_particles_list:
            particle_sets_tool_source_particle_job_list.append(item['source_particle_job'])
            particle_sets_tool_source_particle_parameter_name_list.append(item['source_particle_parameter_name'])
        particle_sets_tool_class = MyJobAPIs.CreateParticleSetsTool(dealjobs)
        particle_sets_tool_parameters = particle_sets_tool_class.default_particle_sets_tool_job_parameters
        particle_sets_tool_parameters['Split num. batches'] = 1
        if parameters['max_particle_num_to_use'] is not None:
            particle_sets_tool_parameters['Split batch size'] = parameters['max_particle_num_to_use']
        particle_sets_tool_parameters['Split randomize'] = True
        new_particle_sets_tool_job = particle_sets_tool_class.QueueParticleSetsToolJob(particle_sets_tool_parameters, particle_sets_tool_source_particle_job_list, particle_sets_tool_source_particle_parameter_name_list)
        print('Particle sets tool job (', new_particle_sets_tool_job.uid, ') has been queued, please wait for job to complete...', flush=True)
        new_particle_sets_tool_job.wait_for_done()

        # restack_particles_class = MyJobAPIs.CreateRestackParticles(dealjobs)
        # restack_particles_parameters = restack_particles_class.default_restack_particles_job_parameters
        # new_restack_particles_job = restack_particles_class.QueueRestackJob(restack_particles_parameters, [new_particle_sets_tool_job.uid], ['split_0'])
        # print('Restack particles job (', new_restack_particles_job.uid, ') has been queued, please wait for job to complete...', flush=True)
        # new_restack_particles_job.wait_for_done()

        mytoolbox.savetojson(new_particle_sets_tool_job.uid, globaldir + 'metadata/particle_job_uid.json', False)

    wholetimeend = datetime.datetime.now()
    print('Import and extract finished! Time spent:', (wholetimeend - wholetimebegin), flush=True)



def TruncateParticles(project_dir, truncation_type, particle_cutoff_condition):
    if not (project_dir[-1] == '/'):
        globaldir = project_dir + '/'
    else:
        globaldir = project_dir
    parameters = mytoolbox.readjson(globaldir + 'parameters/parameters.json')
    cshandle = CSLogin.cshandleclass.GetCryoSPARCHandle(email=parameters['cryosparc_username'], password=parameters['cryosparc_password'])
    dealjobs = MyJobAPIs.DealJobs(cshandle, parameters['project'], parameters['workspace'], parameters['lane'])

    def TruncateParticles(truncation_type, particle_cutoff_condition, DealJobs_instance):
        globaldir = os.getcwd()
        if not (globaldir[-1] == '/'):
            globaldir = globaldir + '/'

        source_particle_job = mytoolbox.readjson(globaldir + 'metadata/external_get_confidence_jobuid.json')['external_get_confidence_jobuid']
        confidence_dict = mytoolbox.readjson(globaldir + 'metadata/confidence_dict.json')

        # 创建空白 external job
        '''Create an empty external job'''
        project = DealJobs_instance.cshandle.find_project(DealJobs_instance.project)
        job = project.create_external_job(DealJobs_instance.workspace, title='Truncate Particles')

        # 创建和链接输入
        '''Create and link input'''
        add_input_name = 'input_particles'
        job.add_input(type='particle', name=add_input_name, min=1, slots=['blob'], title='Input particles for selection')
        job.connect(target_input=add_input_name, source_job_uid=source_particle_job, source_output='particles_with_confidence')

        # 获取输入的cs信息，包含add_input()函数的slots参数指定的所有内容
        '''Get the cs information of the input, including all contents specified by the slots parameter of the add_input() function'''
        input_particles_dataset = job.load_input(name=add_input_name)

        with job.run():
            selected_uids = []
            sorted_confidence_dict = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
            if (truncation_type == 'num'):
                safe_particle_number = (int)(particle_cutoff_condition) if ((int)(particle_cutoff_condition) <= len(sorted_confidence_dict)) else len(sorted_confidence_dict)
                for confidence_dict_item_ptr in range(safe_particle_number):
                    selected_uids.append((int)(sorted_confidence_dict[confidence_dict_item_ptr][0]))
            elif (truncation_type == 'score'):
                safe_particle_score = np.clip((float)(particle_cutoff_condition), -1.0, 1.0)
                for confidence_dict_item_ptr in range(len(sorted_confidence_dict)):
                    if (sorted_confidence_dict[confidence_dict_item_ptr][1] < safe_particle_score):
                        break
                    selected_uids.append((int)(sorted_confidence_dict[confidence_dict_item_ptr][0]))

            if (len(selected_uids) <= 0):
                job.log('Warning: no particle picked, please set another truncation value...')

            output_particles_dataset = input_particles_dataset.query({'uid': selected_uids})

            # 创建输出，并保存本job的cs信息
            '''Create output and save the cs information of this job'''
            # cs里修改过的slot需要在add_output()函数的slots参数中全部指定来用于保存为新的，剩下的内容passthrough参数会自动从input中拉取并保存在xxx_passthrough.cs中
            '''Slots that have been modified in cs need to be specified in the slots parameter of the add_output() function to be used as new, and the remaining contents will be automatically pulled from the input and saved in xxx_passthrough.cs'''
            add_output_name = 'particles_selected'
            job.add_output(type='particle', name=add_output_name, slots=['blob'], passthrough=add_input_name)
            job.save_output(add_output_name, output_particles_dataset)

        return job

    new_truncate_particles_job = TruncateParticles((str)(truncation_type), (float)(particle_cutoff_condition), dealjobs)
    print('New truncate particles job', new_truncate_particles_job.uid, 'finished', flush=True)