import threading
import time
import os

import sys
sys.path.append(os.path.dirname(__file__))

import mytoolbox



# 提供基本的处理job的方法
class DealJobs():
    def __init__(self, cshandle, project, workspace, lane):
        self.cshandle = cshandle
        self.project = project
        self.workspace = workspace
        self.lane = lane

    # 安全地clear jobs，确保成功且不出现报错
    def ClearJobSafely(self, uids_waiting_for_clear):
        # uids_waiting_for_clear是一个list，里面包含的uid为str类型
        for uid in uids_waiting_for_clear:
            job = self.cshandle.find_job(self.project, uid)
            while True:
                if (job.status == 'building'):
                    break
                if job.status in ['launched', 'started', 'running', 'waiting']:
                    job.kill()
                try:
                    job.clear()
                    job.wait_for_status('building', 5)
                    if (job.status == 'building'):
                        break
                except:
                    pass
                time.sleep(1)

    # 重启失败的作业
    def restartfailedjobs(self, uids_waiting_for_restart):
        # uids_waiting_for_restart是一个list，里面包含的uid为str类型
        restarted_uids = []
        for uid in uids_waiting_for_restart:
            job = self.cshandle.find_job(self.project, uid)
            if (job.status == 'failed'):
                job.clear()
                job.queue(self.lane)
                restarted_uids.append(uid)
        return restarted_uids

    # 获取NU-Refine Job计算的最终的分辨率
    def GetNURefineFinalResolution(self, jobuid):
        jobjson = mytoolbox.readjson((str)(self.cshandle.find_project(self.project).dir()) + '/' + jobuid + '/job.json')
        fsc_final_resolution = -1.0
        for item in jobjson['output_result_groups']:
            if (item['type'] == 'volume'):
                fsc_final_resolution = item['latest_summary_stats']['fsc_info_best']['radwn_final_A']
                break
        if (fsc_final_resolution < 0):
            print('Reading resolution failed...', flush=True)
            exit()
        else:
            return fsc_final_resolution

    # 获取Reconstruct Only Job计算的最终的分辨率
    def GetReconstructOnlyFinalResolution(self, jobuid):
        jobjson = mytoolbox.readjson((str)(self.cshandle.find_project(self.project).dir()) + '/' + jobuid + '/job.json')
        fsc_final_resolution = -1.0
        for item in jobjson['output_result_groups']:
            if (item['type'] == 'volume'):
                fsc_final_resolution = item['latest_summary_stats']['fsc_info_best']['radwn_final_A']
                break
        if (fsc_final_resolution < 0):
            print('Reading resolution failed...', flush=True)
            exit()
        else:
            return fsc_final_resolution

    # 获取Orientation Diagnostics Job计算得到的CFAR值和SCF值
    def GetOrientationDiagnosticsCFARandSCF(self, jobuid, if_need_scf=True):
        jobjson = mytoolbox.readjson((str)(self.cshandle.find_project(self.project).dir()) + '/' + jobuid + '/job.json')
        cfar = jobjson['output_result_groups'][0]['latest_summary_stats']['cfar']
        if if_need_scf:
            scf = jobjson['output_result_groups'][0]['latest_summary_stats']['scf_star']
            return cfar, scf
        else:
            return cfar, None


class HostAndGPUPools():
    # 创建一个hostname和gpu上的job计数池，用于帮助计数job和提交任务前判断资源是否足够提交新job
    def __init__(self, dealJobs_instance, hostname_list, gpu_num_list, max_num_of_jobs_list):
        # lane: str类型，标注是哪个lane上的gpu资源
        # hostname_list: list类型，list内为str类型
        # gpu_num_list: list类型，list内为int类型
        # max_num_of_jobs_list: list类型，list内为int类型
        # gpunum_list和hostnamelist以及max_num_of_jobs_list顺序一一对应，指出每个hostname下有几张卡，以及hostname下的单张卡最多放几个job
        self.dealjobtools = dealJobs_instance
        whole_gpu_info_dict = {}
        for i in range(len(hostname_list)):
            hostname = hostname_list[i]
            each_gpu_info_dict = {}
            for gpu in range(gpu_num_list[i]):
                each_gpu_info_dict[gpu] = {'running_job_num': 0, 'max_num_of_jobs': max_num_of_jobs_list[i]}
            whole_gpu_info_dict[hostname] = each_gpu_info_dict
        self.max_job_num = sum([(int)(gpu_num_list[_] * max_num_of_jobs_list[_]) for _ in range(len(gpu_num_list))])
        self.hostname_list = hostname_list
        self.whole_gpu_info_dict = whole_gpu_info_dict
        self.global_thread_lock = threading.Lock()

    def RequestGPUs(self, gpus_num_needed):
        # 判断每个hostname上的gpus上是否还能再放一个job，如果可以则在这些gpu上再放一个计数并返回申请到的hostname以及gpu id list，list内类型为int，否则返回两个None
        # 如果有多个gpu都能用，那么优先分配当前卡上正在跑的job数量最少的卡，以帮助平衡gpu占用，避免部分gpu空置
        # gpus_num_needed: int类型，表示申请的gpu数量
        with self.global_thread_lock:
            for hostname in self.hostname_list:
                gpus_list_requested = []
                for gpu, _ in self.whole_gpu_info_dict[hostname].items():
                    if (self.whole_gpu_info_dict[hostname][gpu]['running_job_num'] < self.whole_gpu_info_dict[hostname][gpu]['max_num_of_jobs']):
                        gpus_list_requested.append([gpu, self.whole_gpu_info_dict[hostname][gpu]['running_job_num']])
                if (len(gpus_list_requested) >= gpus_num_needed):
                    sorted_gpus_list_requested = sorted(gpus_list_requested, key=lambda x: x[1])
                    return_gpu_list = []
                    for gpu_ptr in range(gpus_num_needed):
                        gpu = sorted_gpus_list_requested[gpu_ptr][0]
                        self.whole_gpu_info_dict[hostname][gpu]['running_job_num'] += 1
                        return_gpu_list.append(gpu)
                    return hostname, return_gpu_list
            return None, None

    def GiveBackGPUs(self, hostname, gpus):
        # 使用完后归还资源计数
        # hostname: str类型
        # gpus: list类型，list内为int类型
        with self.global_thread_lock:
            for gpu in gpus:
                self.whole_gpu_info_dict[hostname][gpu]['running_job_num'] -= 1

    def AutoQueueJob(self, job, gpus_num_needed, max_trying_time=3):
        # 对于已创建的job（一般为building状态，failed或killed也适用）自动启动并run，待任务完成后自动归还gpu资源
        # job: cryosparc的job类
        # gpus_num_needed: 需要多少张显卡
        # max_trying_time: 如果中间有失败，自动restart，该参数指出总共尝试多少次（包含第一次）
        hostname = None
        gpus_list = None
        while ((hostname == None) or (gpus_list == None)):
            hostname, gpus_list = self.RequestGPUs(gpus_num_needed)
            time.sleep(3)
        for i in range(max_trying_time):
            self.dealjobtools.ClearJobSafely([job.uid])
            job.queue(self.dealjobtools.lane, hostname, gpus_list)
            job.wait_for_done()
            if (job.status == 'completed'):
                break
        self.GiveBackGPUs(hostname, gpus_list)




# # 此处为创建 External job 的函数模板，具体内容需要自己定制
# def CreateExternal(DealJobs_instance, input_job_list):
#
#     # 创建空白 external job
#     project = DealJobs_instance.cshandle.find_project(DealJobs_instance.project)
#     new_job = project.create_external_job(DealJobs_instance.workspace)
#
#     # 创建和链接输入
#     add_input_name = 'input_particles'
#     new_job.add_input(type='particle', name=add_input_name, min=1, slots=['blob', 'alignments3D', 'ctf'], title='Input particles for selection')
#     for inputjob in input_job_list:
#         new_job.connect(target_input=add_input_name, source_job_uid=inputjob, source_output='particles')
#
#     # 获取输入的cs信息，包含add_input()函数的slots参数指定的所有内容
#     input_particles_dataset = new_job.load_input(name=add_input_name)
#
#     # 此处运行job内部需要的内容
#     with new_job.run():
#         output_particles_dataset = input_particles_dataset.slice(345, 545)
#
#         # 创建输出，并保存本job的cs信息
#         # cs里修改过的slot需要在add_output()函数的slots参数中全部指定来用于保存为新的，剩下的内容passthrough参数会自动从input中拉取并保存在xxx_passthrough.cs中
#         add_output_name = 'particles_selected'
#         new_job.add_output(type='particle', name=add_output_name, slots=['blob'], passthrough=add_input_name)
#         new_job.save_output(add_output_name, output_particles_dataset)
#
#     return new_job.uid


class CreateImportMovies():
    def __init__(self, DealJobs_instance):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        self.dealjobtools = DealJobs_instance
        self.default_import_movies_job_parameters = mytoolbox.readjson(filedir + '/parameters/import_movies_parameters.json')

    def QueueImportMoviesJob(self, parameters):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, mytoolbox.readjson(filedir + '/parameters/import_movies_job_name.json'))

        parameters_map = mytoolbox.readjson(filedir + '/parameters/import_movies_parameters_map.json')
        for param_title in parameters:
            job.set_param(parameters_map[param_title], parameters[param_title])

        job.queue(self.dealjobtools.lane)
        return job


class CreateImportMicrographs():
    def __init__(self, DealJobs_instance):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        self.dealjobtools = DealJobs_instance
        self.default_import_micrographs_job_parameters = mytoolbox.readjson(filedir + '/parameters/import_micrographs_parameters.json')

    def QueueImportMicrographsJob(self, parameters, source_movie_job_list=None, source_movie_parameter_name_list=None):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, mytoolbox.readjson(filedir + '/parameters/import_micrographs_job_name.json'))
        if source_movie_job_list is not None:
            for i in range(len(source_movie_job_list)):
                job.connect('movies', source_movie_job_list[i], source_movie_parameter_name_list[i])

        parameters_map = mytoolbox.readjson(filedir + '/parameters/import_micrographs_parameters_map.json')
        for param_title in parameters:
            job.set_param(parameters_map[param_title], parameters[param_title])

        job.queue(self.dealjobtools.lane)
        return job


class CreatePatchMotionCorrection():
    def __init__(self, DealJobs_instance):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        self.dealjobtools = DealJobs_instance
        self.default_patch_motion_correction_job_parameters = mytoolbox.readjson(filedir + '/parameters/patch_motion_correction_parameters.json')

    def QueuePatchMotionCorrectionJob(self, parameters, source_movie_job_list, source_movie_parameter_name_list, source_doseweight_job=None, source_doseweight_parameter_name=None, hostname=None, gpus=None):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, mytoolbox.readjson(filedir + '/parameters/patch_motion_correction_job_name.json'))
        for i in range(len(source_movie_job_list)):
            job.connect('movies', source_movie_job_list[i], source_movie_parameter_name_list[i])
        if source_doseweight_job is not None:
            job.connect('doseweights', source_doseweight_job, source_doseweight_parameter_name)

        parameters_map = mytoolbox.readjson(filedir + '/parameters/patch_motion_correction_parameters_map.json')
        for param_title in parameters:
            job.set_param(parameters_map[param_title], parameters[param_title])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job


class CreatePatchCtfEstimation():
    def __init__(self, DealJobs_instance):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        self.dealjobtools = DealJobs_instance
        self.default_patch_ctf_estimation_job_parameters = mytoolbox.readjson(filedir + '/parameters/patch_ctf_estimation_parameters.json')

    def QueuePatchCtfEstimationJob(self, parameters, source_exposure_job_list, source_exposure_parameter_name_list, hostname=None, gpus=None):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, mytoolbox.readjson(filedir + '/parameters/patch_ctf_estimation_job_name.json'))
        for i in range(len(source_exposure_job_list)):
            job.connect('exposures', source_exposure_job_list[i], source_exposure_parameter_name_list[i])

        parameters_map = mytoolbox.readjson(filedir + '/parameters/patch_ctf_estimation_parameters_map.json')
        for param_title in parameters:
            job.set_param(parameters_map[param_title], parameters[param_title])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job


class CreateBlobPicker():
    def __init__(self, DealJobs_instance):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        self.dealjobtools = DealJobs_instance
        self.default_blob_picker_job_parameters = mytoolbox.readjson(filedir + '/parameters/blob_picker_parameters.json')

    def QueueBlobPickerJob(self, parameters, source_micrograph_job_list, source_micrograph_parameter_name_list, hostname=None, gpus=None):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, mytoolbox.readjson(filedir + '/parameters/blob_picker_job_name.json'))
        for i in range(len(source_micrograph_job_list)):
            job.connect('micrographs', source_micrograph_job_list[i], source_micrograph_parameter_name_list[i])

        parameters_map = mytoolbox.readjson(filedir + '/parameters/blob_picker_parameters_map.json')
        for param_title in parameters:
            job.set_param(parameters_map[param_title], parameters[param_title])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job


class CreateExtractMicrographs():
    def __init__(self, DealJobs_instance):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        self.dealjobtools = DealJobs_instance
        self.default_extract_micrographs_job_parameters = mytoolbox.readjson(filedir + '/parameters/extract_micrographs_parameters.json')

    def QueueExtractMicrographsJob(self, parameters, source_micrograph_job_list, source_micrograph_parameter_name_list, source_particle_job_list, source_particle_parameter_name_list, hostname=None, gpus=None):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, mytoolbox.readjson(filedir + '/parameters/extract_micrographs_job_name.json'))
        for i in range(len(source_micrograph_job_list)):
            job.connect('micrographs', source_micrograph_job_list[i], source_micrograph_parameter_name_list[i])
        for i in range(len(source_particle_job_list)):
            job.connect('particles', source_particle_job_list[i], source_particle_parameter_name_list[i])

        parameters_map = mytoolbox.readjson(filedir + '/parameters/extract_micrographs_parameters_map.json')
        for param_title in parameters:
            job.set_param(parameters_map[param_title], parameters[param_title])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job


class CreateRestackParticles():
    def __init__(self, DealJobs_instance):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        self.dealjobtools = DealJobs_instance
        self.default_restack_particles_job_parameters = mytoolbox.readjson(filedir + '/parameters/restack_particles_parameters.json')

    def QueueRestackJob(self, parameters, source_particle_job_list, source_particle_parameter_name_list):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, mytoolbox.readjson(filedir + '/parameters/restack_particles_job_name.json'))
        for i in range(len(source_particle_job_list)):
            job.connect('particles', source_particle_job_list[i], source_particle_parameter_name_list[i])

        parameters_map = mytoolbox.readjson(filedir + '/parameters/restack_particles_parameters_map.json')
        for param_title in parameters:
            job.set_param(parameters_map[param_title], parameters[param_title])

        job.queue(self.dealjobtools.lane)
        return job


class CreateHomoAbinit():
    def __init__(self, DealJobs_instance):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        self.dealjobtools = DealJobs_instance
        self.default_homo_abinit_job_parameters = mytoolbox.readjson(filedir + '/parameters/abinit_parameters.json')

    def QueueHomoAbinitJob(self, parameters, source_particle_job_list, source_particle_parameter_name_list, hostname=None, gpus=None):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, mytoolbox.readjson(filedir + '/parameters/abinit_job_name.json'))
        for i in range(len(source_particle_job_list)):
            job.connect('particles', source_particle_job_list[i], source_particle_parameter_name_list[i])

        parameters_map = mytoolbox.readjson(filedir + '/parameters/abinit_parameters_map.json')
        for param_title in parameters:
            job.set_param(parameters_map[param_title], parameters[param_title])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job


class CreateNonuniformRefine():
    def __init__(self, DealJobs_instance):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        self.dealjobtools = DealJobs_instance
        self.default_nonuniform_refine_job_parameters = filedir + '/parameters/nurefine_parameters.json'

    def QueueNonuniformRefineJob(self, parameters, source_particle_job_list, source_particle_parameter_name_list, source_map_job, source_map_parameter_name, source_mask_job=None, source_mask_parameter_name=None, hostname=None, gpus=None):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, mytoolbox.readjson(filedir + '/parameters/nurefine_job_name.json'))
        for i in range(len(source_particle_job_list)):
            job.connect('particles', source_particle_job_list[i], source_particle_parameter_name_list[i])
        job.connect('volume', source_map_job, source_map_parameter_name)
        if source_mask_job is not None:
            job.connect('mask', source_mask_job, source_mask_parameter_name)

        parameters_map = mytoolbox.readjson(filedir + '/parameters/nurefine_parameters_map.json')
        for param_title in parameters:
            job.set_param(parameters_map[param_title], parameters[param_title])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job


class CreateParticleSetsTool():
    def __init__(self, DealJobs_instance):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        self.dealjobtools = DealJobs_instance
        self.default_particle_sets_tool_job_parameters = mytoolbox.readjson(filedir + '/parameters/particle_set_tools_parameters.json')

    def QueueParticleSetsToolJob(self, parameters, source_particle_job_list_A, source_particle_parameter_name_list_A, source_particle_job_list_B=None, source_particle_parameter_name_list_B=None):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, mytoolbox.readjson(filedir + '/parameters/particle_set_tools_job_name.json'))
        for i in range(len(source_particle_job_list_A)):
            job.connect('particles_A', source_particle_job_list_A[i], source_particle_parameter_name_list_A[i])
        if source_particle_job_list_B is not None:
            for i in range(len(source_particle_job_list_B)):
                job.connect('particles_B', source_particle_job_list_B[i], source_particle_parameter_name_list_B[i])

        parameters_map = mytoolbox.readjson(filedir + '/parameters/particle_set_tools_parameters_map.json')
        for param_title in parameters:
            job.set_param(parameters_map[param_title], parameters[param_title])

        job.queue(self.dealjobtools.lane)
        return job


class CreateOrientationDiagnostics():
    def __init__(self, DealJobs_instance):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        self.dealjobtools = DealJobs_instance
        self.default_orientation_diagnostics_job_parameters = mytoolbox.readjson(filedir + '/parameters/orientation_diagnostics_parameters.json')

    def QueueOrientationDiagnosticsJob(self, parameters, source_volume_job, source_volume_parameter_name, source_particle_job_list=None, source_particle_parameter_name_list=None, source_mask_job=None, source_mask_parameter_name=None, hostname=None, gpus=None):
        filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, mytoolbox.readjson(filedir + '/parameters/orientation_diagnostics_job_name.json'))
        job.connect('volume', source_volume_job, source_volume_parameter_name)
        if source_particle_job_list is not None:
            for i in range(len(source_particle_job_list)):
                job.connect('particles', source_particle_job_list[i], source_particle_parameter_name_list[i])
        if source_mask_job is not None:
            job.connect('mask', source_mask_job, source_mask_parameter_name)

        parameters_map = mytoolbox.readjson(filedir + '/parameters/orientation_diagnostics_parameters_map.json')
        for param_title in parameters:
            job.set_param(parameters_map[param_title], parameters[param_title])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job