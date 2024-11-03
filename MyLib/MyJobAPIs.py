import threading
import time
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import MyLib.mytoolbox as mytoolbox



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
        self.dealjobtools = DealJobs_instance
        self.default_import_movies_job_parameters = {
            'Movies data path': None,
            'Gain reference path': None,
            'Defect file path': None,
            'Flip gain ref & defect file in X?': False,
            'Flip gain ref & defect file in Y?': False,
            'Rotate gain ref?': 0,
            'Raw pixel size (A)': None,
            'Accelerating Voltage (kV)': None,
            'Spherical Aberration (mm)': None,
            'Total exposure dose (e/A^2)': None,
            'Negative Stain Data': False,
            'Phase Plate Data': False,
            'Override Exposure Group ID': None,
            'Skip Header Check': True,
            'EER Number of Fractions': 40,
            'EER Upsampling Factor': 2,
            'Output Constant CTF': False,

            'Import Beam Shift Values from XML Files': False,
            'EPU XML metadata path': None,
            'Length of movie filename prefix to cut for XML correspondence': None,
            'Length of movie filename suffix to cut for XML correspondence': None,
            'Length of XML filename prefix to cut for movie correspondence': None,
            'Length of XML filename suffix to cut for movie correspondence': 4,

            'Number of CPUs to parallelize': 4
        }

    def QueueImportMoviesJob(self, parameters):
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, 'import_movies')

        job.set_param('blob_paths', parameters['Movies data path'])
        job.set_param('gainref_path', parameters['Gain reference path'])
        job.set_param('defect_path', parameters['Defect file path'])
        job.set_param('gainref_flip_x', parameters['Flip gain ref & defect file in X?'])
        job.set_param('gainref_flip_y', parameters['Flip gain ref & defect file in Y?'])
        job.set_param('gainref_rotate_num', parameters['Rotate gain ref?'])
        job.set_param('psize_A', parameters['Raw pixel size (A)'])
        job.set_param('accel_kv', parameters['Accelerating Voltage (kV)'])
        job.set_param('cs_mm', parameters['Spherical Aberration (mm)'])
        job.set_param('total_dose_e_per_A2', parameters['Total exposure dose (e/A^2)'])
        job.set_param('negative_stain_data', parameters['Negative Stain Data'])
        job.set_param('phase_plate_data', parameters['Phase Plate Data'])
        job.set_param('override_exp_group_id', parameters['Override Exposure Group ID'])
        job.set_param('skip_header_check', parameters['Skip Header Check'])
        job.set_param('eer_num_fractions', parameters['EER Number of Fractions'])
        job.set_param('eer_upsamp_factor', parameters['EER Upsampling Factor'])
        job.set_param('output_constant_ctf', parameters['Output Constant CTF'])

        job.set_param('parse_xml_files', parameters['Import Beam Shift Values from XML Files'])
        job.set_param('xml_paths', parameters['EPU XML metadata path'])
        job.set_param('mov_cut_prefix_xml', parameters['Length of movie filename prefix to cut for XML correspondence'])
        job.set_param('mov_cut_suffix_xml', parameters['Length of movie filename suffix to cut for XML correspondence'])
        job.set_param('xml_cut_prefix_xml', parameters['Length of XML filename prefix to cut for movie correspondence'])
        job.set_param('xml_cut_suffix_xml', parameters['Length of XML filename suffix to cut for movie correspondence'])

        job.set_param('compute_num_cpus', parameters['Number of CPUs to parallelize'])

        job.queue(self.dealjobtools.lane)
        return job


class CreateImportMicrographs():
    def __init__(self, DealJobs_instance):
        self.dealjobtools = DealJobs_instance
        self.default_import_micrographs_job_parameters = {
            'Micrographs data path': None,
            'Length of movie path prefix to cut': None,
            'Length of movie path suffix to cut': None,
            'Length of mic. path prefix to cut': None,
            'Length of mic. path suffix to cut': None,
            'Pixel size (A)': None,
            'Accelerating Voltage (kV)': None,
            'Spherical Aberration (mm)': None,
            'Total exposure dose (e/A^2)': None,
            'Negative Stain Data': False,
            'Phase Plate Data': False,
            'Override Exposure Group ID': None,
            'Skip Header Check': True,
            'Output Constant CTF': False,

            'Import Beam Shift Values from XML Files': False,
            'EPU XML metadata path': None,
            'Length of mic filename prefix to cut for XML correspondence': None,
            'Length of mic filename suffix to cut for XML correspondence': None,
            'Length of XML filename prefix to cut for mic correspondence': None,
            'Length of XML filename suffix to cut for mic correspondence': 4,

            'Number of CPUs to parallelize during header check': 4
        }

    def QueueImportMicrographsJob(self, parameters, source_movie_job_list=None, source_movie_parameter_name_list=None):
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, 'import_micrographs')
        if source_movie_job_list is not None:
            for i in range(len(source_movie_job_list)):
                job.connect('movies', source_movie_job_list[i], source_movie_parameter_name_list[i])

        job.set_param('blob_paths', parameters['Micrographs data path'])
        job.set_param('source_cut_prefix', parameters['Length of movie path prefix to cut'])
        job.set_param('source_cut_suff', parameters['Length of movie path suffix to cut'])
        job.set_param('query_cut_prefix', parameters['Length of mic. path prefix to cut'])
        job.set_param('query_cut_suff', parameters['Length of mic. path suffix to cut'])
        job.set_param('psize_A', parameters['Pixel size (A)'])
        job.set_param('accel_kv', parameters['Accelerating Voltage (kV)'])
        job.set_param('cs_mm', parameters['Spherical Aberration (mm)'])
        job.set_param('total_dose_e_per_A2', parameters['Total exposure dose (e/A^2)'])
        job.set_param('negative_stain_data', parameters['Negative Stain Data'])
        job.set_param('phase_plate_data', parameters['Phase Plate Data'])
        job.set_param('override_exp_group_id', parameters['Override Exposure Group ID'])
        job.set_param('skip_header_check', parameters['Skip Header Check'])
        job.set_param('output_constant_ctf', parameters['Output Constant CTF'])

        job.set_param('parse_xml_files', parameters['Import Beam Shift Values from XML Files'])
        job.set_param('xml_paths', parameters['EPU XML metadata path'])
        job.set_param('mov_cut_prefix_xml', parameters['Length of mic filename prefix to cut for XML correspondence'])
        job.set_param('mov_cut_suffix_xml', parameters['Length of mic filename suffix to cut for XML correspondence'])
        job.set_param('xml_cut_prefix_xml', parameters['Length of XML filename prefix to cut for mic correspondence'])
        job.set_param('xml_cut_suffix_xml', parameters['Length of XML filename suffix to cut for mic correspondence'])

        job.set_param('compute_num_cpus', parameters['Number of CPUs to parallelize during header check'])

        job.queue(self.dealjobtools.lane)
        return job


class CreatePatchMotionCorrection():
    def __init__(self, DealJobs_instance):
        self.dealjobtools = DealJobs_instance
        self.default_patch_motion_correction_job_parameters = {
            'Make motion diagnostic plots': True,
            'Number of movies to plot': 10,
            'Only process this many movies': None,
            'Low-memory mode': False,
            'Save results in 16-bit floating point': False,

            'Maximum alignment resolution (A)': 5,
            'B-factor during alignment': 500,
            'Start frame (included, 0-based)': 0,
            'End frame (excluded, 0-based) ': None,
            'Output F-crop factor': '1',
            'Override e/A^2': None,
            'Allow Variable Dose': False,
            'Calibrated smoothing': 0.5,
            'Override knots Z': None,
            'Override knots Y': None,
            'Override knots X': None,

            'Random seed': None,

            'Number of GPUs to parallelize': 1
        }

    def QueuePatchMotionCorrectionJob(self, parameters, source_movie_job_list, source_movie_parameter_name_list, source_doseweight_job=None, source_doseweight_parameter_name=None, hostname=None, gpus=None):
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, 'patch_motion_correction_multi')
        for i in range(len(source_movie_job_list)):
            job.connect('movies', source_movie_job_list[i], source_movie_parameter_name_list[i])
        if source_doseweight_job is not None:
            job.connect('doseweights', source_doseweight_job, source_doseweight_parameter_name)

        job.set_param('do_plots', parameters['Make motion diagnostic plots'])
        job.set_param('num_plots', parameters['Number of movies to plot'])
        job.set_param('random_num', parameters['Only process this many movies'])
        job.set_param('memoryfix2', parameters['Low-memory mode'])
        job.set_param('output_f16', parameters['Save results in 16-bit floating point'])

        job.set_param('res_max_align', parameters['Maximum alignment resolution (A)'])
        job.set_param('bfactor', parameters['B-factor during alignment'])
        job.set_param('frame_start', parameters['Start frame (included, 0-based)'])
        job.set_param('frame_end', parameters['End frame (excluded, 0-based) '])
        job.set_param('output_fcrop_factor', parameters['Output F-crop factor'])
        job.set_param('override_total_exp', parameters['Override e/A^2'])
        job.set_param('variable_dose', parameters['Allow Variable Dose'])
        job.set_param('smooth_lambda_cal', parameters['Calibrated smoothing'])
        job.set_param('override_K_Z', parameters['Override knots Z'])
        job.set_param('override_K_Y', parameters['Override knots Y'])
        job.set_param('override_K_X', parameters['Override knots X'])

        job.set_param('compute_num_gpus', parameters['Number of GPUs to parallelize'])

        if parameters['Random seed'] is not None:
            job.set_param('random_seed', parameters['Random seed'])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job


class CreatePatchCtfEstimation():
    def __init__(self, DealJobs_instance):
        self.dealjobtools = DealJobs_instance
        self.default_patch_ctf_estimation_job_parameters = {
            'Number of movies to plot': 10,
            'Only process this many movies': None,
            'Classic mode': False,

            'Amplitude Contrast': 0.1,
            'Minimum resolution (A)': 25,
            'Maximum resolution (A)': 4,
            'Minimum search defocus (A)': 1000,
            'Maximum search defocus (A)': 40000,
            'Min. search phase-shift (rad)': 0,
            'Max. search phase-shift (rad)': 3.141592653589793,
            'Do phase refine only': False,
            'Override knots Y': None,
            'Override knots X': None,

            'Number of GPUs to parallelize': 1
        }

    def QueuePatchCtfEstimationJob(self, parameters, source_exposure_job_list, source_exposure_parameter_name_list, hostname=None, gpus=None):
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, 'patch_ctf_estimation_multi')
        for i in range(len(source_exposure_job_list)):
            job.connect('exposures', source_exposure_job_list[i], source_exposure_parameter_name_list[i])

        job.set_param('num_plots', parameters['Number of movies to plot'])
        job.set_param('random_num', parameters['Only process this many movies'])
        job.set_param('classic_mode', parameters['Classic mode'])

        job.set_param('amp_contrast', parameters['Amplitude Contrast'])
        job.set_param('res_min_align', parameters['Minimum resolution (A)'])
        job.set_param('res_max_align', parameters['Maximum resolution (A)'])
        job.set_param('df_search_min', parameters['Minimum search defocus (A)'])
        job.set_param('df_search_max', parameters['Maximum search defocus (A)'])
        job.set_param('phase_shift_min', parameters['Min. search phase-shift (rad)'])
        job.set_param('phase_shift_max', parameters['Max. search phase-shift (rad)'])
        job.set_param('do_phase_shift_refine_only', parameters['Do phase refine only'])
        job.set_param('override_K_Y', parameters['Override knots Y'])
        job.set_param('override_K_X', parameters['Override knots X'])

        job.set_param('compute_num_gpus', parameters['Number of GPUs to parallelize'])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job


class CreateBlobPicker():
    def __init__(self, DealJobs_instance):
        self.dealjobtools = DealJobs_instance
        self.default_blob_picker_job_parameters = {
            'Minimum particle diameter (A)': None,
            'Maximum particle diameter (A)': None,
            'Use circular blob': True,
            'Use elliptical blob': False,
            'Use ring blob': False,
            'Lowpass filter to apply to templates (A)': 20,
            'Lowpass filter to apply to micrographs (A)': 20,
            'Angular sampling (degrees)': 5,
            'Min. separation dist (diameters)': 1.0,
            'Number of mics to process': None,
            'Number of mics to plot': 10,
            'Maximum number of local maxima to consider': 4000,
            'Recenter templates': True
        }

    def QueueBlobPickerJob(self, parameters, source_micrograph_job_list, source_micrograph_parameter_name_list, hostname=None, gpus=None):
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, 'blob_picker_gpu')
        for i in range(len(source_micrograph_job_list)):
            job.connect('micrographs', source_micrograph_job_list[i], source_micrograph_parameter_name_list[i])

        job.set_param('diameter', parameters['Minimum particle diameter (A)'])
        job.set_param('diameter_max', parameters['Maximum particle diameter (A)'])
        job.set_param('use_circle', parameters['Use circular blob'])
        job.set_param('use_ellipse', parameters['Use elliptical blob'])
        job.set_param('use_ring', parameters['Use ring blob'])
        job.set_param('lowpass_res_template', parameters['Lowpass filter to apply to templates (A)'])
        job.set_param('lowpass_res', parameters['Lowpass filter to apply to micrographs (A)'])
        job.set_param('angular_spacing_deg', parameters['Angular sampling (degrees)'])
        job.set_param('min_distance', parameters['Min. separation dist (diameters)'])
        job.set_param('num_process', parameters['Number of mics to process'])
        job.set_param('num_plot', parameters['Number of mics to plot'])
        job.set_param('max_num_hits', parameters['Maximum number of local maxima to consider'])
        job.set_param('recenter_templates', parameters['Recenter templates'])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job


class CreateExtractMicrographs():
    def __init__(self, DealJobs_instance):
        self.dealjobtools = DealJobs_instance
        self.default_extract_micrographs_job_parameters = {
            'Number of GPUs to parallelize (0 for CPU-only)': 1,

            'Extraction box size (pix)': 256,
            'Fourier crop to box size (pix)': None,
            'Save results in 16-bit floating point': False,
            'Force re-extract CTFs from micrographs': False,
            'Recenter using aligned shifts': True,
            'Number of mics to extract': None,
            'Flip mic. in x before extract?': False,
            'Flip mic. in y before extract?': False,
            'Scale constant (override)': None
        }

    def QueueExtractMicrographsJob(self, parameters, source_micrograph_job_list, source_micrograph_parameter_name_list, source_particle_job_list, source_particle_parameter_name_list, hostname=None, gpus=None):
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, 'extract_micrographs_multi')
        for i in range(len(source_micrograph_job_list)):
            job.connect('micrographs', source_micrograph_job_list[i], source_micrograph_parameter_name_list[i])
        for i in range(len(source_particle_job_list)):
            job.connect('particles', source_particle_job_list[i], source_particle_parameter_name_list[i])

        job.set_param('compute_num_gpus', parameters['Number of GPUs to parallelize (0 for CPU-only)'])

        job.set_param('box_size_pix', parameters['Extraction box size (pix)'])
        job.set_param('bin_size_pix', parameters['Fourier crop to box size (pix)'])
        job.set_param('output_f16', parameters['Save results in 16-bit floating point'])
        job.set_param('force_reextract_CTF', parameters['Force re-extract CTFs from micrographs'])
        job.set_param('recenter_using_shifts', parameters['Recenter using aligned shifts'])
        job.set_param('num_extract', parameters['Number of mics to extract'])
        job.set_param('flip_x', parameters['Flip mic. in x before extract?'])
        job.set_param('flip_y', parameters['Flip mic. in y before extract?'])
        job.set_param('scale_const_override', parameters['Scale constant (override)'])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job


class CreateRestackParticles():
    def __init__(self, DealJobs_instance):
        self.dealjobtools = DealJobs_instance
        self.default_restack_particles_job_parameters = {
            'Save results in 16-bit floating point': False,
            'Num threads': 2,
            'Num particles to extract': None,
            'Particle batch size': 10000
        }

    def QueueRestackJob(self, parameters, source_particle_job_list, source_particle_parameter_name_list):
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, 'restack_particles')
        for i in range(len(source_particle_job_list)):
            job.connect('particles', source_particle_job_list[i], source_particle_parameter_name_list[i])

        job.set_param('output_f16', parameters['Save results in 16-bit floating point'])
        job.set_param('num_threads', parameters['Num threads'])
        job.set_param('num_particles', parameters['Num particles to extract'])
        job.set_param('batch_size', parameters['Particle batch size'])

        job.queue(self.dealjobtools.lane)
        return job


class CreateHomoAbinit():
    def __init__(self, DealJobs_instance):
        self.dealjobtools = DealJobs_instance
        self.default_homo_abinit_job_parameters = {
            'Window dataset (real-space)': True,
            'Window inner radius': 0.85,
            'Window outer radius': 0.99,

            'Number of Ab-Initio classes': 1,
            'Num particles to use': None,
            'Maximum resolution (Angstroms)': 12.0,
            'Initial resolution (Angstroms)': 35.0,
            'Number of initial iterations': 200,
            'Number of final iterations': 300,
            'Fourier radius step': 0.04,
            'Window structures in real space': True,
            'Center structures in real space': True,
            'Correct for per-micrograph optimal scales': False,
            'Compute per-image optimal scales': False,
            'SGD Momentum': 0,
            'Sparsity prior': 0,
            'Initial minibatch size': 90,
            'Final minibatch size': 300,
            'Abinit minisize epsilon': 0.05,
            'Abinit minisize minp': 0.01,
            'Initial minibatch size num iters': 300,
            'Noise model (white, symmetric or coloured)': 'symmetric',
            'Noise priorw': 50,
            'Noise initw': 5000,
            'Noise initial sigma-scale': None,
            'Class similarity': 0.1,
            'Class similarity anneal start iter': 300,
            'Class similarity anneal end iter': 350,
            'Target 3D ESS Fraction': 0.011,
            'Symmetry': 'C1',
            'Initial learning rate duration': 100,
            'Initial learning rate': 0.4,
            'Enforce non-negativity': True,
            'Ignore DC component': True,
            'Initial structure random seed': None,
            'Initial structure lowpass (Fourier radius)': 7,
            'Use fast codepaths': True,
            'Show plots from intermediate steps': True,

            'Random seed': None,

            'Cache particle images on SSD': True
        }

    def QueueHomoAbinitJob(self, parameters, source_particle_job_list, source_particle_parameter_name_list, hostname=None, gpus=None):
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, 'homo_abinit')
        for i in range(len(source_particle_job_list)):
            job.connect('particles', source_particle_job_list[i], source_particle_parameter_name_list[i])

        job.set_param('prepare_window_dataset', parameters['Window dataset (real-space)'])
        job.set_param('prepare_window_inner_radius', parameters['Window inner radius'])
        job.set_param('prepare_window_outer_radius', parameters['Window outer radius'])

        job.set_param('abinit_K', parameters['Number of Ab-Initio classes'])
        job.set_param('abinit_num_particles', parameters['Num particles to use'])
        job.set_param('abinit_max_res', parameters['Maximum resolution (Angstroms)'])
        job.set_param('abinit_init_res', parameters['Initial resolution (Angstroms)'])
        job.set_param('abinit_num_init_iters', parameters['Number of initial iterations'])
        job.set_param('abinit_num_final_iters', parameters['Number of final iterations'])
        job.set_param('abinit_radwn_step', parameters['Fourier radius step'])
        job.set_param('abinit_window', parameters['Window structures in real space'])
        job.set_param('abinit_center', parameters['Center structures in real space'])
        job.set_param('abinit_scale_mg_correct', parameters['Correct for per-micrograph optimal scales'])
        job.set_param('abinit_scale_compute', parameters['Compute per-image optimal scales'])
        job.set_param('abinit_mom', parameters['SGD Momentum'])
        job.set_param('abinit_sparsity', parameters['Sparsity prior'])
        job.set_param('abinit_minisize_init', parameters['Initial minibatch size'])
        job.set_param('abinit_minisize', parameters['Final minibatch size'])
        job.set_param('abinit_minisize_epsilon', parameters['Abinit minisize epsilon'])
        job.set_param('abinit_minisize_minp', parameters['Abinit minisize minp'])
        job.set_param('abinit_minisize_num_init_iters', parameters['Initial minibatch size num iters'])
        job.set_param('abinit_noise_model', parameters['Noise model (white, symmetric or coloured)'])
        job.set_param('abinit_noise_priorw', parameters['Noise priorw'])
        job.set_param('abinit_noise_initw', parameters['Noise initw'])
        job.set_param('abinit_noise_init_sigmascale', parameters['Noise initial sigma-scale'])
        job.set_param('abinit_class_anneal_beta', parameters['Class similarity'])
        job.set_param('abinit_class_anneal_start', parameters['Class similarity anneal start iter'])
        job.set_param('abinit_class_anneal_end', parameters['Class similarity anneal end iter'])
        job.set_param('abinit_target_initial_ess_fraction', parameters['Target 3D ESS Fraction'])
        job.set_param('abinit_symmetry', parameters['Symmetry'])
        job.set_param('abinit_high_lr_duration', parameters['Initial learning rate duration'])
        job.set_param('abinit_high_lr', parameters['Initial learning rate'])
        job.set_param('abinit_nonneg', parameters['Enforce non-negativity'])
        job.set_param('abinit_ignore_dc', parameters['Ignore DC component'])
        job.set_param('abinit_seed_init', parameters['Initial structure random seed'])
        job.set_param('abinit_init_radwn_cutoff', parameters['Initial structure lowpass (Fourier radius)'])
        job.set_param('abinit_use_engine', parameters['Use fast codepaths'])
        job.set_param('intermediate_plots', parameters['Show plots from intermediate steps'])

        job.set_param('compute_use_ssd', parameters['Cache particle images on SSD'])

        if parameters['Random seed'] is not None:
            job.set_param('random_seed', parameters['Random seed'])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job


class CreateNonuniformRefine():
    def __init__(self, DealJobs_instance):
        self.dealjobtools = DealJobs_instance
        self.default_nonuniform_refine_job_parameters = {
            'Window dataset (real-space)': True,
            'Window inner radius': 0.85,
            'Window outer radius': 0.99,

            'Symmetry': 'C1',
            'Symmetry relaxation method': 'none',
            'Do symmetry alignment': True,
            'Re-estimate greyscale level of input reference': True,
            'Number of extra final passes': 0,
            'Maximum align resolution (A)': None,
            'Initial lowpass resolution (A)': 30,
            'GSFSC split resolution (A)': 20,
            'Force re-do GS split': True,
            'Adaptive Marginalization': True,
            'Non-uniform refine enable': True,
            'Non-uniform filter order': 8,
            'Non-uniform AWF': 3,
            'Enforce non-negativity': False,
            'Window structure in real space': True,
            'Skip interpolant premult': True,
            'Ignore DC component': True,
            'Ignore tilt': False,
            'Ignore trefoil': False,
            'Ignore tetra': False,
            'Use random ordering of particles': True,
            'Disable auto batchsize': False,
            'Batchsize epsilon': 0.001,
            'Batchsize snrfactor': 50.0,
            'Reset input per-particle scale': True,
            'Minimize over per-particle scale': False,
            'Scale min/use start iter': 0,
            'Noise model (white, symmetric or coloured)': 'symmetric',
            'Noise priorw': 50,
            'Noise initw': 200,
            'Noise initial sigma-scale': 3,
            'Initialize noise model from images': False,
            'Dynamic mask threshold (0-1)': 0.2,
            'Dynamic mask near (A)': 6.0,
            'Dynamic mask far (A)': 14.0,
            'Dynamic mask start resolution (A)': 12.0,
            'Dynamic mask use absolute value': False,
            'Show plots from intermediate steps': True,
            'GPU batch size of images (homo)': None,

            'Optimize per-particle defocus': False,
            'Num. particles to plot': 3,
            'Minimum Fit Res (A) (defocus)': 20,
            'Defocus Search Range (A +/-)': 2000,
            'GPU batch size of images (defocus)': None,

            'Optimize per-group CTF params': False,
            'Num. groups to plot': 3,
            'Binning to apply to plots': 1,
            'Minimum Fit Res (A) (CTF)': 10,
            'Fit Tilt': True,
            'Fit Trefoil': True,
            'Fit Spherical Aberration': False,
            'Fit Tetrafoil': False,
            'Fit Anisotropic Mag.': False,
            'GPU batch size of images (CTF)': None,

            'Do EWS correction': False,
            'EWS curvature sign': 'negative',

            'Random seed': None,

            'Cache particle images on SSD': True,
            'Low-Memory Mode': False
        }

    def QueueNonuniformRefineJob(self, parameters, source_particle_job_list, source_particle_parameter_name_list, source_map_job, source_map_parameter_name, source_mask_job=None, source_mask_parameter_name=None, hostname=None, gpus=None):
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, 'nonuniform_refine_new')
        for i in range(len(source_particle_job_list)):
            job.connect('particles', source_particle_job_list[i], source_particle_parameter_name_list[i])
        job.connect('volume', source_map_job, source_map_parameter_name)
        if source_mask_job is not None:
            job.connect('mask', source_mask_job, source_mask_parameter_name)

        job.set_param('prepare_window_dataset', parameters['Window dataset (real-space)'])
        job.set_param('prepare_window_inner_radius', parameters['Window inner radius'])
        job.set_param('prepare_window_outer_radius', parameters['Window outer radius'])

        job.set_param('refine_symmetry', parameters['Symmetry'])
        job.set_param('refine_relax_symmetry', parameters['Symmetry relaxation method'])
        job.set_param('refine_symmetry_do_align', parameters['Do symmetry alignment'])
        job.set_param('refine_do_init_scale_est', parameters['Re-estimate greyscale level of input reference'])
        job.set_param('refine_num_final_iterations', parameters['Number of extra final passes'])
        job.set_param('refine_res_align_max', parameters['Maximum align resolution (A)'])
        job.set_param('refine_res_init', parameters['Initial lowpass resolution (A)'])
        job.set_param('refine_res_gsfsc_split', parameters['GSFSC split resolution (A)'])
        job.set_param('refine_gs_resplit', parameters['Force re-do GS split'])
        job.set_param('refine_do_marg', parameters['Adaptive Marginalization'])
        job.set_param('refine_nu_enable', parameters['Non-uniform refine enable'])
        job.set_param('refine_nu_order', parameters['Non-uniform filter order'])
        job.set_param('refine_nu_awf', parameters['Non-uniform AWF'])
        job.set_param('refine_clip', parameters['Enforce non-negativity'])
        job.set_param('refine_window', parameters['Window structure in real space'])
        job.set_param('refine_skip_premult', parameters['Skip interpolant premult'])
        job.set_param('refine_ignore_dc', parameters['Ignore DC component'])
        job.set_param('refine_ignore_tilt', parameters['Ignore tilt'])
        job.set_param('refine_ignore_trefoil', parameters['Ignore trefoil'])
        job.set_param('refine_ignore_tetra', parameters['Ignore tetra'])
        job.set_param('refine_batch_random', parameters['Use random ordering of particles'])
        job.set_param('refine_batchsize_no_auto', parameters['Disable auto batchsize'])
        job.set_param('refine_batchsize_epsilon', parameters['Batchsize epsilon'])
        job.set_param('refine_batchsize_snrfactor', parameters['Batchsize snrfactor'])
        job.set_param('refine_scale_reset', parameters['Reset input per-particle scale'])
        job.set_param('refine_scale_min', parameters['Minimize over per-particle scale'])
        job.set_param('refine_scale_start_iter', parameters['Scale min/use start iter'])
        job.set_param('refine_noise_model', parameters['Noise model (white, symmetric or coloured)'])
        job.set_param('refine_noise_priorw', parameters['Noise priorw'])
        job.set_param('refine_noise_initw', parameters['Noise initw'])
        job.set_param('refine_noise_init_sigmascale', parameters['Noise initial sigma-scale'])
        job.set_param('refine_noise_init_from_imgs', parameters['Initialize noise model from images'])
        job.set_param('refine_dynamic_mask_thresh_factor', parameters['Dynamic mask threshold (0-1)'])
        job.set_param('refine_dynamic_mask_near_ang', parameters['Dynamic mask near (A)'])
        job.set_param('refine_dynamic_mask_far_ang', parameters['Dynamic mask far (A)'])
        job.set_param('refine_dynamic_mask_start_res', parameters['Dynamic mask start resolution (A)'])
        job.set_param('refine_dynamic_mask_use_abs', parameters['Dynamic mask use absolute value'])
        job.set_param('intermediate_plots', parameters['Show plots from intermediate steps'])
        job.set_param('refine_compute_batch_size', parameters['GPU batch size of images (homo)'])

        job.set_param('refine_defocus_refine', parameters['Optimize per-particle defocus'])
        job.set_param('crl_num_plots', parameters['Num. particles to plot'])
        job.set_param('crl_min_res_A', parameters['Minimum Fit Res (A) (defocus)'])
        job.set_param('crl_df_range', parameters['Defocus Search Range (A +/-)'])
        job.set_param('crl_compute_batch_size', parameters['GPU batch size of images (defocus)'])

        job.set_param('refine_ctf_global_refine', parameters['Optimize per-group CTF params'])
        job.set_param('crg_num_plots', parameters['Num. groups to plot'])
        job.set_param('crg_plot_binfactor', parameters['Binning to apply to plots'])
        job.set_param('crg_min_res_A', parameters['Minimum Fit Res (A) (CTF)'])
        job.set_param('crg_do_tilt', parameters['Fit Tilt'])
        job.set_param('crg_do_trefoil', parameters['Fit Trefoil'])
        job.set_param('crg_do_spherical', parameters['Fit Spherical Aberration'])
        job.set_param('crg_do_tetrafoil', parameters['Fit Tetrafoil'])
        job.set_param('crg_do_anisomag', parameters['Fit Anisotropic Mag.'])
        job.set_param('crg_compute_batch_size', parameters['GPU batch size of images (CTF)'])

        job.set_param('refine_do_ews_correct', parameters['Do EWS correction'])
        job.set_param('refine_ews_zsign', parameters['EWS curvature sign'])

        job.set_param('compute_use_ssd', parameters['Cache particle images on SSD'])
        job.set_param('low_memory_mode', parameters['Low-Memory Mode'])

        if parameters['Random seed'] is not None:
            job.set_param('random_seed', parameters['Random seed'])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job


class CreateParticleSetsTool():
    def __init__(self, DealJobs_instance):
        self.dealjobtools = DealJobs_instance
        self.default_particle_sets_tool_job_parameters = {
            'Action': 'split',
            'Split num. batches': 2,
            'Split batch size': None,
            'Split randomize': False,

            'Field to Intersect': 'uid',
            'Ignore Leading UID': False,
        }

    def QueueParticleSetsToolJob(self, parameters, source_particle_job_list_A, source_particle_parameter_name_list_A, source_particle_job_list_B=None, source_particle_parameter_name_list_B=None):
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, 'particle_sets')
        for i in range(len(source_particle_job_list_A)):
            job.connect('particles_A', source_particle_job_list_A[i], source_particle_parameter_name_list_A[i])
        if source_particle_job_list_B is not None:
            for i in range(len(source_particle_job_list_B)):
                job.connect('particles_B', source_particle_job_list_B[i], source_particle_parameter_name_list_B[i])

        job.set_param('set_operation', parameters['Action'])
        job.set_param('set_split_num', parameters['Split num. batches'])
        job.set_param('set_split_size', parameters['Split batch size'])
        job.set_param('set_split_random', parameters['Split randomize'])

        job.set_param('set_intersection_field', parameters['Field to Intersect'])
        job.set_param('remove_leading_uid', parameters['Ignore Leading UID'])

        job.queue(self.dealjobtools.lane)
        return job


class CreateOrientationDiagnostics():
    def __init__(self, DealJobs_instance):
        self.dealjobtools = DealJobs_instance
        self.default_orientation_diagnostics_job_parameters = {
            'Cone Half-angle (deg)': 20.0,
            'Number of Directions': 3072,

            'SCF Fourier Radius': 23,
            'Symmetry': 'C1',
            'Use this many particles': 10000,

            'Number of Threads': 24
        }

    def QueueOrientationDiagnosticsJob(self, parameters, source_volume_job, source_volume_parameter_name, source_particle_job_list=None, source_particle_parameter_name_list=None, source_mask_job=None, source_mask_parameter_name=None, hostname=None, gpus=None):
        job = self.dealjobtools.cshandle.create_job(self.dealjobtools.project, self.dealjobtools.workspace, 'orientation_diagnostics')
        job.connect('volume', source_volume_job, source_volume_parameter_name)
        if source_particle_job_list is not None:
            for i in range(len(source_particle_job_list)):
                job.connect('particles', source_particle_job_list[i], source_particle_parameter_name_list[i])
        if source_mask_job is not None:
            job.connect('mask', source_mask_job, source_mask_parameter_name)

        job.set_param('od_cone_angle', parameters['Cone Half-angle (deg)'])
        job.set_param('od_num_dirs', parameters['Number of Directions'])

        job.set_param('od_scf_radwn', parameters['SCF Fourier Radius'])
        job.set_param('od_scf_sym', parameters['Symmetry'])
        job.set_param('od_scf_num_part', parameters['Use this many particles'])

        job.set_param('od_nthds', parameters['Number of Threads'])

        if hostname is not None:
            job.queue(self.dealjobtools.lane, hostname, gpus)
        else:
            job.queue(self.dealjobtools.lane)
        return job