#!/usr/bin/env python

import sys
import os

import MyLib.CSLogin as CSLogin
import MyLib.MyJobAPIs as MyJobAPIs
import MyLib.mytoolbox as mytoolbox




globaldir = os.getcwd()
if not (globaldir[-1] == '/'):
    globaldir = globaldir + '/'
parameters = mytoolbox.readjson(globaldir + 'parameters/parameters.json')
cshandle = CSLogin.cshandleclass.GetCryoSPARCHandle(email=parameters['cryosparc_username'], password=parameters['cryosparc_password'])
dealjobs = MyJobAPIs.DealJobs(cshandle, parameters['project'], parameters['workspace'], parameters['lane'])



def TruncateParticlesWithQuantity(particle_number, DealJobs_instance=dealjobs):
    globaldir = os.getcwd()
    if not (globaldir[-1] == '/'):
        globaldir = globaldir + '/'
    if (particle_number <= 0):
        print('Target particle number is illegal...', flush=True)
        return None, None

    # 判断是否存在get confidence job以及对应的confidence_dict文件
    '''Check whether the get confidence job exists and the corresponding confide_dict file exists'''
    if not os.path.exists(globaldir + 'metadata/external_get_confidence_jobuid.json'):
        print('external_get_confidence_jobuid.json do not exists...', flush=True)
        return None, None
    if not os.path.exists(globaldir + 'metadata/confidence_dict.json'):
        print('confidence_dict.json do not exists...', flush=True)
        return None, None
    source_particle_job = mytoolbox.readjson(globaldir + 'metadata/external_get_confidence_jobuid.json')['external_get_confidence_jobuid']
    confidence_dict = mytoolbox.readjson(globaldir + 'metadata/confidence_dict.json')

    # 创建空白 external job
    '''Create an empty external job'''
    project = DealJobs_instance.cshandle.find_project(DealJobs_instance.project)
    job = project.create_external_job(DealJobs_instance.workspace)

    # 创建和链接输入
    '''Create and link input'''
    add_input_name = 'input_particles'
    job.add_input(type='particle', name=add_input_name, min=1, slots=['blob', 'ctf'], title='Input particles for selection')
    job.connect(target_input=add_input_name, source_job_uid=source_particle_job, source_output='particles_with_confidence')

    # 获取输入的cs信息，包含add_input()函数的slots参数指定的所有内容
    '''Get the cs information of the input, including all contents specified by the slots parameter of the add_input() function'''

    input_particles_dataset = job.load_input(name=add_input_name)

    with job.run():
        selected_uids = []
        sorted_confidence_dict = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
        safe_particle_number = particle_number if (particle_number <= len(sorted_confidence_dict)) else len(sorted_confidence_dict)
        for confidence_dict_item_ptr in range(safe_particle_number):
            selected_uids.append(sorted_confidence_dict[confidence_dict_item_ptr][0])

        output_particles_dataset = input_particles_dataset.query({'uid': selected_uids})

        # 创建输出，并保存本job的cs信息
        '''Create output and save the cs information of this job'''
        # cs里修改过的slot需要在add_output()函数的slots参数中全部指定来用于保存为新的，剩下的内容passthrough参数会自动从input中拉取并保存在xxx_passthrough.cs中
        '''Slots that have been modified in cs need to be specified in the slots parameter of the add_output() function to be used as new, and the remaining contents will be automatically pulled from the input and saved in xxx_passthrough.cs'''
        add_output_name = 'particles_selected'
        job.add_output(type='particle', name=add_output_name, slots=['blob'], passthrough=add_input_name)
        job.save_output(add_output_name, output_particles_dataset)

    return job, safe_particle_number



new_truncate_particles_job, final_particle_number = TruncateParticlesWithQuantity((int)(sys.argv[1]))
print(new_truncate_particles_job.uid, 'finished, containing', final_particle_number, 'particles', flush=True)