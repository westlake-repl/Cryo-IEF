#!/usr/bin/env python

import shutil
import os

import sys
sys.path.append(os.path.dirname(__file__))

import cryowizardlib.mytoolbox as mytoolbox



def get_job_parameters(cshandle, job_name_list, job_parameters_save_path, job_parameters_map_save_path, job_name_save_path):
    ignore_param_list = ['random_seed']
    # get all_job_parameters
    specs = cshandle.get_job_specs()
    all_job_parameters = {}
    for i in range(len(specs)):
        for j in range(len(specs[i]['contains'])):
            job_name = specs[i]['contains'][j]['name']
            job_parameters = {}
            for param in specs[i]['contains'][j]['params_base']:
                job_parameters[param] = {'title': specs[i]['contains'][j]['params_base'][param]['title'], 'default_value': specs[i]['contains'][j]['params_base'][param]['value']}
            all_job_parameters[job_name] = job_parameters
    # get target job parameters
    job_parameters = None
    job_parameters_map = None
    selected_job_name = None
    for job_name_ptr in range(len(job_name_list)):
        job_name = job_name_list[job_name_ptr]
        if job_name in all_job_parameters:
            job_parameters = {}
            job_parameters_map = {}
            selected_job_name = job_name
            for param in all_job_parameters[job_name]:
                if param in ignore_param_list:
                    continue
                job_parameters[all_job_parameters[job_name][param]['title']] = all_job_parameters[job_name][param]['default_value']
                job_parameters_map[all_job_parameters[job_name][param]['title']] = param
            break
    mytoolbox.savetojson(job_parameters, job_parameters_save_path)
    mytoolbox.savetojson(job_parameters_map, job_parameters_map_save_path)
    mytoolbox.savetojson(selected_job_name, job_name_save_path)



def install(user_email, user_password, license, hostname, port, cryoranker_model_weight_path):
    filedir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))

    # create license file
    mytoolbox.savetojson({'license': (str)(license), 'host': (str)(hostname), 'port': (int)(port)}, filedir + '/CryoWizard/parameters/cs_login_info.json')

    # save model weight info
    cryoranker_inference_settings = mytoolbox.readyaml(filedir + '/CryoRanker/cryoranker_inference_settings.yml')
    cryoranker_inference_settings['path_model_proj'] = os.path.normpath(cryoranker_model_weight_path)
    mytoolbox.savetoyaml(cryoranker_inference_settings, filedir + '/CryoRanker/cryoranker_inference_settings.yml')

    # create cryosparc job parameters
    import cryowizardlib.CSLogin as CSLogin
    cshandle = CSLogin.cshandleclass.GetCryoSPARCHandle(email=user_email, password=user_password)

    import_movies_job_name_list = ['import_movies']
    import_micrographs_job_name_list = ['import_micrographs']
    patch_motion_correction_job_name_list = ['patch_motion_correction_multi']
    patch_ctf_estimation_job_name_list = ['patch_ctf_estimation_multi']
    blob_picker_job_name_list = ['blob_picker_gpu']
    extract_from_micrographs_job_name_list = ['extract_micrographs_multi']
    abinit_job_name_list = ['homo_abinit']
    nurefine_job_name_list = ['nonuniform_refine_new', 'nonuniform_refine']
    restack_particles_job_name_list = ['restack_particles']
    particle_set_tools_job_name_list = ['particle_sets']
    orientation_diagnostics_job_name_list = ['orientation_diagnostics']

    get_job_parameters(cshandle, import_movies_job_name_list, filedir + '/CryoWizard/parameters/import_movies_parameters.json', filedir + '/CryoWizard/parameters/import_movies_parameters_map.json', filedir + '/CryoWizard/parameters/import_movies_job_name.json')
    get_job_parameters(cshandle, import_micrographs_job_name_list, filedir + '/CryoWizard/parameters/import_micrographs_parameters.json', filedir + '/CryoWizard/parameters/import_micrographs_parameters_map.json', filedir + '/CryoWizard/parameters/import_micrographs_job_name.json')
    get_job_parameters(cshandle, patch_motion_correction_job_name_list, filedir + '/CryoWizard/parameters/patch_motion_correction_parameters.json', filedir + '/CryoWizard/parameters/patch_motion_correction_parameters_map.json', filedir + '/CryoWizard/parameters/patch_motion_correction_job_name.json')
    get_job_parameters(cshandle, patch_ctf_estimation_job_name_list, filedir + '/CryoWizard/parameters/patch_ctf_estimation_parameters.json', filedir + '/CryoWizard/parameters/patch_ctf_estimation_parameters_map.json', filedir + '/CryoWizard/parameters/patch_ctf_estimation_job_name.json')
    get_job_parameters(cshandle, blob_picker_job_name_list, filedir + '/CryoWizard/parameters/blob_picker_parameters.json', filedir + '/CryoWizard/parameters/blob_picker_parameters_map.json', filedir + '/CryoWizard/parameters/blob_picker_job_name.json')
    get_job_parameters(cshandle, extract_from_micrographs_job_name_list, filedir + '/CryoWizard/parameters/extract_micrographs_parameters.json', filedir + '/CryoWizard/parameters/extract_micrographs_parameters_map.json', filedir + '/CryoWizard/parameters/extract_micrographs_job_name.json')
    get_job_parameters(cshandle, abinit_job_name_list, filedir + '/CryoWizard/parameters/abinit_parameters.json', filedir + '/CryoWizard/parameters/abinit_parameters_map.json', filedir + '/CryoWizard/parameters/abinit_job_name.json')
    get_job_parameters(cshandle, nurefine_job_name_list, filedir + '/CryoWizard/parameters/nurefine_parameters.json', filedir + '/CryoWizard/parameters/nurefine_parameters_map.json', filedir + '/CryoWizard/parameters/nurefine_job_name.json')
    get_job_parameters(cshandle, restack_particles_job_name_list, filedir + '/CryoWizard/parameters/restack_particles_parameters.json', filedir + '/CryoWizard/parameters/restack_particles_parameters_map.json', filedir + '/CryoWizard/parameters/restack_particles_job_name.json')
    get_job_parameters(cshandle, particle_set_tools_job_name_list, filedir + '/CryoWizard/parameters/particle_set_tools_parameters.json', filedir + '/CryoWizard/parameters/particle_set_tools_parameters_map.json', filedir + '/CryoWizard/parameters/particle_set_tools_job_name.json')
    get_job_parameters(cshandle, orientation_diagnostics_job_name_list, filedir + '/CryoWizard/parameters/orientation_diagnostics_parameters.json', filedir + '/CryoWizard/parameters/orientation_diagnostics_parameters_map.json', filedir + '/CryoWizard/parameters/orientation_diagnostics_job_name.json')

    nurefine_parameters = mytoolbox.readjson(filedir + '/CryoWizard/parameters/nurefine_parameters.json')
    final_nurefine_parameters = mytoolbox.readjson(filedir + '/CryoWizard/parameters/nurefine_parameters.json')

    nurefine_parameters['Number of extra final passes'] = 3
    nurefine_parameters['Initial lowpass resolution (A)'] = 15
    mytoolbox.savetojson(nurefine_parameters, filedir + '/CryoWizard/parameters/nurefine_parameters.json')

    final_nurefine_parameters['Number of extra final passes'] = 3
    final_nurefine_parameters['Initial lowpass resolution (A)'] = 15
    mytoolbox.savetojson(final_nurefine_parameters, filedir + '/CryoWizard/parameters/final_nurefine_parameters.json')



def check_import_parameters_ptr(parameter_folder_path):
    normalized_parameter_folder_path = os.path.normpath(parameter_folder_path)
    i = 0
    while True:
        if not os.path.exists(normalized_parameter_folder_path + '/import_parameters_' + (str)(i)):
            os.makedirs(normalized_parameter_folder_path + '/import_parameters_' + (str)(i))
            break
        else:
            i += 1
    return 'import_parameters_' + (str)(i)

def CreateImportParametersFiles(project_dir, input_type, import_parameters_folder_name):
    globaldir = os.path.normpath(project_dir)
    source_python_file_path = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
    try:
        if (input_type == 'movie'):
            for item in ['import_movies_parameters.json', 'patch_motion_correction_parameters.json', 'patch_ctf_estimation_parameters.json', 'blob_picker_parameters.json', 'extract_micrographs_parameters.json']:
                shutil.copy(source_python_file_path + '/CryoWizard/parameters/' + item, globaldir + '/parameters/' + import_parameters_folder_name + '/' + item)
            mytoolbox.savetojson('Movies', globaldir + '/parameters/' + import_parameters_folder_name + '/input_type.json', False)
        elif (input_type == 'micrograph'):
            for item in ['import_micrographs_parameters.json', 'patch_ctf_estimation_parameters.json', 'blob_picker_parameters.json', 'extract_micrographs_parameters.json']:
                shutil.copy(source_python_file_path + '/CryoWizard/parameters/' + item, globaldir + '/parameters/' + import_parameters_folder_name + '/' + item)
            mytoolbox.savetojson('Micrographs', globaldir + '/parameters/' + import_parameters_folder_name + '/input_type.json', False)
        elif (input_type == 'particle'):
            mytoolbox.savetojson('Particles', globaldir + '/parameters/' + import_parameters_folder_name + '/input_type.json', False)
            mytoolbox.savetojson({'source_particle_job_uid': 'J'}, globaldir + '/parameters/' + import_parameters_folder_name + '/particle_job_uid.json', False)
        elif (input_type == 'extension_movie'):
            mytoolbox.savetojson('extension_movies', globaldir + '/parameters/' + import_parameters_folder_name + '/input_type.json', False)
            mytoolbox.savetojson({'source_job_uid': 'J', 'source_group_name': None}, globaldir + '/parameters/' + import_parameters_folder_name + '/job_uid.json', False)
            for item in ['patch_motion_correction_parameters.json', 'patch_ctf_estimation_parameters.json', 'blob_picker_parameters.json', 'extract_micrographs_parameters.json']:
                shutil.copy(source_python_file_path + '/CryoWizard/parameters/' + item, globaldir + '/parameters/' + import_parameters_folder_name + '/' + item)
        elif (input_type == 'extension_micrograph'):
            mytoolbox.savetojson('extension_micrographs', globaldir + '/parameters/' + import_parameters_folder_name + '/input_type.json', False)
            mytoolbox.savetojson({'source_job_uid': 'J', 'source_group_name': None}, globaldir + '/parameters/' + import_parameters_folder_name + '/job_uid.json', False)
            for item in ['patch_ctf_estimation_parameters.json', 'blob_picker_parameters.json', 'extract_micrographs_parameters.json']:
                shutil.copy(source_python_file_path + '/CryoWizard/parameters/' + item, globaldir + '/parameters/' + import_parameters_folder_name + '/' + item)
        elif (input_type == 'extension_particle'):
            mytoolbox.savetojson('extension_particles', globaldir + '/parameters/' + import_parameters_folder_name + '/input_type.json', False)
            mytoolbox.savetojson({'source_job_uid': 'J', 'source_group_name': None}, globaldir + '/parameters/' + import_parameters_folder_name + '/job_uid.json', False)
        else:
            shutil.rmtree(globaldir + '/parameters/' + import_parameters_folder_name)
            print('Input type error...', flush=True)
            exit()
    except:
        shutil.rmtree(globaldir + '/parameters/' + import_parameters_folder_name)
        print('Error...', flush=True)
        exit()



def CreateParametersFiles(project_dir):
    globaldir = os.path.normpath(project_dir)
    if not os.path.exists(globaldir + '/parameters/'):
        os.makedirs(globaldir + '/parameters/')
        source_python_file_path = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
        for item in ['parameters.json', 'abinit_parameters.json', 'nurefine_parameters.json', 'final_nurefine_parameters.json', 'restack_particles_parameters.json', 'orientation_diagnostics_parameters.json', 'GetConfidence.sh']:
            shutil.copy(source_python_file_path + '/CryoWizard/parameters/' + item, globaldir + '/parameters/' + item)

