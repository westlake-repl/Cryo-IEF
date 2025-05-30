#!/usr/bin/env python

from argparse import ArgumentParser
import math
import os
import CryoWizard.cryowizardlib.mytoolbox as mytoolbox
import yaml
from easydict import EasyDict



if __name__ == '__main__':
    '''get config'''
    parser = ArgumentParser()

    parser.add_argument('--path_result_dir', default=None, type=str)

    parser.add_argument('--CryoWizardInstall', action="store_true")
    parser.add_argument('--CreateParameterFiles', action="store_true")
    parser.add_argument('--CreateImportParameterFiles', action="store_true")
    parser.add_argument('--ImportAndExtract', action="store_true")
    parser.add_argument('--SelectAndRefine', action="store_true")
    parser.add_argument('--TruncateParticles', action="store_true")

    # CryoWizardInstall parameters
    parser.add_argument('--cryosparc_username', default=None, type=str)
    parser.add_argument('--cryosparc_password', default=None, type=str)
    parser.add_argument('--cryosparc_license', default=None, type=str)
    parser.add_argument('--cryosparc_hostname', default=None, type=str)
    parser.add_argument('--cryosparc_port', default=None, type=str)
    parser.add_argument('--cryoranker_model_weight', default=None, type=str)

    # CreateParameterFiles parameters
    # parser.add_argument('--cryosparc_username', default=None, type=str)   # repeat parameter
    # parser.add_argument('--cryosparc_password', default=None, type=str)   # repeat parameter
    parser.add_argument('--cryosparc_project', default=None, type=str)
    parser.add_argument('--cryosparc_workspace', default=None, type=str)
    parser.add_argument('--cryosparc_lane', default=None, type=str)
    parser.add_argument('--slurm', action="store_true")
    parser.add_argument('--inference_gpu_ids', default=None, type=str)

    # CreateImportParameterFiles parameters
    parser.add_argument('--input_type', default=None, type=str)
    parser.add_argument('--symmetry', default=None, type=str)
    ## movie/micrograph parameters
    parser.add_argument('--movies_data_path', default=None, type=str)
    parser.add_argument('--gain_reference_path', default=None, type=str)
    parser.add_argument('--micrographs_data_path', default=None, type=str)
    parser.add_argument('--raw_pixel_size', default=None, type=float)
    parser.add_argument('--accelerating_voltage', default=None, type=float)
    parser.add_argument('--spherical_aberration', default=None, type=float)
    parser.add_argument('--total_exposure_dose', default=None, type=float)
    parser.add_argument('--particle_diameter', default=None, type=int)
    parser.add_argument('--gpu_num', default=None, type=int)
    ## particle parameters
    parser.add_argument('--particle_job_uid', default=None, type=str)


    # TruncateParticles parameters
    parser.add_argument('--truncation_type', default=None, type=str)
    parser.add_argument('--particle_cutoff_condition', default=None, type=float)

    args = parser.parse_args()

    args.path_proj_dir = os.path.dirname(os.path.abspath(__file__))


    print('Confirming parameters...', flush=True)

    # cfg = EasyDict()
    # with open(args.path_proj_dir + '/CryoWizard/cryowizard_settings.yml', 'r') as stream:
    #     config = yaml.safe_load(stream)
    # for k, v in config.items():
    #     cfg[k] = v

    cfg = EasyDict(mytoolbox.readyaml(args.path_proj_dir + '/CryoWizard/cryowizard_settings.yml'))

    if args.path_result_dir is not None:
        cfg['path_result_dir'] = args.path_result_dir
    if args.CryoWizardInstall:
        cfg['CryoWizardInstall'] = True
    if args.CreateParameterFiles:
        cfg['CreateParameterFiles'] = True
    if args.CreateImportParameterFiles:
        cfg['CreateImportParameterFiles'] = True
    if args.ImportAndExtract:
        cfg['ImportAndExtract'] = True
    if args.SelectAndRefine:
        cfg['SelectAndRefine'] = True
    if args.TruncateParticles:
        cfg['TruncateParticles'] = True

    if args.cryosparc_username is not None:
        cfg['CryoWizardInstall_settings']['cryosparc_username'] = args.cryosparc_username
    if args.cryosparc_password is not None:
        cfg['CryoWizardInstall_settings']['cryosparc_password'] = args.cryosparc_password
    if args.cryosparc_license is not None:
        cfg['CryoWizardInstall_settings']['cryosparc_license'] = args.cryosparc_license
    if args.cryosparc_hostname is not None:
        cfg['CryoWizardInstall_settings']['cryosparc_hostname'] = args.cryosparc_hostname
    if args.cryosparc_port is not None:
        cfg['CryoWizardInstall_settings']['cryosparc_port'] = args.cryosparc_port
    if args.cryoranker_model_weight is not None:
        cfg['CryoWizardInstall_settings']['cryoranker_model_weight'] = args.cryoranker_model_weight

    if args.cryosparc_project is not None:
        cfg['CreateParameterFiles_settings']['cryosparc_project'] = args.cryosparc_project
    if args.cryosparc_workspace is not None:
        cfg['CreateParameterFiles_settings']['cryosparc_workspace'] = args.cryosparc_workspace
    if args.cryosparc_lane is not None:
        cfg['CreateParameterFiles_settings']['cryosparc_lane'] = args.cryosparc_lane
    if args.slurm:
        cfg['CreateParameterFiles_settings']['if_slurm'] = True
    if args.truncation_type is not None:
        cfg['CreateParameterFiles_settings']['inference_gpu_ids'] = args.inference_gpu_ids

    if args.input_type is not None:
        cfg['CreateImportParameterFiles_settings']['input_type'] = args.input_type
    if args.symmetry is not None:
        cfg['CreateImportParameterFiles_settings']['symmetry'] = args.symmetry
    if args.movies_data_path is not None:
        cfg['CreateImportParameterFiles_settings']['movies_data_path'] = args.movies_data_path
    if args.gain_reference_path is not None:
        cfg['CreateImportParameterFiles_settings']['gain_reference_path'] = args.gain_reference_path
    if args.micrographs_data_path is not None:
        cfg['CreateImportParameterFiles_settings']['micrographs_data_path'] = args.micrographs_data_path
    if args.raw_pixel_size is not None:
        cfg['CreateImportParameterFiles_settings']['raw_pixel_size'] = args.raw_pixel_size
    if args.accelerating_voltage is not None:
        cfg['CreateImportParameterFiles_settings']['accelerating_voltage'] = args.accelerating_voltage
    if args.spherical_aberration is not None:
        cfg['CreateImportParameterFiles_settings']['spherical_aberration'] = args.spherical_aberration
    if args.total_exposure_dose is not None:
        cfg['CreateImportParameterFiles_settings']['total_exposure_dose'] = args.total_exposure_dose
    if args.particle_diameter is not None:
        cfg['CreateImportParameterFiles_settings']['particle_diameter'] = args.particle_diameter
    if args.gpu_num is not None:
        cfg['CreateImportParameterFiles_settings']['gpu_num'] = args.gpu_num
    if args.particle_job_uid is not None:
        cfg['CreateImportParameterFiles_settings']['particle_job_uid'] = args.particle_job_uid

    if args.truncation_type is not None:
        cfg['TruncateParticles_settings']['truncation_type'] = args.truncation_type
    if args.particle_cutoff_condition is not None:
        cfg['TruncateParticles_settings']['particle_cutoff_condition'] = args.particle_cutoff_condition


    if cfg['path_result_dir'] is not None:
        globaldir = cfg['path_result_dir']
    else:
        globaldir = os.path.normpath(os.getcwd())


    if cfg['CryoWizardInstall']:
        print('Start installation...', flush=True)
        from CryoWizard import initialize
        initialize.install(cfg['CryoWizardInstall_settings']['cryosparc_username'],
                           cfg['CryoWizardInstall_settings']['cryosparc_password'],
                           cfg['CryoWizardInstall_settings']['cryosparc_license'],
                           cfg['CryoWizardInstall_settings']['cryosparc_hostname'],
                           cfg['CryoWizardInstall_settings']['cryosparc_port'],
                           cfg['CryoWizardInstall_settings']['cryoranker_model_weight'])
        print('Installation done', flush=True)

    if cfg['CreateParameterFiles']:
        print('Creating base parameters...', flush=True)
        from CryoWizard import initialize
        initialize.CreateParametersFiles(globaldir)
        parameters = mytoolbox.readjson(globaldir + '/parameters/parameters.json')
        parameters['cryosparc_username'] = cfg['CryoWizardInstall_settings']['cryosparc_username']
        parameters['cryosparc_password'] = cfg['CryoWizardInstall_settings']['cryosparc_password']
        parameters['project'] = cfg['CreateParameterFiles_settings']['cryosparc_project']
        parameters['workspace'] = cfg['CreateParameterFiles_settings']['cryosparc_workspace']
        parameters['lane'] = cfg['CreateParameterFiles_settings']['cryosparc_lane']
        parameters['if_slurm'] = cfg['CreateParameterFiles_settings']['if_slurm']
        parameters['inference_gpu_ids'] = cfg['CreateParameterFiles_settings']['inference_gpu_ids']
        mytoolbox.savetojson(parameters, globaldir + '/parameters/parameters.json', False)
        if cfg['CreateImportParameterFiles_settings']['symmetry'] is not None:
            final_nurefine_parameters = mytoolbox.readjson(globaldir + '/parameters/final_nurefine_parameters.json')
            final_nurefine_parameters['Symmetry'] = cfg['CreateImportParameterFiles_settings']['symmetry']
            mytoolbox.savetojson(final_nurefine_parameters, globaldir + '/parameters/final_nurefine_parameters.json', False)
            orientation_diagnostics_parameters = mytoolbox.readjson(globaldir + '/parameters/orientation_diagnostics_parameters.json')
            orientation_diagnostics_parameters['Symmetry'] = cfg['CreateImportParameterFiles_settings']['symmetry']
            mytoolbox.savetojson(orientation_diagnostics_parameters, globaldir + '/parameters/orientation_diagnostics_parameters.json', False)
        print('Creating base parameters done', flush=True)

    if cfg['CreateImportParameterFiles']:
        print('Creating import parameters...', flush=True)
        from CryoWizard import initialize
        import_parameters_folder_name = initialize.check_import_parameters_ptr(globaldir + '/parameters')
        initialize.CreateImportParametersFiles(globaldir, cfg['CreateImportParameterFiles_settings']['input_type'], import_parameters_folder_name)
        if (cfg['CreateImportParameterFiles_settings']['input_type'] == 'movie'):
            import_movies_parameters = mytoolbox.readjson(globaldir + '/parameters/' + import_parameters_folder_name + '/import_movies_parameters.json')
            motion_correction_parameters = mytoolbox.readjson(globaldir + '/parameters/' + import_parameters_folder_name + '/patch_motion_correction_parameters.json')
            ctf_estimation_parameters = mytoolbox.readjson(globaldir + '/parameters/' + import_parameters_folder_name + '/patch_ctf_estimation_parameters.json')
            blob_pick_parameters = mytoolbox.readjson(globaldir + '/parameters/' + import_parameters_folder_name + '/blob_picker_parameters.json')
            extract_parameters = mytoolbox.readjson(globaldir + '/parameters/' + import_parameters_folder_name + '/extract_micrographs_parameters.json')
            if cfg['CreateImportParameterFiles_settings']['movies_data_path'] is not None:
                import_movies_parameters['Movies data path'] = (str)(cfg['CreateImportParameterFiles_settings']['movies_data_path'])
            if cfg['CreateImportParameterFiles_settings']['gain_reference_path'] is not None:
                import_movies_parameters['Gain reference path'] = str(cfg['CreateImportParameterFiles_settings']['gain_reference_path'])
            if cfg['CreateImportParameterFiles_settings']['raw_pixel_size'] is not None:
                import_movies_parameters['Raw pixel size (A)'] = (float)(cfg['CreateImportParameterFiles_settings']['raw_pixel_size'])
            if cfg['CreateImportParameterFiles_settings']['accelerating_voltage'] is not None:
                import_movies_parameters['Accelerating Voltage (kV)'] = (float)(cfg['CreateImportParameterFiles_settings']['accelerating_voltage'])
            if cfg['CreateImportParameterFiles_settings']['spherical_aberration'] is not None:
                import_movies_parameters['Spherical Aberration (mm)'] = (float)(cfg['CreateImportParameterFiles_settings']['spherical_aberration'])
            if cfg['CreateImportParameterFiles_settings']['total_exposure_dose'] is not None:
                import_movies_parameters['Total exposure dose (e/A^2)'] = (float)(cfg['CreateImportParameterFiles_settings']['total_exposure_dose'])
            mytoolbox.savetojson(import_movies_parameters, globaldir + '/parameters/' + import_parameters_folder_name + '/import_movies_parameters.json', False)
            if cfg['CreateImportParameterFiles_settings']['particle_diameter'] is not None:
                particle_pixel_size = (float)(cfg['CreateImportParameterFiles_settings']['raw_pixel_size'])
                particle_diameter = (int)(cfg['CreateImportParameterFiles_settings']['particle_diameter'])
                blob_pick_parameters['Minimum particle diameter (A)'] = (particle_diameter - 10) if ((particle_diameter - 10) >= 0) else 0
                blob_pick_parameters['Maximum particle diameter (A)'] = (particle_diameter + 10) if ((particle_diameter + 10) >= 0) else 0
                extract_parameters['Extraction box size (pix)'] = (int)(math.floor((float)(particle_diameter) / particle_pixel_size / 5.0) * 10.0)
            if cfg['CreateImportParameterFiles_settings']['gpu_num'] is not None:
                motion_correction_parameters['Number of GPUs to parallelize'] = (int)(cfg['CreateImportParameterFiles_settings']['gpu_num'])
                ctf_estimation_parameters['Number of GPUs to parallelize'] = (int)(cfg['CreateImportParameterFiles_settings']['gpu_num'])
                extract_parameters['Number of GPUs to parallelize (0 for CPU-only)'] = (int)(cfg['CreateImportParameterFiles_settings']['gpu_num'])
            mytoolbox.savetojson(motion_correction_parameters, globaldir + '/parameters/' + import_parameters_folder_name + '/patch_motion_correction_parameters.json', False)
            mytoolbox.savetojson(ctf_estimation_parameters, globaldir + '/parameters/' + import_parameters_folder_name + '/patch_ctf_estimation_parameters.json', False)
            mytoolbox.savetojson(blob_pick_parameters, globaldir + '/parameters/' + import_parameters_folder_name + '/blob_picker_parameters.json', False)
            mytoolbox.savetojson(extract_parameters, globaldir + '/parameters/' + import_parameters_folder_name + '/extract_micrographs_parameters.json', False)
        elif (cfg['CreateImportParameterFiles_settings']['input_type'] == 'micrograph'):
            import_micrographs_parameters = mytoolbox.readjson(globaldir + '/parameters/' + import_parameters_folder_name + '/import_micrographs_parameters.json')
            ctf_estimation_parameters = mytoolbox.readjson(globaldir + '/parameters/' + import_parameters_folder_name + '/patch_ctf_estimation_parameters.json')
            blob_pick_parameters = mytoolbox.readjson(globaldir + '/parameters/' + import_parameters_folder_name + '/blob_picker_parameters.json')
            extract_parameters = mytoolbox.readjson(globaldir + '/parameters/' + import_parameters_folder_name + '/extract_micrographs_parameters.json')
            if cfg['CreateImportParameterFiles_settings']['micrographs_data_path'] is not None:
                import_micrographs_parameters['Micrographs data path'] = (str)(cfg['CreateImportParameterFiles_settings']['micrographs_data_path'])
            if cfg['CreateImportParameterFiles_settings']['raw_pixel_size'] is not None:
                import_micrographs_parameters['Pixel size (A)'] = (float)(cfg['CreateImportParameterFiles_settings']['raw_pixel_size'])
            if cfg['CreateImportParameterFiles_settings']['accelerating_voltage'] is not None:
                import_micrographs_parameters['Accelerating Voltage (kV)'] = (float)(cfg['CreateImportParameterFiles_settings']['accelerating_voltage'])
            if cfg['CreateImportParameterFiles_settings']['spherical_aberration'] is not None:
                import_micrographs_parameters['Spherical Aberration (mm)'] = (float)(cfg['CreateImportParameterFiles_settings']['spherical_aberration'])
            if cfg['CreateImportParameterFiles_settings']['total_exposure_dose'] is not None:
                import_micrographs_parameters['Total exposure dose (e/A^2)'] = (float)(cfg['CreateImportParameterFiles_settings']['total_exposure_dose'])
            mytoolbox.savetojson(import_micrographs_parameters, globaldir + '/parameters/' + import_parameters_folder_name + '/import_micrographs_parameters.json', False)
            if cfg['CreateImportParameterFiles_settings']['particle_diameter'] is not None:
                particle_pixel_size = (float)(cfg['CreateImportParameterFiles_settings']['raw_pixel_size'])
                particle_diameter = (int)(cfg['CreateImportParameterFiles_settings']['particle_diameter'])
                blob_pick_parameters['Minimum particle diameter (A)'] = (particle_diameter - 10) if ((particle_diameter - 10) >= 0) else 0
                blob_pick_parameters['Maximum particle diameter (A)'] = (particle_diameter + 10) if ((particle_diameter + 10) >= 0) else 0
                extract_parameters['Extraction box size (pix)'] = (int)(math.floor((float)(particle_diameter) / particle_pixel_size / 5.0) * 10.0)
            if cfg['CreateImportParameterFiles_settings']['gpu_num'] is not None:
                ctf_estimation_parameters['Number of GPUs to parallelize'] = (int)(cfg['CreateImportParameterFiles_settings']['gpu_num'])
                extract_parameters['Number of GPUs to parallelize (0 for CPU-only)'] = (int)(cfg['CreateImportParameterFiles_settings']['gpu_num'])
            mytoolbox.savetojson(ctf_estimation_parameters, globaldir + '/parameters/' + import_parameters_folder_name + '/patch_ctf_estimation_parameters.json', False)
            mytoolbox.savetojson(blob_pick_parameters, globaldir + '/parameters/' + import_parameters_folder_name + '/blob_picker_parameters.json', False)
            mytoolbox.savetojson(extract_parameters, globaldir + '/parameters/' + import_parameters_folder_name + '/extract_micrographs_parameters.json', False)
        elif (cfg['CreateImportParameterFiles_settings']['input_type'] == 'particle'):
            particle_job_uid_parameters = mytoolbox.readjson(globaldir + '/parameters/' + import_parameters_folder_name + '/particle_job_uid.json')
            if cfg['CreateImportParameterFiles_settings']['particle_job_uid'] is not None:
                particle_job_uid_parameters['source_particle_job_uid'] = (str)(cfg['CreateImportParameterFiles_settings']['particle_job_uid'])
            mytoolbox.savetojson(particle_job_uid_parameters, globaldir + '/parameters/' + import_parameters_folder_name + '/particle_job_uid.json', False)
        print('Creating import parameters done', flush=True)

    if cfg['ImportAndExtract']:
        print('Ready to import and extract particles...', flush=True)
        from CryoWizard import particles
        particles.ImportAndExtract(globaldir)

    if cfg['SelectAndRefine']:
        print('Ready to select particles and refine...', flush=True)
        from CryoWizard import refine
        refine.SelectAndRefine(globaldir)

    if cfg['TruncateParticles']:
        from CryoWizard import particles
        particles.TruncateParticles(globaldir, cfg['TruncateParticles_settings']['truncation_type'], cfg['TruncateParticles_settings']['particle_cutoff_condition'])
