import flask
import flask_socketio
import shutil
import psutil
import math
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cryowizardlib.CSLogin as CSLogin
import cryowizardlib.MyJobAPIs as MyJobAPIs
import cryowizardlib.mytoolbox as mytoolbox



filedir = os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

cryowizard_settings = mytoolbox.readyaml(filedir + '/CryoWizard/cryowizard_settings.yml')
WEB_PORT = cryowizard_settings['CryoWizardWebApp_settings']['web_port']

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'some_kind_of_secret_key'
socketio = flask_socketio.SocketIO(app)



@app.route('/')
def index():
    response = flask.make_response(flask.render_template('index.html'))
    return response


def normalize_path(target_folder):
    if (target_folder == '~'):
        normalized_target_folder = os.path.normpath(os.path.expanduser(target_folder))
    else:
        normalized_target_folder = os.path.normpath(target_folder)
    head_slash_count = 0
    for char_ptr in range(len(normalized_target_folder)):
        if not (normalized_target_folder[char_ptr] == '/'):
            break
        head_slash_count = char_ptr
    normalized_target_folder = normalized_target_folder[head_slash_count:]
    if (normalized_target_folder == '/'):
        normalized_parent_folder = None
    else:
        normalized_parent_folder = os.path.normpath(os.path.dirname(normalized_target_folder))
    return normalized_target_folder, normalized_parent_folder


def search_folder_items(normalized_target_folder):
    if os.path.exists(normalized_target_folder):
        response_folder_items = []
        folder_items = os.listdir(normalized_target_folder)
        for item in folder_items:
            if os.path.isdir(normalized_target_folder + '/' + item):
                response_folder_items.append({'name': item, 'type': 'dir'})
            else:
                response_folder_items.append({'name': item, 'type': 'file'})
        sorted_response_folder_items = sorted(response_folder_items, key=lambda x:(x['type'], x['name']))
        return sorted_response_folder_items
    else:
        return []


@socketio.on('project_dir_get_folder_items_action')
def project_dir_get_folder_items_action(single_project_card_index, target_folder):
    normalized_target_folder, _ = normalize_path(target_folder)
    if os.path.exists(normalized_target_folder):
        normalized_target_folder, normalized_parent_folder = normalize_path(target_folder)
        sorted_response_folder_items = search_folder_items(normalized_target_folder)
    else:
        normalized_target_folder, normalized_parent_folder = normalize_path('~')
        sorted_response_folder_items = search_folder_items(normalized_target_folder)
    flask_socketio.emit('js_project_dir_get_folder_items_action', {'single_project_card_index': single_project_card_index, 'response_folder': normalized_target_folder, 'response_folder_items': sorted_response_folder_items, 'response_parent_folder': normalized_parent_folder})


@socketio.on('project_dir_add_new_folder_action')
def project_dir_add_new_folder_action(single_project_card_index, target_folder, new_folder_name):
    normalized_target_folder, normalized_parent_folder = normalize_path(target_folder)
    create_specified_name_flag = False
    try:
        if (new_folder_name is not None) and (len(new_folder_name) > 0) and (not os.path.exists(normalized_target_folder + '/' + (str)(new_folder_name))):
            os.makedirs(normalized_target_folder + '/' + (str)(new_folder_name))
            create_specified_name_flag = True
    except:
        pass
    if not create_specified_name_flag:
        new_folder_ptr = 0
        while True:
            if not os.path.exists(normalized_target_folder + '/New_Folder_' + (str)(new_folder_ptr)):
                os.makedirs(normalized_target_folder + '/New_Folder_' + (str)(new_folder_ptr))
                break
            new_folder_ptr += 1
    sorted_response_folder_items = search_folder_items(normalized_target_folder)
    flask_socketio.emit('js_project_dir_get_folder_items_action', {'single_project_card_index': single_project_card_index, 'response_folder': normalized_target_folder, 'response_folder_items': sorted_response_folder_items, 'response_parent_folder': normalized_parent_folder})


@socketio.on('project_dir_delete_folder_action')
def project_dir_delete_folder_action(single_project_card_index, target_folder, delete_item_name):
    normalized_target_folder, normalized_parent_folder = normalize_path(target_folder)
    if os.path.exists(normalized_target_folder + '/' + delete_item_name):
        if os.path.isdir(normalized_target_folder + '/' + delete_item_name):
            shutil.rmtree(normalized_target_folder + '/' + delete_item_name)
        else:
            os.remove(normalized_target_folder + '/' + delete_item_name)
    sorted_response_folder_items = search_folder_items(normalized_target_folder)
    flask_socketio.emit('js_project_dir_get_folder_items_action', {'single_project_card_index': single_project_card_index, 'response_folder': normalized_target_folder, 'response_folder_items': sorted_response_folder_items, 'response_parent_folder': normalized_parent_folder})


@socketio.on('project_dir_save_button_action')
def project_dir_save_button_action(single_project_card_index, single_project_card_dict, if_first_load):
    response_single_project_card_dict = single_project_card_dict
    project_dir = ''
    if ((response_single_project_card_dict['project_path'] is not None) and os.path.exists(response_single_project_card_dict['project_path'])):
        project_dir = os.path.normpath((str)(response_single_project_card_dict['project_path']))

    if os.path.exists(project_dir):
        if not os.path.exists(project_dir + '/parameters'):
            os.system('cd ' + project_dir + '&&python ' + filedir + '/CryoWizard_main.py --CreateParameterFiles')
        if os.path.exists(project_dir + '/parameters'):
            global_parameters = mytoolbox.readjson(project_dir + '/parameters/parameters.json')
            final_nurefine_parameters = mytoolbox.readjson(project_dir + '/parameters/final_nurefine_parameters.json')
            parameters_folder_item_list = os.listdir(project_dir + '/parameters')

            response_single_project_card_dict['project_path'] = project_dir
            response_single_project_card_dict['cryosparc_username'] = global_parameters['cryosparc_username']
            response_single_project_card_dict['cryosparc_password'] = global_parameters['cryosparc_password']
            response_single_project_card_dict['cryosparc_location_project'] = global_parameters['project']
            response_single_project_card_dict['cryosparc_location_workspace'] = global_parameters['workspace']
            response_single_project_card_dict['refine_symmetry'] = final_nurefine_parameters['Symmetry']

            response_single_project_card_dict["input_card_dict"] = {}
            response_single_project_card_dict["input_card_count"] = 0

            input_card_count = 0
            for file_item in parameters_folder_item_list:
                if (file_item[:len('import_parameters_')] == 'import_parameters_'):
                    import_parameters_type = mytoolbox.readjson(project_dir + '/parameters/' + file_item + '/input_type.json')
                    if (import_parameters_type == 'Movies'):
                        import_movies_parameters = mytoolbox.readjson(project_dir + '/parameters/' + file_item + '/import_movies_parameters.json')
                        ctf_estimation_parameters = mytoolbox.readjson(project_dir + '/parameters/' + file_item + '/patch_ctf_estimation_parameters.json')
                        blob_pick_parameters = mytoolbox.readjson(project_dir + '/parameters/' + file_item + '/blob_picker_parameters.json')

                        movies_data_path = import_movies_parameters['Movies data path'] if import_movies_parameters['Movies data path'] is not None else ''
                        gain_reference_path = import_movies_parameters['Gain reference path'] if import_movies_parameters['Gain reference path'] is not None else ''
                        raw_pixel_size = (str)(import_movies_parameters['Raw pixel size (A)']) if import_movies_parameters['Raw pixel size (A)'] is not None else ''
                        accelerating_voltage = (str)(import_movies_parameters['Accelerating Voltage (kV)']) if import_movies_parameters['Accelerating Voltage (kV)'] is not None else ''
                        spherical_aberration = (str)(import_movies_parameters['Spherical Aberration (mm)']) if import_movies_parameters['Spherical Aberration (mm)'] is not None else ''
                        total_exposure_dose = (str)(import_movies_parameters['Total exposure dose (e/A^2)']) if import_movies_parameters['Total exposure dose (e/A^2)'] is not None else ''
                        minimum_particle_diameter = blob_pick_parameters['Minimum particle diameter (A)'] if blob_pick_parameters['Minimum particle diameter (A)'] is not None else 0
                        maximum_particle_diameter = blob_pick_parameters['Maximum particle diameter (A)'] if blob_pick_parameters['Maximum particle diameter (A)'] is not None else 20
                        particle_diameter = (str)((int)((minimum_particle_diameter + maximum_particle_diameter) / 2.0))
                        number_of_GPUs_to_parallelize = (str)(ctf_estimation_parameters['Number of GPUs to parallelize']) if ctf_estimation_parameters['Number of GPUs to parallelize'] is not None else ''

                        new_input_card_index = 'movie_' + (str)(input_card_count)
                        input_card_count += 1
                        response_single_project_card_dict["input_card_dict"][new_input_card_index] = {
                            'input_type': 'movie',
                            'input_folder_name': file_item,
                            'movies_data_path': movies_data_path,
                            'gain_reference_path': gain_reference_path,
                            'raw_pixel_size': raw_pixel_size,
                            'accelerating_voltage': accelerating_voltage,
                            'spherical_aberration': spherical_aberration,
                            'total_exposure_dose': total_exposure_dose,
                            'particle_diameter': particle_diameter,
                            'gpu_num': number_of_GPUs_to_parallelize
                        }
                        response_single_project_card_dict["input_card_count"] = input_card_count
                    elif (import_parameters_type == 'Micrographs'):
                        import_micrographs_parameters = mytoolbox.readjson(project_dir + '/parameters/' + file_item + '/import_micrographs_parameters.json')
                        ctf_estimation_parameters = mytoolbox.readjson(project_dir + '/parameters/' + file_item + '/patch_ctf_estimation_parameters.json')
                        blob_pick_parameters = mytoolbox.readjson(project_dir + '/parameters/' + file_item + '/blob_picker_parameters.json')

                        micrographs_data_path = import_micrographs_parameters['Micrographs data path'] if import_micrographs_parameters['Micrographs data path'] is not None else ''
                        raw_pixel_size = (str)(import_micrographs_parameters['Pixel size (A)']) if import_micrographs_parameters['Pixel size (A)'] is not None else ''
                        accelerating_voltage = (str)(import_micrographs_parameters['Accelerating Voltage (kV)']) if import_micrographs_parameters['Accelerating Voltage (kV)'] is not None else ''
                        spherical_aberration = (str)(import_micrographs_parameters['Spherical Aberration (mm)']) if import_micrographs_parameters['Spherical Aberration (mm)'] is not None else ''
                        total_exposure_dose = (str)(import_micrographs_parameters['Total exposure dose (e/A^2)']) if import_micrographs_parameters['Total exposure dose (e/A^2)'] is not None else ''
                        minimum_particle_diameter = blob_pick_parameters['Minimum particle diameter (A)'] if blob_pick_parameters['Minimum particle diameter (A)'] is not None else 0
                        maximum_particle_diameter = blob_pick_parameters['Maximum particle diameter (A)'] if blob_pick_parameters['Maximum particle diameter (A)'] is not None else 20
                        particle_diameter = (str)((int)((minimum_particle_diameter + maximum_particle_diameter) / 2.0))
                        number_of_GPUs_to_parallelize = (str)(ctf_estimation_parameters['Number of GPUs to parallelize']) if ctf_estimation_parameters['Number of GPUs to parallelize'] is not None else ''

                        new_input_card_index = 'micrograph_' + (str)(input_card_count)
                        input_card_count += 1
                        response_single_project_card_dict["input_card_dict"][new_input_card_index] = {
                            'input_type': 'micrograph',
                            'input_folder_name': file_item,
                            'micrographs_data_path': micrographs_data_path,
                            'pixel_size': raw_pixel_size,
                            'accelerating_voltage': accelerating_voltage,
                            'spherical_aberration': spherical_aberration,
                            'total_exposure_dose': total_exposure_dose,
                            'particle_diameter': particle_diameter,
                            'gpu_num': number_of_GPUs_to_parallelize
                        }
                        response_single_project_card_dict["input_card_count"] = input_card_count
                    elif (import_parameters_type == 'Particles'):
                        source_particle_jobuid = mytoolbox.readjson(project_dir + '/parameters/' + file_item + '/particle_job_uid.json')['source_particle_job_uid']

                        source_particle_jobuid = source_particle_jobuid if source_particle_jobuid is not None else ''

                        new_input_card_index = 'particle_' + (str)(input_card_count)
                        input_card_count += 1
                        response_single_project_card_dict["input_card_dict"][new_input_card_index] = {
                            'input_type': 'particle',
                            'input_folder_name': file_item,
                            'source_particle_job_uid': source_particle_jobuid
                        }
                        response_single_project_card_dict["input_card_count"] = input_card_count

            flask_socketio.emit('js_project_dir_save_button_action', {'single_project_card_index': single_project_card_index, 'response_single_project_card_dict': response_single_project_card_dict, 'if_path_correct': True, 'if_first_load': if_first_load})
            return

    response_single_project_card_dict['project_path'] = None
    flask_socketio.emit('js_project_dir_save_button_action', {'single_project_card_index': single_project_card_index, 'response_single_project_card_dict': response_single_project_card_dict, 'if_path_correct': False, 'if_first_load': if_first_load})
    return


@socketio.on('parameters_save_button_action')
def parameters_save_button_action(single_project_card_index, single_project_card_dict):
    response_single_project_card_dict = single_project_card_dict
    project_dir = os.path.normpath((str)(response_single_project_card_dict['project_path']))

    global_parameters = mytoolbox.readjson(project_dir + '/parameters/parameters.json')
    final_nurefine_parameters = mytoolbox.readjson(project_dir + '/parameters/final_nurefine_parameters.json')
    orientation_diagnostics_parameters = mytoolbox.readjson(project_dir + '/parameters/orientation_diagnostics_parameters.json')
    parameters_folder_item_list = os.listdir(project_dir + '/parameters')

    exist_input_card_import_parameters_folder_name_list = []
    for input_key, input_card in response_single_project_card_dict['input_card_dict'].items():
        if input_card['input_folder_name'] is not None:
            exist_input_card_import_parameters_folder_name_list.append(input_card['input_folder_name'])
    for file_item in parameters_folder_item_list:
        if (file_item[:len('import_parameters_')] == 'import_parameters_'):
            if file_item not in exist_input_card_import_parameters_folder_name_list:
                shutil.rmtree(project_dir + '/parameters/' + file_item)

    global_parameters['cryosparc_username'] = response_single_project_card_dict['cryosparc_username']
    global_parameters['cryosparc_password'] = response_single_project_card_dict['cryosparc_password']
    global_parameters['project'] = response_single_project_card_dict['cryosparc_location_project']
    global_parameters['workspace'] = response_single_project_card_dict['cryosparc_location_workspace']
    mytoolbox.savetojson(global_parameters, project_dir + '/parameters/parameters.json', False)

    final_nurefine_parameters['Symmetry'] = response_single_project_card_dict['refine_symmetry']
    mytoolbox.savetojson(final_nurefine_parameters, project_dir + '/parameters/final_nurefine_parameters.json', False)
    orientation_diagnostics_parameters['Symmetry'] = response_single_project_card_dict['refine_symmetry']
    mytoolbox.savetojson(orientation_diagnostics_parameters, project_dir + '/parameters/orientation_diagnostics_parameters.json', False)

    for input_key, input_card in response_single_project_card_dict['input_card_dict'].items():
        if (input_card['input_type'] == 'movie'):
            if input_card['input_folder_name'] is None:
                i = 0
                while True:
                    if not os.path.exists(project_dir + '/parameters/import_parameters_' + (str)(i)):
                        input_card['input_folder_name'] = 'import_parameters_' + (str)(i)
                        response_single_project_card_dict['input_card_dict'][input_key]['input_folder_name'] = input_card['input_folder_name']
                        break
                    else:
                        i += 1
                os.system('cd ' + project_dir + '&&python ' + filedir + '/CryoWizard_main.py --CreateImportParameterFiles --input_type movie')

            input_folder_name = input_card['input_folder_name']

            import_movies_parameters = mytoolbox.readjson(project_dir + '/parameters/' + input_folder_name + '/import_movies_parameters.json')
            motion_correction_parameters = mytoolbox.readjson(project_dir + '/parameters/' + input_folder_name + '/patch_motion_correction_parameters.json')
            ctf_estimation_parameters = mytoolbox.readjson(project_dir + '/parameters/' + input_folder_name + '/patch_ctf_estimation_parameters.json')
            blob_pick_parameters = mytoolbox.readjson(project_dir + '/parameters/' + input_folder_name + '/blob_picker_parameters.json')
            extract_parameters = mytoolbox.readjson(project_dir + '/parameters/' + input_folder_name + '/extract_micrographs_parameters.json')

            particle_pixel_size = (float)(input_card['raw_pixel_size']) if (len(input_card['raw_pixel_size']) > 0) else None
            particle_diameter = (int)(input_card['particle_diameter']) if (len(input_card['particle_diameter']) > 0) else None
            import_movies_parameters['Movies data path'] = input_card['movies_data_path'] if (len(input_card['movies_data_path']) > 0) else None
            import_movies_parameters['Gain reference path'] = input_card['gain_reference_path'] if (len(input_card['gain_reference_path']) > 0) else None
            import_movies_parameters['Raw pixel size (A)'] = particle_pixel_size
            import_movies_parameters['Accelerating Voltage (kV)'] = (float)(input_card['accelerating_voltage']) if (len(input_card['accelerating_voltage']) > 0) else None
            import_movies_parameters['Spherical Aberration (mm)'] = (float)(input_card['spherical_aberration']) if (len(input_card['spherical_aberration']) > 0) else None
            import_movies_parameters['Total exposure dose (e/A^2)'] = (float)(input_card['total_exposure_dose']) if (len(input_card['total_exposure_dose']) > 0) else None
            blob_pick_parameters['Minimum particle diameter (A)'] = (particle_diameter - 10) if ((particle_diameter - 10) >= 0) else 0
            blob_pick_parameters['Maximum particle diameter (A)'] = (particle_diameter + 10) if ((particle_diameter + 10) >= 0) else 0
            extract_parameters['Extraction box size (pix)'] = (int)(math.floor((float)(particle_diameter) / particle_pixel_size / 5.0) * 10.0) if ((particle_pixel_size is not None) and (particle_diameter is not None)) else None
            motion_correction_parameters['Number of GPUs to parallelize'] = (int)(input_card['gpu_num']) if (len(input_card['gpu_num']) > 0) else None
            ctf_estimation_parameters['Number of GPUs to parallelize'] = (int)(input_card['gpu_num']) if (len(input_card['gpu_num']) > 0) else None
            extract_parameters['Number of GPUs to parallelize (0 for CPU-only)'] = (int)(input_card['gpu_num']) if (len(input_card['gpu_num']) > 0) else None

            mytoolbox.savetojson(import_movies_parameters, project_dir + '/parameters/' + input_folder_name + '/import_movies_parameters.json', False)
            mytoolbox.savetojson(motion_correction_parameters, project_dir + '/parameters/' + input_folder_name + '/patch_motion_correction_parameters.json', False)
            mytoolbox.savetojson(ctf_estimation_parameters, project_dir + '/parameters/' + input_folder_name + '/patch_ctf_estimation_parameters.json', False)
            mytoolbox.savetojson(blob_pick_parameters, project_dir + '/parameters/' + input_folder_name + '/blob_picker_parameters.json', False)
            mytoolbox.savetojson(extract_parameters, project_dir + '/parameters/' + input_folder_name + '/extract_micrographs_parameters.json', False)
        elif (input_card['input_type'] == 'micrograph'):
            if input_card['input_folder_name'] is None:
                i = 0
                while True:
                    if not os.path.exists(project_dir + '/parameters/import_parameters_' + (str)(i)):
                        input_card['input_folder_name'] = 'import_parameters_' + (str)(i)
                        response_single_project_card_dict['input_card_dict'][input_key]['input_folder_name'] = input_card['input_folder_name']
                        break
                    else:
                        i += 1
                os.system('cd ' + project_dir + '&&python ' + filedir + '/CryoWizard_main.py --CreateImportParameterFiles --input_type micrograph')

            input_folder_name = input_card['input_folder_name']

            import_micrographs_parameters = mytoolbox.readjson(project_dir + '/parameters/' + input_folder_name + '/import_micrographs_parameters.json')
            ctf_estimation_parameters = mytoolbox.readjson(project_dir + '/parameters/' + input_folder_name + '/patch_ctf_estimation_parameters.json')
            blob_pick_parameters = mytoolbox.readjson(project_dir + '/parameters/' + input_folder_name + '/blob_picker_parameters.json')
            extract_parameters = mytoolbox.readjson(project_dir + '/parameters/' + input_folder_name + '/extract_micrographs_parameters.json')

            particle_pixel_size = (float)(input_card['pixel_size']) if (len(input_card['pixel_size']) > 0) else None
            particle_diameter = (int)(input_card['particle_diameter']) if (len(input_card['particle_diameter']) > 0) else None
            import_micrographs_parameters['Micrographs data path'] = input_card['micrographs_data_path'] if (len(input_card['micrographs_data_path']) > 0) else None
            import_micrographs_parameters['Pixel size (A)'] = particle_pixel_size
            import_micrographs_parameters['Accelerating Voltage (kV)'] = (float)(input_card['accelerating_voltage']) if (len(input_card['accelerating_voltage']) > 0) else None
            import_micrographs_parameters['Spherical Aberration (mm)'] = (float)(input_card['spherical_aberration']) if (len(input_card['spherical_aberration']) > 0) else None
            import_micrographs_parameters['Total exposure dose (e/A^2)'] = (float)(input_card['total_exposure_dose']) if (len(input_card['total_exposure_dose']) > 0) else None
            blob_pick_parameters['Minimum particle diameter (A)'] = (particle_diameter - 10) if ((particle_diameter - 10) >= 0) else 0
            blob_pick_parameters['Maximum particle diameter (A)'] = (particle_diameter + 10) if ((particle_diameter + 10) >= 0) else 0
            extract_parameters['Extraction box size (pix)'] = (int)(math.floor((float)(particle_diameter) / particle_pixel_size / 5.0) * 10.0) if ((particle_pixel_size is not None) and (particle_diameter is not None)) else None
            ctf_estimation_parameters['Number of GPUs to parallelize'] = (int)(input_card['gpu_num']) if (len(input_card['gpu_num']) > 0) else None
            extract_parameters['Number of GPUs to parallelize (0 for CPU-only)'] = (int)(input_card['gpu_num']) if (len(input_card['gpu_num']) > 0) else None

            mytoolbox.savetojson(import_micrographs_parameters, project_dir + '/parameters/' + input_folder_name + '/import_micrographs_parameters.json', False)
            mytoolbox.savetojson(ctf_estimation_parameters, project_dir + '/parameters/' + input_folder_name + '/patch_ctf_estimation_parameters.json', False)
            mytoolbox.savetojson(blob_pick_parameters, project_dir + '/parameters/' + input_folder_name + '/blob_picker_parameters.json', False)
            mytoolbox.savetojson(extract_parameters, project_dir + '/parameters/' + input_folder_name + '/extract_micrographs_parameters.json', False)
        elif (input_card['input_type'] == 'particle'):
            if input_card['input_folder_name'] is None:
                i = 0
                while True:
                    if not os.path.exists(project_dir + '/parameters/import_parameters_' + (str)(i)):
                        input_card['input_folder_name'] = 'import_parameters_' + (str)(i)
                        response_single_project_card_dict['input_card_dict'][input_key]['input_folder_name'] = input_card['input_folder_name']
                        break
                    else:
                        i += 1
                os.system('cd ' + project_dir + '&&python ' + filedir + '/CryoWizard_main.py --CreateImportParameterFiles --input_type particle')

            input_folder_name = input_card['input_folder_name']

            source_particle_jobuid = input_card['source_particle_job_uid'] if (len(input_card['source_particle_job_uid']) > 0) else None

            mytoolbox.savetojson({'source_particle_job_uid': source_particle_jobuid}, project_dir + '/parameters/' + input_folder_name + '/particle_job_uid.json', False)

    flask_socketio.emit('js_parameters_save_button_action', {'single_project_card_index': single_project_card_index, 'response_single_project_card_dict': response_single_project_card_dict})


@socketio.on('result_panel_run_button_action')
def result_panel_run_button_action(single_project_card_index, target_project_dir):
    if not os.path.exists(target_project_dir + '/metadata'):
        os.makedirs(target_project_dir + '/metadata')
    if not os.path.exists(target_project_dir + '/metadata/import_and_refine.lock'):
        with open(target_project_dir + '/metadata/import_and_refine.lock', 'w') as f:
            f.write('lock')
        if not os.path.exists(target_project_dir + '/metadata/processpids'):
            os.makedirs(target_project_dir + '/metadata/processpids')
        with open(target_project_dir + '/metadata/import_and_refine.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('pid=$$\n')
            f.write('echo $pid > ' + target_project_dir + '/metadata/processpids/$pid\n')
            f.write('cd ' + target_project_dir + '\n')
            f.write('python ' + filedir + '/CryoWizard_main.py --ImportAndExtract\n')
            f.write('python ' + filedir + '/CryoWizard_main.py --SelectAndRefine\n')
            f.write('rm -f ' + target_project_dir + '/metadata/processpids/$pid\n')
            f.write('rm -f ' + target_project_dir + '/metadata/import_and_refine.lock\n')
        os.system('cd ' + target_project_dir + '&&nohup bash ' + target_project_dir + '/metadata/import_and_refine.sh > ' + target_project_dir + '/metadata/import_and_refine.log 2>&1 &')
    else:
        flask_socketio.emit('js_alert_message', {'single_project_card_index': single_project_card_index, 'message': 'CryoWizard is already running!'})


def killjobs(processpid_dir):
    if not (processpid_dir[-1] == '/'):
        target_dir = processpid_dir + '/'
    else:
        target_dir = processpid_dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    processpids = os.listdir(target_dir)
    for pid in processpids:
        if (pid == 'slurmuid'):
            try:
                with open(target_dir + pid, 'r') as f:
                    slurminfo = f.readlines()
                slurmjobuid = (slurminfo[0].split())[-1]
                os.system('scancel ' + slurmjobuid)
                os.remove(target_dir + pid)
                break
            except:
                return False
        else:
            if not psutil.pid_exists((int)(pid)):
                os.remove(target_dir + pid)
            try:
                mytoolbox.killprocesswithpid((int)(pid))
                os.remove(target_dir + pid)
                break
            except:
                return False
    return True


@socketio.on('result_panel_kill_button_action')
def result_panel_kill_button_action(single_project_card_index, target_project_dir):
    if not os.path.exists(target_project_dir + '/metadata/processpids'):
        os.makedirs(target_project_dir + '/metadata/processpids')
    kill_result = killjobs(target_project_dir + '/metadata/processpids')
    if os.path.exists(target_project_dir + '/metadata/import_and_refine.lock'):
        os.remove(target_project_dir + '/metadata/import_and_refine.lock')
    if kill_result:
        flask_socketio.emit('js_alert_message', {'single_project_card_index': single_project_card_index, 'message': 'Killed!'})
    else:
        flask_socketio.emit('js_alert_message', {'single_project_card_index': single_project_card_index, 'message': 'Killing failed'})


@app.route('/DownloadMap', methods=['GET'])
def DownloadMap():
    target_project_dir = flask.request.args.get('project_dir')
    single_project_card_index = flask.request.args.get('project_card')
    global_parameters = mytoolbox.readjson(target_project_dir + '/parameters/parameters.json')
    cshandle = CSLogin.cshandleclass.GetCryoSPARCHandle(email=global_parameters['cryosparc_username'], password=global_parameters['cryosparc_password'])
    dealjobs = MyJobAPIs.DealJobs(cshandle, global_parameters['project'], global_parameters['workspace'], global_parameters['lane'])
    try:
        if os.path.exists(target_project_dir + '/metadata/curve_job_parents.json'):
            jobuid = mytoolbox.readjson(target_project_dir + '/metadata/curve_job_parents.json')['best_final_nurefine_job_uid']
            jobdir = (str)(cshandle.find_job(dealjobs.project, jobuid).dir())
            if not (jobdir[-1] == '/'):
                jobdir = jobdir + '/'
            ptr = 0
            while True:
                if (ptr > 999):
                    return '<div>Map download failed...</div><script type="text/javascript">setTimeout(function(){window.close();}, 5000);</script>'
                if os.path.exists(jobdir + jobuid + '_' + ((str)(ptr)).zfill(3) + '_volume_map_sharp.mrc'):
                    ptr += 1
                else:
                    break
            target_file_path = jobdir + jobuid + '_' + ((str)(ptr - 1)).zfill(3) + '_volume_map_sharp.mrc'
            return flask.send_file(target_file_path, as_attachment=True)
        else:
            return '<div>Job did not finish yet, please wait...</div><script type="text/javascript">setTimeout(function(){window.close();}, 5000);</script>'
    except:
        return '<div>Map download failed...</div><script type="text/javascript">setTimeout(function(){window.close();}, 5000);</script>'


@socketio.on('result_panel_get_particles_button_action')
def result_panel_get_particles_button_action(single_project_card_index, target_project_dir, truncation_type, truncation_value):
    if (len((str)(truncation_value)) == 0):
        flask_socketio.emit('js_alert_message', {'single_project_card_index': single_project_card_index, 'message': 'Get particle input could not be empty...'})
    elif not os.path.exists(target_project_dir + '/metadata/external_get_confidence_jobuid.json'):
        flask_socketio.emit('js_alert_message', {'single_project_card_index': single_project_card_index, 'message': 'external_get_confidence_jobuid.json do not exists...'})
    elif not os.path.exists(target_project_dir + '/metadata/confidence_dict.json'):
        flask_socketio.emit('js_alert_message', {'single_project_card_index': single_project_card_index, 'message': 'confidence_dict.json do not exists...'})
    else:
        if (truncation_type == 'num'):
            flask_socketio.emit('js_alert_message', {'single_project_card_index': single_project_card_index, 'message': 'Truncate particle job created!'})
            os.system('cd ' + target_project_dir + '&&python ' + filedir + '/CryoWizard_main.py --TruncateParticles --truncation_type num --particle_cutoff_condition ' + (str)(truncation_value))
        elif (truncation_type == 'score'):
            flask_socketio.emit('js_alert_message', {'single_project_card_index': single_project_card_index, 'message': 'Truncate particle job created!'})
            os.system('cd ' + target_project_dir + '&&python ' + filedir + '/CryoWizard_main.py --TruncateParticles --truncation_type score --particle_cutoff_condition ' + (str)(truncation_value))


@socketio.on('result_panel_output_panel_show_data')
def result_panel_output_panel_show_data(single_project_card_index, target_project_dir):
    if ((target_project_dir is not None) and (os.path.exists(target_project_dir + '/metadata/import_and_refine.log'))):
        with open(target_project_dir + '/metadata/import_and_refine.log', 'r') as f:
            response_data = f.readlines()
        flask_socketio.emit('js_result_panel_output_panel_show_data', {'single_project_card_index': single_project_card_index, 'data': response_data})



if (__name__ == '__main__'):
    print('Web service start, press Ctrl+C to quit if you want to stop this web service.', flush=True)
    socketio.run(app, host='0.0.0.0', port=WEB_PORT)