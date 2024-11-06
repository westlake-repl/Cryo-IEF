import flask
import subprocess
import shutil
import psutil
import signal
import math
import time
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import MyLib.CSLogin as CSLogin
import MyLib.MyJobAPIs as MyJobAPIs
import MyLib.mytoolbox as mytoolbox



def createinputdiv(source_input_parameters_folder):
    if not (source_input_parameters_folder[-1] == '/'):
        input_parameters_folder = source_input_parameters_folder + '/'
    else:
        input_parameters_folder = source_input_parameters_folder
    source_input_type = mytoolbox.readjson(input_parameters_folder + 'input_type.json')
    if (source_input_type == 'Movies'):
        forlder_name = os.path.basename(os.path.dirname(input_parameters_folder))

        import_movies_parameters = mytoolbox.readjson(input_parameters_folder + 'import_movies_parameters.json')
        ctf_estimation_parameters = mytoolbox.readjson(input_parameters_folder + 'patch_ctf_estimation_parameters.json')
        blob_pick_parameters = mytoolbox.readjson(input_parameters_folder + 'blob_picker_parameters.json')

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

        return '''
            <div class="single_inputparameter_div">
                <div class="single_item_div">
                    <div class="input_prompt_2"></div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter inputbox_foldername" type="text" name="''' + forlder_name + '''_folder_name" value="''' + forlder_name + '''[Movies]" readonly="readonly"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Movies data path: </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_movies_data_path" value="''' + movies_data_path + '''"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Gain reference path: </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_gain_reference_path" value="''' + gain_reference_path + '''"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Raw pixel size (A): </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_raw_pixel_size" value="''' + raw_pixel_size + '''"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Accelerating Voltage (kV): </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_accelerating_voltage" value="''' + accelerating_voltage + '''"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Spherical Aberration (mm): </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_spherical_aberration" value="''' + spherical_aberration + '''"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Total exposure dose (e/A^2): </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_total_exposure_dose" value="''' + total_exposure_dose + '''"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Particle diameter (A): </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_particle_diameter" value="''' + particle_diameter + '''"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Number of GPUs to parallelize: </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_number_of_GPUs_to_parallelize" value="''' + number_of_GPUs_to_parallelize + '''"></div>
                </div>
            </div>
        '''
    elif (source_input_type == 'Micrographs'):
        forlder_name = os.path.basename(os.path.dirname(input_parameters_folder))

        import_micrographs_parameters = mytoolbox.readjson(input_parameters_folder + 'import_micrographs_parameters.json')
        ctf_estimation_parameters = mytoolbox.readjson(input_parameters_folder + 'patch_ctf_estimation_parameters.json')
        blob_pick_parameters = mytoolbox.readjson(input_parameters_folder + 'blob_picker_parameters.json')

        micrographs_data_path = import_micrographs_parameters['Micrographs data path'] if import_micrographs_parameters['Micrographs data path'] is not None else ''
        raw_pixel_size = (str)(import_micrographs_parameters['Pixel size (A)']) if import_micrographs_parameters['Pixel size (A)'] is not None else ''
        accelerating_voltage = (str)(import_micrographs_parameters['Accelerating Voltage (kV)']) if import_micrographs_parameters['Accelerating Voltage (kV)'] is not None else ''
        spherical_aberration = (str)(import_micrographs_parameters['Spherical Aberration (mm)']) if import_micrographs_parameters['Spherical Aberration (mm)'] is not None else ''
        total_exposure_dose = (str)(import_micrographs_parameters['Total exposure dose (e/A^2)']) if import_micrographs_parameters['Total exposure dose (e/A^2)'] is not None else ''
        minimum_particle_diameter = blob_pick_parameters['Minimum particle diameter (A)'] if blob_pick_parameters['Minimum particle diameter (A)'] is not None else 0
        maximum_particle_diameter = blob_pick_parameters['Maximum particle diameter (A)'] if blob_pick_parameters['Maximum particle diameter (A)'] is not None else 20
        particle_diameter = (str)((int)((minimum_particle_diameter + maximum_particle_diameter) / 2.0))
        number_of_GPUs_to_parallelize = (str)(ctf_estimation_parameters['Number of GPUs to parallelize']) if ctf_estimation_parameters['Number of GPUs to parallelize'] is not None else ''

        return '''
            <div class="single_inputparameter_div">
                <div class="single_item_div">
                    <div class="input_prompt_2"></div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter inputbox_foldername" type="text" name="''' + forlder_name + '''_folder_name" value="''' + forlder_name + '''[Micrographs]" readonly="readonly"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Micrographs data path: </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_micrographs_data_path" value="''' + micrographs_data_path + '''"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Pixel size (A): </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_raw_pixel_size" value="''' + raw_pixel_size + '''"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Accelerating Voltage (kV): </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_accelerating_voltage" value="''' + accelerating_voltage + '''"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Spherical Aberration (mm): </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_spherical_aberration" value="''' + spherical_aberration + '''"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Total exposure dose (e/A^2): </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_total_exposure_dose" value="''' + total_exposure_dose + '''"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Particle diameter (A): </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_particle_diameter" value="''' + particle_diameter + '''"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Number of GPUs to parallelize: </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_number_of_GPUs_to_parallelize" value="''' + number_of_GPUs_to_parallelize + '''"></div>
                </div>
            </div>
        '''
    elif (source_input_type == 'Particles'):
        forlder_name = os.path.basename(os.path.dirname(input_parameters_folder))

        source_particle_jobuid = mytoolbox.readjson(input_parameters_folder + 'particle_job_uid.json')['source_particle_job_uid']
        source_particle_jobuid = source_particle_jobuid if source_particle_jobuid is not None else ''

        return '''
            <div class="single_inputparameter_div">
                <div class="single_item_div">
                    <div class="input_prompt_2"></div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter inputbox_foldername" type="text" name="''' + forlder_name + '''_folder_name" value="''' + forlder_name + '''[Particles]" readonly="readonly"></div>
                </div>
                <div class="single_item_div">
                    <div class="input_prompt_2">Source Particle Job uid: </div>
                    <div class="inputbox_div_inputparameter"><input class="inputbox_inputparameter" type="text" name="''' + forlder_name + '''_source_particle_jobuid" value="''' + source_particle_jobuid + '''"></div>
                </div>
            </div>
        '''
    else:
        print('Input type error!', flush=True)
        exit()


def excutecommand(command):
    command_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        line = ''
        single_character = command_process.stdout.read(1)
        while not ((single_character == '\r') or (single_character == '\n')):
            line += single_character
            single_character = command_process.stdout.read(1)
        line = line.rstrip()
        if (len(line) == 0) and (command_process.poll() is not None):
            break
        if (line == 'cryo_select_completed'):
            break
        if (len(line) > 0):
            line = line.replace('<', '&lt;')
            line = line.replace('>', '&gt;')
            yield 'data:' + line + '\n\n'
    yield 'data:my_command_completed\n\n'



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
            for i in range(5):
                try:
                    with open(target_dir + pid, 'r') as f:
                        slurminfo = f.readlines()
                    slurmjobuid = (slurminfo[0].split())[-1]
                    os.system('scancel ' + slurmjobuid)
                    os.remove(target_dir + pid)
                    break
                except:
                    pass
        else:
            if not psutil.pid_exists((int)(pid)):
                os.remove(target_dir + pid)
            # for i in range(5):
            while True:
                try:
                    mytoolbox.killprocesswithpid((int)(pid))
                    os.remove(target_dir + pid)
                    break
                except:
                    pass



filedir = os.path.dirname(os.path.dirname(__file__))
if not (filedir[-1] == '/'):
    filedir = filedir + '/'
app = flask.Flask(__name__)
cookie_life_time = 60 * 60 * 24 * 365



@app.route('/')
def index():
    response = flask.make_response(flask.render_template('index.html', CreateParameters={}, BaseParameters={}))
    response.set_cookie('globaldir', '', max_age=cookie_life_time)
    return response


@app.route('/CreateParameters', methods=['POST'])
def CreateParameters():
    globaldir = flask.request.form.get('globaldir')
    if not (globaldir[-1] == '/'):
        globaldir = globaldir + '/'

    if (flask.request.form['cp_button'] == 'Submit'):
        if not os.path.exists(globaldir):
            os.makedirs(globaldir)
        movie_job_num = (int)(flask.request.form.get('movie_job_num')) if (len(flask.request.form.get('movie_job_num')) > 0) else 0
        micrograph_job_num = (int)(flask.request.form.get('micrograph_job_num')) if (len(flask.request.form.get('micrograph_job_num')) > 0) else 0
        particle_job_num = (int)(flask.request.form.get('particle_job_num')) if (len(flask.request.form.get('particle_job_num')) > 0) else 0
        if os.path.exists(globaldir + 'parameters'):
            movie_job_num = 0
            micrograph_job_num = 0
            particle_job_num = 0
            ptr_count = 0
            parameters_files_list = os.listdir(globaldir + 'parameters/')
            while True:
                if ('import_parameters_' + (str)(ptr_count)) in parameters_files_list:
                    source_input_parameters_folder = globaldir + 'parameters/' + 'import_parameters_' + (str)(ptr_count) + '/'
                    source_input_type = mytoolbox.readjson(source_input_parameters_folder + 'input_type.json')
                    if (source_input_type == 'Movies'):
                        movie_job_num += 1
                    elif (source_input_type == 'Micrographs'):
                        micrograph_job_num += 1
                    elif (source_input_type == 'Particles'):
                        particle_job_num += 1
                    ptr_count += 1
                else:
                    break
        else:
            os.system('cd ' + globaldir + '&&python ' + filedir + 'CreateParameterFiles.py')
            if movie_job_num is not None:
                for _ in range(movie_job_num):
                    os.system('cd ' + globaldir + '&&python ' + filedir + 'CreateImportParameterFiles.py movie')
            if micrograph_job_num is not None:
                for _ in range(micrograph_job_num):
                    os.system('cd ' + globaldir + '&&python ' + filedir + 'CreateImportParameterFiles.py micrograph')
            if particle_job_num is not None:
                for _ in range(particle_job_num):
                    os.system('cd ' + globaldir + '&&python ' + filedir + 'CreateImportParameterFiles.py particle')

        global_parameters = mytoolbox.readjson(globaldir + 'parameters/parameters.json')
        final_nurefine_parameters = mytoolbox.readjson(globaldir + 'parameters/final_nurefine_parameters.json')

        base_parameters = global_parameters
        base_parameters['refine_symmetry'] = final_nurefine_parameters['Symmetry']
        base_parameters['movie_job_num'] = movie_job_num
        base_parameters['micrograph_job_num'] = micrograph_job_num
        base_parameters['particle_job_num'] = particle_job_num

        ptr_count = 0
        input_folders_str = ''
        parameters_files_list = os.listdir(globaldir + 'parameters/')
        while True:
            if ('import_parameters_' + (str)(ptr_count)) in parameters_files_list:
                source_input_parameters_folder = globaldir + 'parameters/' + 'import_parameters_' + (str)(ptr_count) + '/'
                input_folders_str += createinputdiv(source_input_parameters_folder)
                ptr_count += 1
            else:
                break

        response = flask.make_response(flask.render_template('index.html', CreateParameters={'globaldir': globaldir, 'input_folders': input_folders_str}, BaseParameters=base_parameters))
        response.set_cookie('globaldir', globaldir, max_age=cookie_life_time)
        return response

    elif (flask.request.form['cp_button'] == 'Clear'):
        if os.path.exists(globaldir + 'metadata'):
            shutil.rmtree(globaldir + 'metadata')
        if os.path.exists(globaldir + 'parameters'):
            shutil.rmtree(globaldir + 'parameters')

        return flask.make_response(flask.render_template('index.html', CreateParameters={}, BaseParameters={}))


@app.route('/ModifyBaseParameters', methods=['POST'])
def ModifyBaseParameters():
    globaldir = flask.request.cookies.get('globaldir')
    global_parameters = mytoolbox.readjson(globaldir + 'parameters/parameters.json')
    nurefine_parameters = mytoolbox.readjson(globaldir + 'parameters/nurefine_parameters.json')
    final_nurefine_parameters = mytoolbox.readjson(globaldir + 'parameters/final_nurefine_parameters.json')
    orientation_diagnostics_parameters = mytoolbox.readjson(globaldir + 'parameters/orientation_diagnostics_parameters.json')

    # global_parameters['linux_username'] = flask.request.form.get('linux_username') if (len(flask.request.form.get('linux_username')) > 0) else None
    # global_parameters['linux_password'] = flask.request.form.get('linux_password') if (len(flask.request.form.get('linux_password')) > 0) else None
    global_parameters['cryosparc_username'] = flask.request.form.get('cryosparc_username') if (len(flask.request.form.get('cryosparc_username')) > 0) else None
    global_parameters['cryosparc_password'] = flask.request.form.get('cryosparc_password') if (len(flask.request.form.get('cryosparc_password')) > 0) else None
    global_parameters['project'] = flask.request.form.get('project') if (len(flask.request.form.get('project')) > 0) else None
    global_parameters['workspace'] = flask.request.form.get('workspace') if (len(flask.request.form.get('workspace')) > 0) else None

    nurefine_parameters['Symmetry'] = flask.request.form.get('refine_symmetry') if (len(flask.request.form.get('refine_symmetry')) > 0) else 'C1'
    final_nurefine_parameters['Symmetry'] = flask.request.form.get('refine_symmetry') if (len(flask.request.form.get('refine_symmetry')) > 0) else 'C1'
    orientation_diagnostics_parameters['Symmetry'] = flask.request.form.get('refine_symmetry') if (len(flask.request.form.get('refine_symmetry')) > 0) else 'C1'

    if os.path.exists(globaldir + 'parameters/parameters.json'):
        os.remove(globaldir + 'parameters/parameters.json')
    mytoolbox.savetojson(global_parameters, globaldir + 'parameters/parameters.json', False)
    if os.path.exists(globaldir + 'parameters/nurefine_parameters.json'):
        os.remove(globaldir + 'parameters/nurefine_parameters.json')
    mytoolbox.savetojson(nurefine_parameters, globaldir + 'parameters/nurefine_parameters.json', False)
    if os.path.exists(globaldir + 'parameters/final_nurefine_parameters.json'):
        os.remove(globaldir + 'parameters/final_nurefine_parameters.json')
    mytoolbox.savetojson(final_nurefine_parameters, globaldir + 'parameters/final_nurefine_parameters.json', False)
    if os.path.exists(globaldir + 'parameters/orientation_diagnostics_parameters.json'):
        os.remove(globaldir + 'parameters/orientation_diagnostics_parameters.json')
    mytoolbox.savetojson(orientation_diagnostics_parameters, globaldir + 'parameters/orientation_diagnostics_parameters.json', False)

    movie_job_num = 0
    micrograph_job_num = 0
    particle_job_num = 0
    ptr_count = 0
    input_folders_str = ''
    parameters_files_list = os.listdir(globaldir + 'parameters/')
    while True:
        if ('import_parameters_' + (str)(ptr_count)) in parameters_files_list:
            source_input_parameters_folder = globaldir + 'parameters/' + 'import_parameters_' + (str)(ptr_count) + '/'
            source_input_type = mytoolbox.readjson(source_input_parameters_folder + 'input_type.json')
            if (source_input_type == 'Movies'):
                movie_job_num += 1

                forlder_name = os.path.basename(os.path.dirname(source_input_parameters_folder))
                import_movies_parameters = mytoolbox.readjson(source_input_parameters_folder + 'import_movies_parameters.json')
                motion_correction_parameters = mytoolbox.readjson(source_input_parameters_folder + 'patch_motion_correction_parameters.json')
                ctf_estimation_parameters = mytoolbox.readjson(source_input_parameters_folder + 'patch_ctf_estimation_parameters.json')
                blob_pick_parameters = mytoolbox.readjson(source_input_parameters_folder + 'blob_picker_parameters.json')
                extract_parameters = mytoolbox.readjson(source_input_parameters_folder + 'extract_micrographs_parameters.json')

                particle_pixel_size = (float)(flask.request.form.get(forlder_name + '_raw_pixel_size')) if (len(flask.request.form.get(forlder_name + '_raw_pixel_size')) > 0) else None
                particle_diameter = (int)(flask.request.form.get(forlder_name + '_particle_diameter')) if (len(flask.request.form.get(forlder_name + '_particle_diameter')) > 0) else None
                import_movies_parameters['Movies data path'] = flask.request.form.get(forlder_name + '_movies_data_path') if (len(flask.request.form.get(forlder_name + '_movies_data_path')) > 0) else None
                import_movies_parameters['Gain reference path'] = flask.request.form.get(forlder_name + '_gain_reference_path') if (len(flask.request.form.get(forlder_name + '_gain_reference_path')) > 0) else None
                import_movies_parameters['Raw pixel size (A)'] = particle_pixel_size
                import_movies_parameters['Accelerating Voltage (kV)'] = (float)(flask.request.form.get(forlder_name + '_accelerating_voltage')) if (len(flask.request.form.get(forlder_name + '_accelerating_voltage')) > 0) else None
                import_movies_parameters['Spherical Aberration (mm)'] = (float)(flask.request.form.get(forlder_name + '_spherical_aberration')) if (len(flask.request.form.get(forlder_name + '_spherical_aberration')) > 0) else None
                import_movies_parameters['Total exposure dose (e/A^2)'] = (float)(flask.request.form.get(forlder_name + '_total_exposure_dose')) if (len(flask.request.form.get(forlder_name + '_total_exposure_dose')) > 0) else None
                blob_pick_parameters['Minimum particle diameter (A)'] = (particle_diameter - 10) if ((particle_diameter - 10) >= 0) else 0
                blob_pick_parameters['Maximum particle diameter (A)'] = (particle_diameter + 10) if ((particle_diameter + 10) >= 0) else 0
                extract_parameters['Extraction box size (pix)'] = (int)(math.floor((float)(particle_diameter) / particle_pixel_size / 5.0) * 10.0) if ((particle_pixel_size is not None) and (particle_diameter is not None)) else None
                motion_correction_parameters['Number of GPUs to parallelize'] = (int)(flask.request.form.get(forlder_name + '_number_of_GPUs_to_parallelize')) if (len(flask.request.form.get(forlder_name + '_number_of_GPUs_to_parallelize')) > 0) else None
                ctf_estimation_parameters['Number of GPUs to parallelize'] = (int)(flask.request.form.get(forlder_name + '_number_of_GPUs_to_parallelize')) if (len(flask.request.form.get(forlder_name + '_number_of_GPUs_to_parallelize')) > 0) else None
                extract_parameters['Number of GPUs to parallelize (0 for CPU-only)'] = (int)(flask.request.form.get(forlder_name + '_number_of_GPUs_to_parallelize')) if (len(flask.request.form.get(forlder_name + '_number_of_GPUs_to_parallelize')) > 0) else None

                if os.path.exists(source_input_parameters_folder + 'import_movies_parameters.json'):
                    os.remove(source_input_parameters_folder + 'import_movies_parameters.json')
                mytoolbox.savetojson(import_movies_parameters, source_input_parameters_folder + 'import_movies_parameters.json', False)
                if os.path.exists(source_input_parameters_folder + 'patch_motion_correction_parameters.json'):
                    os.remove(source_input_parameters_folder + 'patch_motion_correction_parameters.json')
                mytoolbox.savetojson(motion_correction_parameters, source_input_parameters_folder + 'patch_motion_correction_parameters.json', False)
                if os.path.exists(source_input_parameters_folder + 'patch_ctf_estimation_parameters.json'):
                    os.remove(source_input_parameters_folder + 'patch_ctf_estimation_parameters.json')
                mytoolbox.savetojson(ctf_estimation_parameters, source_input_parameters_folder + 'patch_ctf_estimation_parameters.json', False)
                if os.path.exists(source_input_parameters_folder + 'blob_picker_parameters.json'):
                    os.remove(source_input_parameters_folder + 'blob_picker_parameters.json')
                mytoolbox.savetojson(blob_pick_parameters, source_input_parameters_folder + 'blob_picker_parameters.json', False)
                if os.path.exists(source_input_parameters_folder + 'extract_micrographs_parameters.json'):
                    os.remove(source_input_parameters_folder + 'extract_micrographs_parameters.json')
                mytoolbox.savetojson(extract_parameters, source_input_parameters_folder + 'extract_micrographs_parameters.json', False)
            elif (source_input_type == 'Micrographs'):
                micrograph_job_num += 1

                forlder_name = os.path.basename(os.path.dirname(source_input_parameters_folder))
                import_micrographs_parameters = mytoolbox.readjson(source_input_parameters_folder + 'import_micrographs_parameters.json')
                ctf_estimation_parameters = mytoolbox.readjson(source_input_parameters_folder + 'patch_ctf_estimation_parameters.json')
                blob_pick_parameters = mytoolbox.readjson(source_input_parameters_folder + 'blob_picker_parameters.json')
                extract_parameters = mytoolbox.readjson(source_input_parameters_folder + 'extract_micrographs_parameters.json')

                particle_pixel_size = (float)(flask.request.form.get(forlder_name + '_raw_pixel_size')) if (len(flask.request.form.get(forlder_name + '_raw_pixel_size')) > 0) else None
                particle_diameter = (int)(flask.request.form.get(forlder_name + '_particle_diameter')) if (len(flask.request.form.get(forlder_name + '_particle_diameter')) > 0) else None
                import_micrographs_parameters['Micrographs data path'] = flask.request.form.get(forlder_name + '_micrographs_data_path') if (len(flask.request.form.get(forlder_name + '_micrographs_data_path')) > 0) else None
                import_micrographs_parameters['Pixel size (A)'] = particle_pixel_size
                import_micrographs_parameters['Accelerating Voltage (kV)'] = (float)(flask.request.form.get(forlder_name + '_accelerating_voltage')) if (len(flask.request.form.get(forlder_name + '_accelerating_voltage')) > 0) else None
                import_micrographs_parameters['Spherical Aberration (mm)'] = (float)(flask.request.form.get(forlder_name + '_spherical_aberration')) if (len(flask.request.form.get(forlder_name + '_spherical_aberration')) > 0) else None
                import_micrographs_parameters['Total exposure dose (e/A^2)'] = (float)(flask.request.form.get(forlder_name + '_total_exposure_dose')) if (len(flask.request.form.get(forlder_name + '_total_exposure_dose')) > 0) else None
                blob_pick_parameters['Minimum particle diameter (A)'] = (particle_diameter - 10) if ((particle_diameter - 10) >= 0) else 0
                blob_pick_parameters['Maximum particle diameter (A)'] = (particle_diameter + 10) if ((particle_diameter + 10) >= 0) else 0
                extract_parameters['Extraction box size (pix)'] = (int)(math.floor((float)(particle_diameter) / particle_pixel_size / 5.0) * 10.0) if ((particle_pixel_size is not None) and (particle_diameter is not None)) else None
                ctf_estimation_parameters['Number of GPUs to parallelize'] = (int)(flask.request.form.get(forlder_name + '_number_of_GPUs_to_parallelize')) if (len(flask.request.form.get(forlder_name + '_number_of_GPUs_to_parallelize')) > 0) else None
                extract_parameters['Number of GPUs to parallelize (0 for CPU-only)'] = (int)(flask.request.form.get(forlder_name + '_number_of_GPUs_to_parallelize')) if (len(flask.request.form.get(forlder_name + '_number_of_GPUs_to_parallelize')) > 0) else None

                if os.path.exists(source_input_parameters_folder + 'import_micrographs_parameters.json'):
                    os.remove(source_input_parameters_folder + 'import_micrographs_parameters.json')
                mytoolbox.savetojson(import_micrographs_parameters, source_input_parameters_folder + 'import_micrographs_parameters.json', False)
                if os.path.exists(source_input_parameters_folder + 'patch_ctf_estimation_parameters.json'):
                    os.remove(source_input_parameters_folder + 'patch_ctf_estimation_parameters.json')
                mytoolbox.savetojson(ctf_estimation_parameters, source_input_parameters_folder + 'patch_ctf_estimation_parameters.json', False)
                if os.path.exists(source_input_parameters_folder + 'blob_picker_parameters.json'):
                    os.remove(source_input_parameters_folder + 'blob_picker_parameters.json')
                mytoolbox.savetojson(blob_pick_parameters, source_input_parameters_folder + 'blob_picker_parameters.json', False)
                if os.path.exists(source_input_parameters_folder + 'extract_micrographs_parameters.json'):
                    os.remove(source_input_parameters_folder + 'extract_micrographs_parameters.json')
                mytoolbox.savetojson(extract_parameters, source_input_parameters_folder + 'extract_micrographs_parameters.json', False)
            elif (source_input_type == 'Particles'):
                particle_job_num += 1

                forlder_name = os.path.basename(os.path.dirname(source_input_parameters_folder))
                source_particle_jobuid = flask.request.form.get(forlder_name + '_source_particle_jobuid') if (len(flask.request.form.get(forlder_name + '_source_particle_jobuid')) > 0) else None

                if os.path.exists(source_input_parameters_folder + 'particle_job_uid.json'):
                    os.remove(source_input_parameters_folder + 'particle_job_uid.json')
                mytoolbox.savetojson({'source_particle_job_uid': source_particle_jobuid}, source_input_parameters_folder + 'particle_job_uid.json', False)

            input_folders_str += createinputdiv(source_input_parameters_folder)
            ptr_count += 1
        else:
            break

    base_parameters = global_parameters
    base_parameters['refine_symmetry'] = final_nurefine_parameters['Symmetry']
    base_parameters['movie_job_num'] = movie_job_num
    base_parameters['micrograph_job_num'] = micrograph_job_num
    base_parameters['particle_job_num'] = particle_job_num

    return flask.make_response(flask.render_template('index.html', CreateParameters={'globaldir': globaldir, 'input_folders': input_folders_str}, BaseParameters=base_parameters))


@app.route('/ImportAndRefine', methods=['POST', 'GET'])
def ImportAndRefine():
    globaldir = flask.request.cookies.get('globaldir')
    # global_parameters = mytoolbox.readjson(globaldir + 'parameters/parameters.json')

    if not os.path.exists(globaldir + 'metadata/processpids'):
        os.makedirs(globaldir + 'metadata/processpids')
    killjobs(globaldir + 'metadata/processpids')

    # if (global_parameters['linux_username'] == 'root'):
    with open(globaldir + 'metadata/import_and_refine.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('pid=$$\n')
        f.write('echo $pid > ' + globaldir + 'metadata/processpids/$pid\n')
        f.write('cd ' + globaldir + '\n')
        f.write('python ' + filedir + 'ImportAndExtract.py\n')
        f.write('python ' + filedir + 'SelectAndRefine.py\n')
        f.write('rm -f ' + globaldir + 'metadata/processpids/$pid\n')
        f.write('echo "\ncryo_select_completed\n"')
    os.system('cd ' + globaldir + '&&nohup bash ' + globaldir + 'metadata/import_and_refine.sh > ' + globaldir + 'metadata/import_and_refine.log 2>&1 &')
    while True:
        if os.path.exists(globaldir + 'metadata/import_and_refine.log'):
            break
        time.sleep(1)
    with open(globaldir + 'metadata/tail_import_and_refine.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('pid=$$\n')
        f.write('echo $pid > ' + globaldir + 'metadata/processpids/$pid\n')
        f.write('tail -f ' + globaldir + 'metadata/import_and_refine.log\n')
    return flask.Response(flask.stream_with_context(excutecommand('bash ' + globaldir + 'metadata/tail_import_and_refine.sh')), mimetype='text/event-stream')


# @app.route('/Import', methods=['POST', 'GET'])
# def Import():
#     globaldir = flask.request.cookies.get('globaldir')
#     # global_parameters = mytoolbox.readjson(globaldir + 'parameters/parameters.json')
#
#     if not os.path.exists(globaldir + 'metadata/processpids'):
#         os.makedirs(globaldir + 'metadata/processpids')
#     killjobs(globaldir + 'metadata/processpids')
#
#     # if (global_parameters['linux_username'] == 'root'):
#     with open(globaldir + 'metadata/import_and_extract.sh', 'w') as f:
#         f.write('#!/bin/bash\n')
#         f.write('pid=$$\n')
#         f.write('echo $pid > ' + globaldir + 'metadata/processpids/$pid\n')
#         f.write('cd ' + globaldir + '\n')
#         f.write('python ' + filedir + 'ImportAndExtract.py\n')
#         f.write('rm -f ' + globaldir + 'metadata/processpids/$pid\n')
#         f.write('echo "\ncryo_select_completed\n"')
#     os.system('cd ' + globaldir + '&&nohup bash ' + globaldir + 'metadata/import_and_extract.sh > ' + globaldir + 'metadata/import_and_refine.log 2>&1 &')
#     while True:
#         if os.path.exists(globaldir + 'metadata/import_and_refine.log'):
#             break
#         time.sleep(1)
#     with open(globaldir + 'metadata/tail_import_and_refine.sh', 'w') as f:
#         f.write('#!/bin/bash\n')
#         f.write('pid=$$\n')
#         f.write('echo $pid > ' + globaldir + 'metadata/processpids/$pid\n')
#         f.write('tail -f ' + globaldir + 'metadata/import_and_refine.log\n')
#     return flask.Response(flask.stream_with_context(excutecommand('bash ' + globaldir + 'metadata/tail_import_and_refine.sh')), mimetype='text/event-stream')
#
#
# @app.route('/Refine', methods=['POST', 'GET'])
# def Refine():
#     globaldir = flask.request.cookies.get('globaldir')
#     # global_parameters = mytoolbox.readjson(globaldir + 'parameters/parameters.json')
#
#     if not os.path.exists(globaldir + 'metadata/processpids'):
#         os.makedirs(globaldir + 'metadata/processpids')
#     killjobs(globaldir + 'metadata/processpids')
#
#     # if (global_parameters['linux_username'] == 'root'):
#     with open(globaldir + 'metadata/select_and_refine.sh', 'w') as f:
#         f.write('#!/bin/bash\n')
#         f.write('pid=$$\n')
#         f.write('echo $pid > ' + globaldir + 'metadata/processpids/$pid\n')
#         f.write('cd ' + globaldir + '\n')
#         f.write('python ' + filedir + 'SelectAndRefine.py\n')
#         f.write('rm -f ' + globaldir + 'metadata/processpids/$pid\n')
#         f.write('echo "\ncryo_select_completed\n"')
#     os.system('cd ' + globaldir + '&&nohup bash ' + globaldir + 'metadata/select_and_refine.sh > ' + globaldir + 'metadata/import_and_refine.log 2>&1 &')
#     while True:
#         if os.path.exists(globaldir + 'metadata/import_and_refine.log'):
#             break
#         time.sleep(1)
#     with open(globaldir + 'metadata/tail_import_and_refine.sh', 'w') as f:
#         f.write('#!/bin/bash\n')
#         f.write('pid=$$\n')
#         f.write('echo $pid > ' + globaldir + 'metadata/processpids/$pid\n')
#         f.write('tail -f ' + globaldir + 'metadata/import_and_refine.log\n')
#     return flask.Response(flask.stream_with_context(excutecommand('bash ' + globaldir + 'metadata/tail_import_and_refine.sh')), mimetype='text/event-stream')


@app.route('/KillJob', methods=['POST', 'GET'])
def KillJob():
    globaldir = flask.request.cookies.get('globaldir')
    if not os.path.exists(globaldir + 'metadata/processpids'):
        os.makedirs(globaldir + 'metadata/processpids')
    killjobs(globaldir + 'metadata/processpids')
    return flask.Response(flask.stream_with_context(excutecommand('echo \"[Killed]\"')), mimetype='text/event-stream')


@app.route('/Reconnect', methods=['POST', 'GET'])
def Reconnect():
    globaldir = flask.request.cookies.get('globaldir')
    if os.path.exists(globaldir + 'metadata/tail_import_and_refine.sh'):
        return flask.Response(flask.stream_with_context(excutecommand('bash ' + globaldir + 'metadata/tail_import_and_refine.sh')), mimetype='text/event-stream')
    else:
        return flask.Response(flask.stream_with_context(excutecommand('echo \"' + globaldir + 'metadata/tail_import_and_refine.sh do not exist...\"')), mimetype='text/event-stream')


@app.route('/DownloadMap', methods=['POST', 'GET'])
def DownloadMap():
    globaldir = flask.request.cookies.get('globaldir')
    global_parameters = mytoolbox.readjson(globaldir + 'parameters/parameters.json')
    cshandle = CSLogin.cshandleclass.GetCryoSPARCHandle(email=global_parameters['cryosparc_username'], password=global_parameters['cryosparc_password'])
    dealjobs = MyJobAPIs.DealJobs(cshandle, global_parameters['project'], global_parameters['workspace'], global_parameters['lane'])
    try:
        if os.path.exists(globaldir + 'metadata/curve_job_parents.json'):
            jobuid = mytoolbox.readjson(globaldir + 'metadata/curve_job_parents.json')['best_final_nurefine_job_uid']
            jobdir = (str)(cshandle.find_job(dealjobs.project, jobuid).dir())
            if not (jobdir[-1] == '/'):
                jobdir = jobdir + '/'
            ptr = 0
            while True:
                if (ptr > 999):
                    return 'Map download failed...', 500
                if os.path.exists(jobdir + jobuid + '_' + ((str)(ptr)).zfill(3) + '_volume_map_sharp.mrc'):
                    ptr += 1
                else:
                    break
            target_file_path = jobdir + jobuid + '_' + ((str)(ptr - 1)).zfill(3) + '_volume_map_sharp.mrc'
            return flask.send_file(target_file_path, as_attachment=True)
        else:
            return 'Job did not finish yet, please wait...', 404
    except:
        return 'Error...', 500


if (__name__ == '__main__'):
    app.run(host='0.0.0.0', port=38080)