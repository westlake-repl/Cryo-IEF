import flask
import flask_socketio
import shutil
import psutil
import math
import time
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
socketio = flask_socketio.SocketIO(app, cors_allowed_origins='*')

multithread = mytoolbox.MultiThreadingRun(threadpoolmaxsize=128, try_mode=True)



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


def queue_cryowizard_external_job_func_run(cryosparc_username, cryosparc_password, cryowizard_job, cryowizard_cache_path, project, workspace, parameters):

    # get input jobs info
    cryowizard_job_inputs_info = {}
    for i in range(len(cryowizard_job.doc['input_slot_groups'])):
        input_slot_title = cryowizard_job.doc['input_slot_groups'][i]['title']
        input_slot_connections = cryowizard_job.doc['input_slot_groups'][i]['connections']
        cryowizard_job_inputs_info_jobs_list = []
        for j in range(len(input_slot_connections)):
            input_job_id = input_slot_connections[j]['job_uid']
            input_job_group_name = input_slot_connections[j]['group_name']
            cryowizard_job_inputs_info_jobs_list.append({'job_uid': input_job_id, 'group_name': input_job_group_name})
        cryowizard_job_inputs_info[input_slot_title] = cryowizard_job_inputs_info_jobs_list

    # run cryowizard
    with cryowizard_job.run():
        try:
            # get parameters
            symmetry = (str)(parameters['symmetry']) if (len(parameters['symmetry']) > 0) else None
            particle_diameter = (int)(parameters['diameter']) if (len(parameters['diameter']) > 0) else None
            particle_pixel_size = (float)(parameters['pixelsize']) if (len(parameters['pixelsize']) > 0) else None
            gpu_num = (int)(parameters['gpu_num']) if (len(parameters['gpu_num']) > 0) else None

            if ((symmetry is None) or (particle_diameter is None) or (particle_pixel_size is None) or (gpu_num is None)):
                cryowizard_job.log("Wrong Parameters!", level="error")


            # create base parameters
            cryowizard_job.subprocess(('python ' + filedir + '/CryoWizard_main.py --CreateParameterFiles --path_result_dir ' + cryowizard_cache_path).split(' '), mute=True)

            global_parameters = mytoolbox.readjson(os.path.join(cryowizard_cache_path, 'parameters', 'parameters.json'))
            final_nurefine_parameters = mytoolbox.readjson(os.path.join(cryowizard_cache_path, 'parameters', 'final_nurefine_parameters.json'))
            orientation_diagnostics_parameters = mytoolbox.readjson(os.path.join(cryowizard_cache_path, 'parameters', 'orientation_diagnostics_parameters.json'))

            global_parameters['cryosparc_username'] = cryosparc_username
            global_parameters['cryosparc_password'] = cryosparc_password
            global_parameters['project'] = project
            global_parameters['workspace'] = workspace
            mytoolbox.savetojson(global_parameters, os.path.join(cryowizard_cache_path, 'parameters', 'parameters.json'), False)

            final_nurefine_parameters['Symmetry'] = symmetry
            mytoolbox.savetojson(final_nurefine_parameters, os.path.join(cryowizard_cache_path, 'parameters', 'final_nurefine_parameters.json'), False)
            orientation_diagnostics_parameters['Symmetry'] = symmetry
            mytoolbox.savetojson(orientation_diagnostics_parameters, os.path.join(cryowizard_cache_path, 'parameters', 'orientation_diagnostics_parameters.json'), False)


            # create import parameters
            last_parameter_items = os.listdir(os.path.join(cryowizard_cache_path, 'parameters'))
            for i in range(len(cryowizard_job_inputs_info['Input Movie'])):
                job_input_info = cryowizard_job_inputs_info['Input Movie'][i]
                cryowizard_job.subprocess(('python ' + filedir + '/CryoWizard_main.py --CreateImportParameterFiles --input_type extension_movie --path_result_dir ' + cryowizard_cache_path).split(' '), mute=True)
                new_parameter_items = os.listdir(os.path.join(cryowizard_cache_path, 'parameters'))
                new_import_parameters_folder = None
                for new_item in new_parameter_items:
                    if new_item not in last_parameter_items:
                        new_import_parameters_folder = new_item
                        break
                last_parameter_items = new_parameter_items
                mytoolbox.savetojson({'source_job_uid': job_input_info['job_uid'], 'source_group_name': job_input_info['group_name']}, os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'job_uid.json'), False)

                motion_correction_parameters = mytoolbox.readjson(os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'patch_motion_correction_parameters.json'))
                ctf_estimation_parameters = mytoolbox.readjson(os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'patch_ctf_estimation_parameters.json'))
                blob_pick_parameters = mytoolbox.readjson(os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'blob_picker_parameters.json'))
                extract_parameters = mytoolbox.readjson(os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'extract_micrographs_parameters.json'))

                blob_pick_parameters['Minimum particle diameter (A)'] = (particle_diameter - 10) if ((particle_diameter - 10) >= 0) else 0
                blob_pick_parameters['Maximum particle diameter (A)'] = (particle_diameter + 10) if ((particle_diameter + 10) >= 0) else 0
                if ((1.0 - particle_pixel_size) > math.fabs(1.0 - 2.0 * particle_pixel_size)):
                    motion_correction_parameters['Output F-crop factor'] = '1/2'
                    extract_parameters['Extraction box size (pix)'] = (int)(math.floor((float)(particle_diameter) / (particle_pixel_size * 2.0) / 5.0) * 10.0)
                else:
                    extract_parameters['Extraction box size (pix)'] = (int)(math.floor((float)(particle_diameter) / particle_pixel_size / 5.0) * 10.0)
                if gpu_num is not None:
                    motion_correction_parameters['Number of GPUs to parallelize'] = gpu_num
                    ctf_estimation_parameters['Number of GPUs to parallelize'] = gpu_num
                    extract_parameters['Number of GPUs to parallelize (0 for CPU-only)'] = gpu_num

                mytoolbox.savetojson(motion_correction_parameters, os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'patch_motion_correction_parameters.json'), False)
                mytoolbox.savetojson(ctf_estimation_parameters, os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'patch_ctf_estimation_parameters.json'), False)
                mytoolbox.savetojson(blob_pick_parameters, os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'blob_picker_parameters.json'), False)
                mytoolbox.savetojson(extract_parameters, os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'extract_micrographs_parameters.json'), False)

            for i in range(len(cryowizard_job_inputs_info['Input Micrograph'])):
                job_input_info = cryowizard_job_inputs_info['Input Micrograph'][i]
                cryowizard_job.subprocess(('python ' + filedir + '/CryoWizard_main.py --CreateImportParameterFiles --input_type extension_micrograph --path_result_dir ' + cryowizard_cache_path).split(' '), mute=True)
                new_parameter_items = os.listdir(os.path.join(cryowizard_cache_path, 'parameters'))
                new_import_parameters_folder = None
                for new_item in new_parameter_items:
                    if new_item not in last_parameter_items:
                        new_import_parameters_folder = new_item
                        break
                last_parameter_items = new_parameter_items
                mytoolbox.savetojson({'source_job_uid': job_input_info['job_uid'], 'source_group_name': job_input_info['group_name']}, os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'job_uid.json'), False)

                ctf_estimation_parameters = mytoolbox.readjson(os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'patch_ctf_estimation_parameters.json'))
                blob_pick_parameters = mytoolbox.readjson(os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'blob_picker_parameters.json'))
                extract_parameters = mytoolbox.readjson(os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'extract_micrographs_parameters.json'))

                blob_pick_parameters['Minimum particle diameter (A)'] = (particle_diameter - 10) if ((particle_diameter - 10) >= 0) else 0
                blob_pick_parameters['Maximum particle diameter (A)'] = (particle_diameter + 10) if ((particle_diameter + 10) >= 0) else 0
                extract_parameters['Extraction box size (pix)'] = (int)(math.floor((float)(particle_diameter) / particle_pixel_size / 5.0) * 10.0)
                if gpu_num is not None:
                    ctf_estimation_parameters['Number of GPUs to parallelize'] = gpu_num
                    extract_parameters['Number of GPUs to parallelize (0 for CPU-only)'] = gpu_num

                mytoolbox.savetojson(ctf_estimation_parameters, os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'patch_ctf_estimation_parameters.json'), False)
                mytoolbox.savetojson(blob_pick_parameters, os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'blob_picker_parameters.json'), False)
                mytoolbox.savetojson(extract_parameters, os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'extract_micrographs_parameters.json'), False)

            for i in range(len(cryowizard_job_inputs_info['Input Particle'])):
                job_input_info = cryowizard_job_inputs_info['Input Particle'][i]
                cryowizard_job.subprocess(('python ' + filedir + '/CryoWizard_main.py --CreateImportParameterFiles --input_type extension_particle --path_result_dir ' + cryowizard_cache_path).split(' '), mute=True)
                new_parameter_items = os.listdir(os.path.join(cryowizard_cache_path, 'parameters'))
                new_import_parameters_folder = None
                for new_item in new_parameter_items:
                    if new_item not in last_parameter_items:
                        new_import_parameters_folder = new_item
                        break
                last_parameter_items = new_parameter_items
                mytoolbox.savetojson({'source_job_uid': job_input_info['job_uid'], 'source_group_name': job_input_info['group_name']}, os.path.join(cryowizard_cache_path, 'parameters', new_import_parameters_folder, 'job_uid.json'), False)


            # run
            if not os.path.exists(os.path.join(cryowizard_cache_path, 'metadata')):
                os.makedirs(os.path.join(cryowizard_cache_path, 'metadata'))
            if not os.path.exists(os.path.join(cryowizard_cache_path, 'metadata', 'import_and_refine.lock')):
                with open(os.path.join(cryowizard_cache_path, 'metadata', 'import_and_refine.lock'), 'w') as f:
                    f.write('lock')
                if not os.path.exists(os.path.join(cryowizard_cache_path, 'metadata', 'processpids')):
                    os.makedirs(os.path.join(cryowizard_cache_path, 'metadata', 'processpids'))
                with open(os.path.join(cryowizard_cache_path, 'metadata', 'import_and_refine.sh'), 'w') as f:
                    f.write('#!/bin/bash\n')
                    f.write('pid=$$\n')
                    f.write('echo $pid > ' + cryowizard_cache_path + '/metadata/processpids/$pid\n')
                    f.write('cd ' + cryowizard_cache_path + '\n')
                    f.write('python ' + filedir + '/CryoWizard_main.py --ImportAndExtract\n')
                    f.write('python ' + filedir + '/CryoWizard_main.py --SelectAndRefine\n')
                    f.write('rm -f ' + cryowizard_cache_path + '/metadata/processpids/$pid\n')
                    f.write('rm -f ' + cryowizard_cache_path + '/metadata/import_and_refine.lock\n')
                cryowizard_job.subprocess(('bash ' + cryowizard_cache_path + '/metadata/import_and_refine.sh').split(' '), mute=True)
        except:
            pass

    if not (os.path.exists(os.path.join(cryowizard_cache_path, 'metadata', 'curve_job_parents.json'))):
        cryowizard_job.stop(error=True)


def queue_cryowizard_external_job_func_check_status(cryowizard_job, cryowizard_cache_path):
    # kill all subprocess if not success
    cryowizard_job.wait_for_done(error_on_incomplete=False)
    if ((cryowizard_job.status == 'killed') or (cryowizard_job.status == 'failed')):
        for i in range(1000):
            kill_result = killjobs(os.path.join(cryowizard_cache_path, 'metadata', 'processpids'))
            if kill_result:
                if os.path.exists(os.path.join(cryowizard_cache_path, 'metadata', 'import_and_refine.lock')):
                    os.remove(os.path.join(cryowizard_cache_path, 'metadata', 'import_and_refine.lock'))
                break
            time.sleep(0.01)
    print('cryowizard process done!', flush=True)





@socketio.on('check_cryowizard_user_login_action')
def check_cryowizard_user_login_action(cryosparc_username, cryosparc_password):
    try:
        cshandle = CSLogin.cshandleclass.GetCryoSPARCHandle(email=cryosparc_username, password=cryosparc_password)
        flask_socketio.emit('js_check_cryowizard_user_login_action', {'result': True})
    except:
        flask_socketio.emit('js_check_cryowizard_user_login_action', {'result': False})


@socketio.on('check_cryowizard_external_job_parameters_action')
def check_cryowizard_external_job_parameters_action(cryosparc_username, cryosparc_password, project, jobid):
    cshandle = CSLogin.cshandleclass.GetCryoSPARCHandle(email=cryosparc_username, password=cryosparc_password)
    if os.path.exists(os.path.join(os.path.normpath((str)(cshandle.find_project(project).dir())), jobid, 'cryowizard_ui_parameters.json')):
        parameters = mytoolbox.readjson(os.path.join(os.path.normpath((str)(cshandle.find_project(project).dir())), jobid, 'cryowizard_ui_parameters.json'))
        flask_socketio.emit('js_check_cryowizard_external_job_parameters_action', {'project': project, 'jobid': jobid, 'parameters': parameters})


@socketio.on('create_cryowizard_external_job_action')
def create_cryowizard_external_job_action(cryosparc_username, cryosparc_password, project, workspace, default_parameters):

    def CreateExternal(dealjobinstance, title='CryoWizard'):

        # 创建空白 external job
        project = dealjobinstance.cshandle.find_project(dealjobinstance.project)
        new_job = project.create_external_job(dealjobinstance.workspace, title=title)

        # 创建和链接输入
        add_input_movie_name = 'input_movies'
        add_input_micrograph_name = 'input_micrographs'
        add_input_particle_name = 'input_particles'
        new_job.add_input(type='exposure', name=add_input_movie_name, min=0, title='Input Movie')
        new_job.add_input(type='exposure', name=add_input_micrograph_name, min=0, title='Input Micrograph')
        new_job.add_input(type='particle', name=add_input_particle_name, min=0, title='Input Particle')

        return new_job

    cshandle = CSLogin.cshandleclass.GetCryoSPARCHandle(email=cryosparc_username, password=cryosparc_password)
    dealjobs = MyJobAPIs.DealJobs(cshandle, project, workspace, 'default')

    new_external_job = CreateExternal(dealjobs)

    mytoolbox.savetojson(default_parameters, os.path.join(os.path.normpath((str)(cshandle.find_project(project).dir())), new_external_job.uid, 'cryowizard_ui_parameters.json'))

    flask_socketio.emit('js_create_cryowizard_external_job_action', {'project': project, 'workspace': workspace, 'new_external_jobid': new_external_job.uid})


@socketio.on('queue_cryowizard_external_job_action')
def queue_cryowizard_external_job_action(cryosparc_username, cryosparc_password, project, jobid, parameters):

    cshandle = CSLogin.cshandleclass.GetCryoSPARCHandle(email=cryosparc_username, password=cryosparc_password)
    cryowizard_job = cshandle.find_external_job(project, jobid)
    workspace = cryowizard_job.doc['workspace_uids'][0]

    cryowizard_cache_path = os.path.normpath(os.path.join(os.path.normpath((str)(cshandle.find_project(project).dir())), jobid, 'cryowizard'))
    if os.path.exists(cryowizard_cache_path):
        shutil.rmtree(cryowizard_cache_path)
    os.makedirs(cryowizard_cache_path)

    mytoolbox.savetojson(parameters, os.path.join(os.path.normpath((str)(cshandle.find_project(project).dir())), cryowizard_job.uid, 'cryowizard_ui_parameters.json'))

    multithread.setthread(queue_cryowizard_external_job_func_run,
                          cryosparc_username=cryosparc_username,
                          cryosparc_password=cryosparc_password,
                          cryowizard_job=cryowizard_job,
                          cryowizard_cache_path=cryowizard_cache_path,
                          project=project,
                          workspace=workspace,
                          parameters=parameters)

    multithread.setthread(queue_cryowizard_external_job_func_check_status,
                          cryowizard_job=cryowizard_job,
                          cryowizard_cache_path=cryowizard_cache_path)


    # flask_socketio.emit('js_queue_cryowizard_external_job_action', {'project': project, 'workspace': workspace, 'new_external_jobid': new_external_job.uid})







if (__name__ == '__main__'):
    print('Web service start, press Ctrl+C to quit if you want to stop this web service.', flush=True)
    socketio.run(app, host='0.0.0.0', port=WEB_PORT)