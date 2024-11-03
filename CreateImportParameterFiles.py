#!/usr/bin/env python

import shutil
import sys
import os

import MyLib.mytoolbox as mytoolbox



globaldir = os.getcwd()
if not (globaldir[-1] == '/'):
    globaldir = globaldir + '/'

source_python_file_path = os.path.dirname(__file__)
if not (source_python_file_path[-1] == '/'):
    source_python_file_path = source_python_file_path + '/'

i = 0
while True:
    if not os.path.exists(globaldir + 'parameters/import_parameters_' + (str)(i)):
        os.makedirs(globaldir + 'parameters/import_parameters_' + (str)(i))
        break
    else:
        i += 1
try:
    if (sys.argv[1] == 'movie'):
        for item in ['import_movies_parameters.json', 'patch_motion_correction_parameters.json', 'patch_ctf_estimation_parameters.json', 'blob_picker_parameters.json', 'extract_micrographs_parameters.json']:
            shutil.copy(source_python_file_path + 'parameters/' + item, globaldir + 'parameters/import_parameters_'  + (str)(i) + '/' + item)
        mytoolbox.savetojson('Movies', globaldir + 'parameters/import_parameters_'  + (str)(i) + '/input_type.json', False)
    elif (sys.argv[1] == 'micrograph'):
        for item in ['import_micrographs_parameters.json', 'patch_ctf_estimation_parameters.json', 'blob_picker_parameters.json', 'extract_micrographs_parameters.json']:
            shutil.copy(source_python_file_path + 'parameters/' + item, globaldir + 'parameters/import_parameters_'  + (str)(i) + '/' + item)
        mytoolbox.savetojson('Micrographs', globaldir + 'parameters/import_parameters_'  + (str)(i) + '/input_type.json', False)
    elif (sys.argv[1] == 'particle'):
        mytoolbox.savetojson('Particles', globaldir + 'parameters/import_parameters_'  + (str)(i) + '/input_type.json', False)
        mytoolbox.savetojson({'source_particle_job_uid': 'J'}, globaldir + 'parameters/import_parameters_'  + (str)(i) + '/particle_job_uid.json', False)
    else:
        shutil.rmtree(globaldir + 'parameters/import_parameters_' + (str)(i))
        print('Input type error...', flush=True)
        exit()
except:
    shutil.rmtree(globaldir + 'parameters/import_parameters_' + (str)(i))
    print('Error...', flush=True)
    exit()