#!/usr/bin/env python

import shutil
import os



globaldir = os.getcwd()
if not (globaldir[-1] == '/'):
    globaldir = globaldir + '/'
if not os.path.exists(globaldir + 'parameters/'):
    os.makedirs(globaldir + 'parameters/')
    source_python_file_path = os.path.dirname(__file__)
    if not (source_python_file_path[-1] == '/'):
        source_python_file_path = source_python_file_path + '/'
    for item in ['parameters.json', 'abinit_parameters.json', 'nurefine_parameters.json', 'final_nurefine_parameters.json', 'restack_particles_parameters.json', 'orientation_diagnostics_parameters.json', 'GetConfidence.sh']:
        shutil.copy(source_python_file_path + 'parameters/' + item, globaldir + 'parameters/' + item)

