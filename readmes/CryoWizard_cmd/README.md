
## Using CryoWizard via Command


### Single-Command Pipeline Execution

The complete CryoWizard pipeline can be executed via a single command:

    (cryo_ief) $ python CryoWizard_main.py \
        --path_result_dir 'path/to/save/your/cryowizard/metadata/folder' \
        --CreateParameterFiles \
        --CreateImportParameterFiles \
        --ImportAndExtract \
        --SelectAndRefine \
        --input_type movie \
        --cryosparc_username 'your_cryosparc_username@xxx.xxx' \
        --cryosparc_password 'your_cryosparc_password' \
        --cryosparc_project P1 \
        --cryosparc_workspace W1 \
        --symmetry D2 \
        --movies_data_path 'path/to/movies_files' \
        --gain_reference_path 'path/to/gain_reference_files' \        
        --raw_pixel_size 0.885 \
        --accelerating_voltage 200 \
        --spherical_aberration 1.4 \
        --total_exposure_dose 30.6 \
        --particle_diameter 160 \
        --gpu_num 1

This command will import movie data and execute the entire pipeline, which will be visible within your cryoSPARC interface.

Also, micrographs can be specified as input:

    (cryo_ief) $ python CryoWizard_main.py \
        --path_result_dir 'path/to/save/your/cryowizard/metadata/folder' \
        --CreateParameterFiles \
        --CreateImportParameterFiles \
        --ImportAndExtract \
        --SelectAndRefine \
        --input_type micrograph \
        --cryosparc_username 'your_cryosparc_username@xxx.xxx' \
        --cryosparc_password 'your_cryosparc_password' \
        --cryosparc_project P1 \
        --cryosparc_workspace W1 \
        --symmetry D2 \
        --micrographs_data_path 'path/to/micrographs_files' \
        --raw_pixel_size 0.885 \
        --accelerating_voltage 200 \
        --spherical_aberration 1.4 \
        --total_exposure_dose 30.6 \
        --particle_diameter 160 \
        --gpu_num 1

or a cryoSPARC particle job (e.g., `Import Particle Stack`, `Extract From Micrographs(Multi-GPU)`, `Restack Particles`) as input:

    (cryo_ief) $ python CryoWizard_main.py \
        --path_result_dir 'path/to/save/your/cryowizard/metadata/folder' \
        --CreateParameterFiles \
        --CreateImportParameterFiles \
        --ImportAndExtract \
        --SelectAndRefine \
        --input_type particle \
        --cryosparc_username 'your_cryosparc_username@xxx.xxx' \
        --cryosparc_password 'your_cryosparc_password' \
        --cryosparc_project P1 \
        --cryosparc_workspace W1 \
        --symmetry D2 \
        --particle_job_uid J1

Note that the single-command pipeline execution supports only one type of input data. For multiple input types, utilize the CryoWizard web interface or execute the pipeline step-by-step.

### Step-by-Step Pipeline Execution

To create a pipeline project folder and generate parameter files, execute:

    (cryo_ief) $ python CryoWizard_main.py \
        --path_result_dir 'path/to/save/your/cryowizard/metadata/folder' \
        --CreateParameterFiles \
        --cryosparc_username 'your_cryosparc_username@xxx.xxx' \
        --cryosparc_password 'your_cryosparc_password' \
        --cryosparc_project P1 \
        --cryosparc_workspace W1 \
        --symmetry D2

Upon execution of these commands, a `parameter` folder will be created, and parameter files will be copied to this folder from `path/to/Cryo-IEF/CryoWizard/parameters`.

To generate import job folders, consider the following example: if raw data for a single target protein is distributed across multiple sources, it may necessitate the creation of two import movie jobs, one import micrograph job, and one import particle stack job to consolidate all raw data:

    ...
    (cryo_ief) $ python CryoWizard_main.py \
        --path_result_dir 'path/to/save/your/cryowizard/metadata/folder' \
        --CreateImportParameterFiles \
        --input_type movie \
        --movies_data_path 'path/to/movies_files' \
        --gain_reference_path 'path/to/gain_reference_files' \        
        --raw_pixel_size 0.885 \
        --accelerating_voltage 200 \
        --spherical_aberration 1.4 \
        --total_exposure_dose 30.6 \
        --particle_diameter 160 \
        --gpu_num 1
    (cryo_ief) $ python CryoWizard_main.py \
        --path_result_dir 'path/to/save/your/cryowizard/metadata/folder' \
        --CreateImportParameterFiles \
        --input_type movie \
        --movies_data_path 'path/to/movies2_files' \
        --gain_reference_path 'path/to/gain_reference2_files' \        
        --raw_pixel_size 0.885 \
        --accelerating_voltage 200 \
        --spherical_aberration 1.4 \
        --total_exposure_dose 30.6 \
        --particle_diameter 160 \
        --gpu_num 1
    (cryo_ief) $ python CryoWizard_main.py \
        --path_result_dir 'path/to/save/your/cryowizard/metadata/folder' \
        --CreateImportParameterFiles \
        --input_type micrograph \
        --micrographs_data_path 'path/to/micrographs_files' \
        --raw_pixel_size 0.885 \
        --accelerating_voltage 200 \
        --spherical_aberration 1.4 \
        --total_exposure_dose 30.6 \
        --particle_diameter 160 \
        --gpu_num 1
    (cryo_ief) $ python CryoWizard_main.py \
        --path_result_dir 'path/to/save/your/cryowizard/metadata/folder' \
        --CreateImportParameterFiles \
        --input_type particle \
        --particle_job_uid J1

Upon execution of these commands, four folders will be generated within the `parameter` directory: `import_parameters_0`, `import_parameters_1`, `import_parameters_2`, and `import_parameters_3`. 
These correspond to the movie, movie, micrograph, and particle jobs, respectively. Each `import_parameters_` folder will contain complete import parameters identical to those used by cryoSPARC import jobs.

All parameter files in the `parameter` folder and `import_parameters_` folders can be manually opened and modified.

Once parameters have been adjusted, proceed with data import and preprocessing:

    ...
    (cryo_ief) $ python CryoWizard_main.py \
        --path_result_dir 'path/to/save/your/cryowizard/metadata/folder' \
        --ImportAndExtract

Execution of this command will initiate the creation and execution of import, motion correction, CTF estimation, blob picker, and extraction jobs within cryoSPARC. All particles will subsequently be consolidated into a single restack job within cryoSPARC.

Next, execute the following command:

    ...
    (cryo_ief) $ python CryoWizard_main.py \
        --path_result_dir 'path/to/save/your/cryowizard/metadata/folder' \
        --SelectAndRefine

This program employs our model to score and automatically select particles, which are then utilized for map volume reconstruction and refinement. Upon completion, results can be reviewed within cryoSPARC.

Additionally, a command is provided to retrieve the top `N` particles (e.g. 50000), sorted by scores generated by the model after executing `SelectAndRefine`:

    ...
    (cryo_ief) $ python CryoWizard_main.py \
        --TruncateParticles \
        --truncation_type num \
        --particle_cutoff_condition N

This command will generate an external job containing the output with the top `N` particles. 

Alternatively, by providing a score `S` (e.g. 0.9), particles with scores greater than the specified threshold can be retrieved:

    ...
    (cryo_ief) $ python CryoWizard_main.py \
        --TruncateParticles \
        --truncation_type score \
        --particle_cutoff_condition S