
## (Optional) More Settings in Parameters.json


### cryowizard_settings.yml

After installation, there are some CryoWizard global setting in `path/to/Cryo-IEF/CryoWizard/cryowizard_settings.yml`, and you can modify these parameters according to your needs before using.

For example, the default setting of CryoSPARC lane is `default`, which is used to specify where the CryoSPARC jobs will be queued on during CryoWizard running. You can set it to your target lane.
    
    ...
    CreateParameterFiles_settings:
       cryosparc_project: null
       cryosparc_workspace: null
       cryosparc_lane: YOUR_TARGET_LANE
       if_slurm: false
       inference_gpu_ids: null
    ...

### parameters.json

The `parameters.json` file in your workflow metadata path contains all adjustable pipeline parameters. You may customize these to suit your project needs after creating base parameters.

The following is an example of the `parameters.json` file:

    {
    
    # cryosparc_username and cryosparc_password are uesd to log in CryoSPARC
    "cryosparc_username": null,
    "cryosparc_password": null,
    
    # project and workspace: where to create jobs in CryoSPARC
    # lane: CryoSPARC lane which will be used by CryoSPARC
    # hostname_gpus_jobnum_lists: max number of jobs running on each gpu, each item in this list is a list too, which is [node, gpu number, max job on each gpu]. e.g. "hostname_gpus_jobnum_lists": [["agpu44", 4, 2]],
    # safe mode: submit jobs on gpu forcibly or not. If false, Submit jobs on gpu forcibly, and lane, hostname_gpus_jobnum_lists and max_trying_time parameters will be used.
    # low_cache_mode: if CryoSPARC has no enough cache space, please set this to true
    # if_slurm: if use slurm to run model. Slurm parameters can be modified manually in GetConfidence.sh after running CreateParameterFiles.py
    "project": null,
    "workspace": null,
    "lane": "default",
    "hostname_gpus_jobnum_lists": [],
    "safe_mode": true,
    "low_cache_mode": false,
    
    # max_trying_time: (if safe_mode is true) how many times to restart a job if this job failed
    # max_particle_num_to_use: max particles used to score and select after running ImportAndExtract.py
    # base_abinit_particle_num and abinit_particle_num_step: e.g. if base_abinit_particle_num = 50000 and abinit_particle_num_step = 2, pipeline will create 2 abinit jobs and use top 50000 and top 100000 particles respectively
    # refine_iteration_num: how many times to iterate in pipeline
    # refine_iteration_min_particle_number_step: ignore refine_iteration_num, if particle number step in some iteration if smaller than this parameter, stop iteration.
    # refine_grid_num_in_first_turn: how many refine job will be created in first iteration. Need even number.
    # refine_grid_num_in_rest_each_turn: how many refine job will be created in rest iteration. Need even number.
    # min_refine_truncation_confidence and max_refine_truncation_confidence: selection region by score
    # min_refine_particle_num and max_refine_particle_num: selection region by particle number
    "max_trying_time": 3,
    "max_particle_num_to_use": null,
    "base_abinit_particle_num": 50000,
    "abinit_particle_num_step": 2,
    "refine_iteration_num": 2,
    "refine_iteration_min_particle_number_step": null,
    "refine_grid_num_in_first_turn": 8,
    "refine_grid_num_in_rest_each_turn": 4,
    "min_refine_truncation_confidence": 0.8,
    "max_refine_truncation_confidence": 0.3,
    "min_refine_particle_num": 30000,
    "max_refine_particle_num": 3000000,
    
    # initial_orientation_balance: balance orientation or not
    # cfar_lower_bound: cfar lower bound to judge prefer orientation volume quality
    # resolution_lower_bound: resolution lower bound to judge prefer orientation volume quality
    # k: cluster number
    "initial_orientation_balance": false,
    "cfar_lower_bound": 0.15,
    "resolution_lower_bound": 4.0,
    "particle_num_multiple_for_cluster": 8.0,
    "k": 8,
    
    # if_slurm: if use slurm to run model. Slurm parameters can be modified manually in GetConfidence.sh after creating base parameters.
    # delete_cache: delete cache folder after running SelectAndRefine.py or not
    # inference_gpu_ids: when running CryoRanker inference during CryoWizard, it will use all gpus which are available by default. And you can set this parameter to indicate the gpu ids you would like to use (e.g. '0,2' means gpu 0 and 2 will be used)
    # accelerate_port_start and accelerate_port_end: this means accelerate process will choose port during 28000-30000.
    "if_slurm": false,
    "delete_cache": true,
    "inference_gpu_ids": null,
    "accelerate_port_start": 28000,
    "accelerate_port_end": 30000