# A comprehensive foundation model for cryo-EM image processing


<a href="https://doi.org/10.1101/2024.11.04.621604"><img src="https://img.shields.io/badge/Paper-bioRxiv-green" style="max-width: 100%;"></a>
<a href="https://huggingface.co/westlake-repl/Cryo-IEF"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000" style="max-width: 100%;"></a>

We present the Cryo-EM Image Evaluation Foundation (Cryo-IEF) model, which has been pre-trained on a substantial dataset comprising approximately 65 million cryo-EM particle images using unsupervised learning techniques. Cryo-IEF excels in various cryo-EM data processing tasks, such as classifying particles from different structures, clustering particles by pose, and assessing the quality of particle images. Upon fine-tuning, the model effectively ranks particle images by quality, enabling the creation of CryoWizard—a fully automated single-particle cryo-EM data processing pipeline. 

Please cite the following paper if this work is useful for your research:
```
@article{yan2024comprehensive,
  title={A comprehensive foundation model for cryo-EM image processing},
  author={Yan, Yang and Fan, Shiqi and Yuan, Fajie and Shen, Huaizong},
  journal={bioRxiv},
  pages={2024--11},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```


## Installation

Since the use of CryoWizard requires reading and writing files in the Project created by CryoSPARC, we strongly recommend installing with a Linux user account that has the permission to read and write the CryoSPARC Project.

### Step 1: Conda Environment Setup

The environment can be configured using `pip` with the`requirements.txt` file:

    (base) $ conda create --name cryo_ief python=3.10
    (base) $ conda activate cryo_ief
    (cryo_ief) $ cd path/to/Cryo-IEF
    (cryo_ief) $ pip install -r requirements.txt

Installation may take several minutes. Subsequently, `cryosparc-tools` must be installed separately to ensure its version matches your CryoSPARC software. Please identify your CryoSPARC version and install the `cryosparc-tools` version that is closest to, and not exceeding, it.

For example, if your CryoSPARC version is 4.6.2, execute:
    
    (cryo_ief) $ pip install cryosparc-tools==999.999.999

You will get two ERROR messages like this:

    ERROR: Could not find a version that satisfies the requirement cryosparc-tools==999.999.999
    (from versions: 0.0.3, 4.1.0, 4.1.1, 4.1.2, 4.1.3, 4.2.0, 4.3.0, 4.3.1, 4.4.0, 4.4.1, 4.5.0, 4.5.1, 4.6.0, 4.6.1, 4.7.0)
    ERROR: No matching distribution found for cryosparc-tools==999.999.999

From the error output, identify the closest `cryosparc-tools` version (less than or equal to your CryoSPARC version), and then install it using the following command:

    (cryo_ief) $ pip install cryosparc-tools==4.6.1

Upon successful execution, the Conda environment setup will be complete.

### Step 2: Download Model Weights

Model weights are accessible via [HuggingFace](https://huggingface.co/westlake-repl/Cryo-IEF). Alternatively, they can be downloaded from the [Cryo-IEF google drive](https://drive.google.com/drive/folders/1C9jIdC5B58ohAwrfRalTngRtLtgIWfM8?usp=sharing) and [CryoRanker google drive](https://drive.google.com/drive/folders/10SUzFZB2s9sGCDkYF258Yx1C11D3tiph?usp=drive_link). 
This model is intended for **academic research only**. Commercial use is prohibited without explicit permission.



[//]: # (## Quickstart)
## Cryo-IEF
Cryo-IEF is a foundation model for cryo-EM image evaluation, pre-trained on an extensive dataset using unsupervised learning. 
To generate particle features with Cryo-IEF encoder, run the following command:

    (base) $ conda activate cryo_ief
    (cryo_ief) $ accelerate launch path/to/Cryo-IEF/code/CryoIEF_inference.py --path_result_dir dir/to/save/results --path_model_proj dir/to/CryoIEF_model_weight --raw_data_path dir/to/cryoSPARC_job
Cryo-IEF is compatible with cryoSPARC job types that generate particle outputs, including `Extracted Particles`, `Restack Particles`, and `Particles Sets`. 
By default, all available GPUs will be utilized for inference. Users can add `--num_processes` or list specific GPU IDs with `--gpu_ids` (e.g. `--gpu_ids 0,1,2,3`) between `accelerate launch` and `path/to/Cryo-IEF/code/CryoRanker_inference.py` to specify the number of GPUs to use.

The particle features extracted by the Cryo-IEF encoder are saved by default to `dir/to/save/results/features_all.data`.
The order of features corresponds to the particle order in the`.cs` file located within `dir/to/cryoSPARC_job`.

During Cryo-IEF inference, raw data undergoes preprocessing and is cached in `dir/to/save/results/processed_data`.
This cache can be safely deleted after inference.

## CryoRanker
CryoRanker integrates Cryo-IEF’s backbone encoder with an additional classification head, 
fine-tuned on a labeled dataset to rank particle images by quality.

    (base) $ conda activate cryo_ief
    (cryo_ief) $ accelerate launch path/to/Cryo-IEF/code/CryoRanker_inference.py --path_result_dir dir/to/save/results --path_model_proj dir/to/CryoRanker_model_weight --raw_data_path dir/to/cryoSPARC_job --num_select N
CryoRanker is compatible with cryoSPARC job types that generate particle outputs, including `Extracted Particles`, `Restack Particles`, and `Particles Sets`. 
By default, all available GPUs will be utilized for inference. Users can add `--num_processes` or list specific GPU IDs with `--gpu_ids` (e.g. `--gpu_ids 0,1,2,3`) between `accelerate launch` and `path/to/Cryo-IEF/code/CryoRanker_inference.py` to specify the number of GPUs to use.

Predicted scores are saved in `dir/to/save/results/scores_predicted_list.csv`.
The order of scores corresponds to the particle order in the `.cs` file located within `dir_to_cryoSPARC_job`.
If `--num_select` is set to `N`, the top `N` particles will be selected and saved as `dir/to/save/results/selected_particles_top_N.cs` and `.../selected_particles_top_N.star`. 
These selected particles can then be loaded into cryoSPARC or RELION for subsequent processing.

During CryoRanker inference, raw data undergoes preprocessing and is cached in `dir/to/save/results/processed_data`.
This cache can be safely deleted after inference.

## CryoWizard ![Beta Badge](https://img.shields.io/badge/status-beta-yellow)
⚠️ CryoWizard is in beta. Expect updates and potential changes to features. Please report any issues encountered.

### Run Install Progress

    (base) $ conda activate cryo_ief
    (cryo_ief) $ cd path/to/Cryo-IEF
    (cryo_ief) $ python CryoWizard_main.py \
        --CryoWizardInstall \
        --cryosparc_username 'your_cryosparc_username' \
        --cryosparc_password 'your_cryosparc_password' \
        --cryosparc_license 'XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX' \
        --cryosparc_hostname your_host_name \
        --cryosparc_port 39000 \
        --cryoranker_model_weight 'path/to/your/downloaded/cryo_ranker_model_weight_folder'

CryoWizard process will automatically queue CryoSPARC jobs during running, and these jobs will run on CryoSPARC lanes. However, the model inference task could not use GPUs on CryoSPARC lanes, and it will perform the inference task on current node (the command execution or the web/extension application launching node). Please make sure there are GPUs available here.

Of course, you also can use Slurm to run your model inference task during CryoWizard running. If you want that CryoWizard would submit model inference tasks with [Slurm](https://slurm.schedmd.com/) automatically during running, there are two ways to set:

1. Global setting, after modifying this, all CryoWizard process will use this setting: Set `if_slurm` to `true` in `path/to/Cryo-IEF/CryoWizard/cryowizard_settings.yml`. Then, modify header settings in `path/to/Cryo-IEF/CryoWizard/parameters/GetConfidence.sh`, and CryoWizard will using these header settings to run model inference task on slurm. Please note that do NOT set `#SBATCH -o` and `#SBATCH -e` parameters in `GetConfidence.sh`, CryoWizard would set these two settings during running automatically.
2. Case by case setting, set in `path/to/save/your/cryowizard/metadata/folder` every time after creating base parameters: Set `if_slurm` to `true` in `path/to/save/your/cryowizard/metadata/folder/parameters/parameters.json` and modify the header setting in `GetConfidence.sh` in `path/to/save/your/cryowizard/metadata/folder/parameters`, and CryoWizard will using these header settings to run model inference task on slurm. Please note that do NOT set `#SBATCH -o` and `#SBATCH -e` parameters in `GetConfidence.sh`, CryoWizard would set these two settings during running automatically.

And please make sure that slurm is installed on your server.


### Using CryoWizard via Chrome Extension

Doc: [CryoWizard Chrome Extension](readmes/CryoWizard_crx/README.md)

### Using CryoWizard via Web Interface

Doc: [CryoWizard Web](readmes/CryoWizard_web/README.md)

### Using CryoWizard via Command

Doc: [CryoWizard Command](readmes/CryoWizard_cmd/README.md)

### (Optional) More Settings in Parameters.json

Doc: [CryoWizard Optional Settions](readmes/CryoWizard_opt/README.md)


