# A comprehensive foundation model for cryo-EM image processing
We present the Cryo-EM Image Evaluation Foundation (Cryo-IEF) model, which has been pre-trained on a substantial dataset comprising approximately 65 million cryo-EM particle images using unsupervised learning techniques. Cryo-IEF excels in various cryo-EM data processing tasks, such as classifying particles from different structures, clustering particles by pose, and assessing the quality of particle images. Upon fine-tuning, the model effectively ranks particle images by quality, enabling the creation of CryoWizard—a fully automated single-particle cryo-EM data processing pipeline. 


## Installation

The CryoWizard pipeline utilizes APIs provided by CryoSPARC. Please ensure that you have installed [CryoSPARC](https://cryosparc.com/). 

You can set up the environment using the `requirements.txt` via `pip`:

    (base) $ conda create --name cryowizard python=3.10
    (base) $ conda activate cryowizard
    (cryowizard) $ cd path/to/requirements
    (cryowizard) $ pip install -r requirements.txt

Note that the version of `cryosparc-tools` specified in `requirements.txt` must correspond to your CryoSPARC installation.

Alternatively, you can install the required packages using `conda`. All necessary packages are listed in `requirements.txt`, with the exception of `conda install`, which can currently only be installed via `pip` (see [cryosparc-tools](https://tools.cryosparc.com/intro.html)).

Model weights are available at [Cryo-IEF google drive](https://drive.google.com/drive/folders/1C9jIdC5B58ohAwrfRalTngRtLtgIWfM8?usp=sharing) and [CryoRanker google drive](https://drive.google.com/drive/folders/10SUzFZB2s9sGCDkYF258Yx1C11D3tiph?usp=drive_link).

After installing the environment, some necessary settings must be configured. First, modify the file `path/to/cryowizard/code/CryoRanker/classification_inference_settings.yml` to set

    path_model_proj: path/to/downloaded/model/weights/2024_0612_165421_full_all_4096_0.6_/checkpoint/checkpoint_epoch_27/

Next, open `path/to/cryowizard/code/MyLib/cs_login_info.json` and adjust the parameters within this file (these parameters are used to access your CryoSPARC account).

[//]: # (## Quickstart)
## Cryo-IEF
Cryo-IEF is a foundation model for cryo-EM image evaluation, pre-trained on an extensive dataset using unsupervised learning. 
To generate particle features with Cryo-IEF encoder, run the following command:

    (base) $ conda activate cryowizard
    (cryowizard) $ python path/to/cryowizard/code/CryoIEF_inference.py --path_result_dir dir_to_save_results --path_model_proj dir_to_the_CryoIEF_model_weight --raw_data_path dir_to_cryoSPARC_job
Cryo-IEF processes only cryoSPARC job types that output particles, such as `Extracted Particles`, `Restack Particles`, and `Particles Sets`. 

The particle features extracted by the Cryo-IEF encoder are saved by default in `dir_to_save_results/features_all.data`.
The order of features aligns with the particle order in the`.cs` file located in `dir_to_cryoSPARC_job`.

During Cryo-IEF inference, raw data is preprocessed and cached in `dir_to_save_results/processed_data`.
After inference, this cache can be deleted while retaining the other output files.

## CryoRanker
CryoRanker integrates Cryo-IEF’s backbone encoder with an additional classification head, 
fine-tuned on a labeled dataset to rank particle images by quality.

    (base) $ conda activate cryowizard
    (cryowizard) $ python path/to/cryowizard/code/CryoRanker_inference.py --path_result_dir dir_to_save_results --path_model_proj dir_to_the_CryoRanker_model_weight --raw_data_path dir_to_cryoSPARC_job
CryoRanker processes only cryoSPARC job types that output particles, such as `Extracted Particles`, `Restack Particles`, and `Particles Sets`. 

The predicted scores are saved in `dir_to_save_results/scores_predicted_list.csv`.
The order of scores aligns with the particle order in the`.cs` file located in `dir_to_cryoSPARC_job`.

During CryoRanker inference, raw data is preprocessed and cached in `dir_to_save_results/processed_data`.
After inference, this cache can be deleted while retaining the other output files.

## CryoWizard
The released version is under development.