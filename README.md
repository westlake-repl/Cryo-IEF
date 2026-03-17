# A Comprehensive Foundation Model for Cryo-EM Image Processing


<a href="https://doi.org/10.1038/s41592-025-02916-8"><img src="https://img.shields.io/badge/Paper-Nature%20Methods-blue" style="max-width: 100%;"></a>
<a href="https://huggingface.co/westlake-repl/Cryo-IEF"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000" style="max-width: 100%;"></a>

We present **Cryo-IEF** (Cryo-EM Image Evaluation Foundation) — a foundation model pre-trained on approximately 65 million cryo-EM particle images via unsupervised learning. Cryo-IEF provides powerful general-purpose representations for cryo-EM data, excelling at tasks such as classifying particles by structure, clustering particles by pose, and assessing image quality.

Building on Cryo-IEF, several downstream tools have been developed that apply its learned representations to specialized tasks:

| Tool | Description | Repository |
|------|-------------|------------|
| **CryoRanker** | Fine-tunes Cryo-IEF to rank particle images by quality for automated particle selection | This repository |
| **CryoDECO** | An _ab initio_ heterogeneous reconstruction algorithm that leverages Cryo-IEF priors to deconstruct compositional and conformational heterogeneity | [GitHub](https://github.com/yanyang1998/CryoDECO) |
| **CryoWizard** | Integrates CryoRanker into a fully automated single-particle cryo-EM processing pipeline | [GitHub](https://github.com/SMART-StructBio-AI/CryoWizard) |

Please cite the following paper if this work is useful for your research:
```
@article{yan_comprehensive_2025,
	title = {A comprehensive foundation model for cryo-{EM} image processing},
	issn = {1548-7105},
	url = {https://doi.org/10.1038/s41592-025-02916-8},
	doi = {10.1038/s41592-025-02916-8},
	abstract = {Cryogenic electron microscopy (cryo-EM) has become a premier technique for determining high-resolution structures of biological macromolecules. However, its broad application is constrained by the demand for specialized expertise. Here, to address this limitation, we introduce the Cryo-EM Image Evaluation Foundation (Cryo-IEF) model, a versatile tool pre-trained on {\textasciitilde}65 million cryo-EM particle images through unsupervised learning. Cryo-IEF performs diverse cryo-EM processing tasks, including particle classification by structure, pose-based clustering and image quality assessment. Building on this foundation, we developed CryoWizard, a fully automated single-particle cryo-EM processing pipeline enabled by fine-tuned Cryo-IEF for efficient particle quality ranking. CryoWizard resolves high-resolution structures across samples of varied properties and effectively mitigates the prevalent challenge of preferred orientation in cryo-EM.},
	journal = {Nature Methods},
	author = {Yan, Yang and Fan, Shiqi and Yuan, Fajie and Shen, Huaizong},
	month = nov,
	year = {2025},
}
```


## Installation

### Step 1: Conda Environment Setup

Create and activate a dedicated conda environment, then install dependencies:

    (base) $ conda create --name cryo_ief python=3.10
    (base) $ conda activate cryo_ief
    (cryo_ief) $ cd path/to/Cryo-IEF
    (cryo_ief) $ pip install -r requirements.txt

> **[CryoData](https://github.com/yanyang1998/cryoief-data)** is used internally for cryo-EM data preprocessing (normalization, LMDB conversion, and PyTorch integration). It is included in `requirements.txt` and installed automatically.

Installation may take several minutes. Subsequently, `cryosparc-tools` must be installed separately to match your CryoSPARC version. To identify the correct version, first run:

    (cryo_ief) $ pip install cryosparc-tools==999.999.999

This will fail and print all available versions, for example:

    ERROR: Could not find a version that satisfies the requirement cryosparc-tools==999.999.999
    (from versions: 0.0.3, 4.1.0, 4.1.1, 4.1.2, 4.1.3, 4.2.0, 4.3.0, 4.3.1, 4.4.0, 4.4.1, 4.5.0, 4.5.1, 4.6.0, 4.6.1, 4.7.0)
    ERROR: No matching distribution found for cryosparc-tools==999.999.999

From the output, select the version closest to (but not exceeding) your CryoSPARC version and install it:

    (cryo_ief) $ pip install cryosparc-tools==4.6.1

### Step 2: Download Model Weights

Model weights are hosted on Hugging Face:

- [Cryo-IEF weights](https://huggingface.co/westlake-repl/Cryo-IEF/tree/main/cryo_ief_checkpoint/cryo_ief_v1_vit_b)
- [CryoRanker weights](https://huggingface.co/westlake-repl/Cryo-IEF/tree/main/cryo_ranker_checkpoint)

> These models are intended for **academic research only**. Commercial use is prohibited without explicit permission.


## Cryo-IEF

Cryo-IEF is a foundation model for cryo-EM image evaluation, pre-trained on an extensive dataset using unsupervised learning. To extract particle features with the Cryo-IEF encoder, run:

    (base) $ conda activate cryo_ief
    (cryo_ief) $ accelerate launch path/to/Cryo-IEF/code/CryoIEF_inference.py \
        --path_result_dir dir/to/save/results \
        --path_model_proj dir/to/CryoIEF_model_weight \
        --raw_data_path dir/to/cryoSPARC_job

Cryo-IEF is compatible with cryoSPARC job types that produce particle outputs, including `Extracted Particles`, `Restack Particles`, and `Particle Sets`. By default, all available GPUs are used. To restrict GPU usage, add `--num_processes N` or `--gpu_ids 0,1,2,3` between `accelerate launch` and the script path.

**Output:** Particle features are saved to `dir/to/save/results/features_all.data`. Feature order matches the particle order in the `.cs` file within `dir/to/cryoSPARC_job`.

**Cache:** Raw data is preprocessed and cached to `dir/to/save/results/processed_data`, which can be safely deleted after inference.


## Downstream Works

Cryo-IEF serves as the foundation for the following downstream tools. Each extends Cryo-IEF's learned representations to address specific challenges in cryo-EM data processing.

### CryoRanker

CryoRanker builds on the Cryo-IEF backbone by adding a classification head, fine-tuned on a labeled dataset to score individual particle images by quality. This enables reliable, automated selection of high-quality particles prior to 3D reconstruction.

To run CryoRanker inference:

    (base) $ conda activate cryo_ief
    (cryo_ief) $ accelerate launch path/to/Cryo-IEF/code/CryoRanker_inference.py \
        --path_result_dir dir/to/save/results \
        --path_model_proj dir/to/CryoRanker_model_weight \
        --raw_data_path dir/to/cryoSPARC_job \
        --num_select N

CryoRanker accepts the same cryoSPARC job types as Cryo-IEF (`Extracted Particles`, `Restack Particles`, `Particle Sets`). GPU options are identical: use `--num_processes N` or `--gpu_ids 0,1,2,3` to control GPU allocation.

**Output:**
- `dir/to/save/results/scores_predicted_list.csv` — predicted quality scores in particle order.
- `dir/to/save/results/selected_particles_top_N.cs` and `.star` — top `N` selected particles (only written when `--num_select N` is set). These files can be loaded directly into cryoSPARC or RELION for downstream processing.

**Cache:** Preprocessed data is cached to `dir/to/save/results/processed_data` and can be safely deleted after inference.

---

### CryoDECO

CryoDECO is an _ab initio_ heterogeneous reconstruction algorithm that leverages structural priors from the Cryo-IEF foundation model to resolve complex structural mixtures in cryo-EM data. By bypassing the random initialization bottleneck common in traditional deep learning approaches, it enables robust classification of both **compositional and conformational heterogeneity**.

> For full documentation and the latest updates, please visit the CryoDECO repository:
> **[https://github.com/yanyang1998/CryoDECO](https://github.com/yanyang1998/CryoDECO)**

---

### CryoWizard

CryoWizard is a fully automated single-particle cryo-EM data processing pipeline that integrates CryoRanker into an end-to-end workflow interfacing with CryoSPARC. It resolves high-resolution structures across samples with varied properties and effectively addresses the common challenge of preferred particle orientation. CryoWizard can be operated via a Chrome Extension, a Web Interface, or the command line.

> CryoWizard has been moved to a dedicated repository. For full documentation and the latest updates, please visit:
> **[https://github.com/SMART-StructBio-AI/CryoWizard](https://github.com/SMART-StructBio-AI/CryoWizard)**