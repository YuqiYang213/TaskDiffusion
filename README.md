#  [ICLR2025] Multi-Task Dense Predictions via Unleashing the Power of Diffusion
<!-- 
<p align="center">
  <img src="imgs/demo.gif" alt="demo">
</p> -->

##  Abstract
We provide the code for TaskDiffusion, a novel multi-task dense prediction framework based on diffusion models. Our code is implemented on PASCAL-Context and NYUD-v2 based on ViT.

- TaskDiffusion builds a novel decoder module based on diffusion model that can capture the underlying conditional distribution of the prediction.
- To further unlock the potential of diffusion models in solving multi-task dense predictions, TaskDiffusion introduces a novel joint denoising diffusion process to capture the task relations during denoising.
- Our proposed TaskDiffusion achieves a new state-of-the-art (SOTA) performance with superior efficiency on PASCAL-Context and NYUD-v2. 

Please check the [paper](https://openreview.net/pdf?id=TzdTRC85SQ) for more details.
<p align="center">
  <img alt="img-name" src="imgs/pipeline.png" width="800">
  <br>
    <em>Framework overview of the proposed TaskDiffusion for multi-task scene understanding.</em>
</p>

# Installation

## 1. Environment
You can use the following command to prepare your environment.
```bash
conda create -n taskdiffusion python=3.7
conda activate taskdiffusion
pip install tqdm Pillow==9.5 easydict pyyaml imageio scikit-image tensorboard six
pip install opencv-python==4.7.0.72 setuptools==59.5.0

pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.5.4 einops==0.4.1
```

## 2. Data
You can download the PASCAL-Context and NYUD-v2 from ATRC's repository in [PASCALContext.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/ER57KyZdEdxPtgMCai7ioV0BXCmAhYzwFftCwkTiMmuM7w?e=2Ex4ab),
[NYUDv2.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/EZ-2tWIDYSFKk7SCcHRimskBhgecungms4WFa_L-255GrQ?e=6jAt4c):
### PASCAL-Context
```bash
tar xfvz PASCALContext.tar.gz
```
### NYUD-v2
```bash
tar xfvz NYUDv2.tar.gz
```

**Attention**: you need to specify the root directory of your own datasets as ```db_root``` variable in ```configs/mypath.py```.


## 3. Training
You can train your own model by using the following commands.
PASCAL-Context:
```bash
bash run_TaskDiffusion_pascal.sh
```

NYUD-v2
```bash
bash run_TaskDiffusion_nyud.sh
```

If you want to train your model based on ViT-Base, you can modify the ```--config_exp``` in ```.sh``` file.

You can also modify the output directory in ```./configs```.

## 4. Evaluate the model
The training script itself includes evaluation. 
For inferring with pre-trained models, you can use the following commands.
PASCAL-Context:
```bash
bash infer_TaskDiffusion_pascal.sh
```

NYUD-v2
```bash
bash infer_TaskDiffusion_nyud.sh
```

For the evaluation of boundary, you can use the evaluation tools in this [repo](https://github.com/prismformore/Boundary-Detection-Evaluation-Tools) following TaskPrompter.

# Pre-trained models
We provide the pretrained models on PASCAL-Context and NYUD-v2.

### Download pre-trained models
|Version | Dataset | Download | Depth (RMSE) | Segmentation (mIoU) |  Human parsing (mIoU) | Saliency (maxF) | Normals (mErr) | Boundary (odsF) | 
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| TaskDiffusion (ViT-L)| PASCAL-Context | [Link](https://pan.baidu.com/s/1Eed7wipnllbZ5LIvehH30A?pwd=j9u5) (Extraction code: j9u5) | - |81.21 | 69.62 | 84.94 | 13.55 | 74.89 |
| TaskDiffusion /w MLoRE (ViT-L)| PASCAL-Context | [Link](https://pan.baidu.com/s/1Ir5SSCNQqDtvw7ZDm5SbnA?pwd=gwhp) (Extraction code: gwhp) | - |81.58 |  71.30 | 85.05 | 13.43 | 76.07 |
| TaskDiffusion (ViT-B)| PASCAL-Context | [Link](https://pan.baidu.com/s/1oXHTXj2B7T8hSC4FtOeaFQ?pwd=xidm) (Extraction code: xidm) | - | 78.83 | 67.40 | 85.31 | 13.38 | 74.68 |
| TaskDiffusion (ViT-L) | NYUD-v2 | [Link](https://pan.baidu.com/s/1PrxvOiNtJ77AwB8vRAW0GA?pwd=ngfp) (Extraction code: ngfp) | 0.5020 | 55.65  | - | - | 18.43 | 78.64 |
| TaskDiffusion /w MLoRE (ViT-L) | NYUD-v2 | [Link](https://pan.baidu.com/s/1Rh_0DVvmcxeTGdDRVyPl0w?pwd=fx2m) (Extraction code: fx2m) | 0.5033 | 56.66  | - | - | 18.13 | 78.89 |

### Infer with the pre-trained models
To evaluate the pre-trained models, you can change the ```--trained_model MODEL_PATH``` in ```infer.sh``` to load the specified model.

#  Cite
<!-- Please consider :star2: star our project to share with your community if you find this repository helpful! -->
If you find our work helpful, please cite:
BibTex:
```
@inproceedings{yangmulti,
  title={Multi-Task Dense Predictions via Unleashing the Power of Diffusion},
  author={Yang, Yuqi and Jiang, Peng-Tao and Hou, Qibin and Zhang, Hao and Chen, Jinwei and Li, Bo},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```

# Contact
If you have any questions, please feel free to contact Me(yangyq2000 AT mail DOT nankai DOT edu DOT cn).

# Acknowledgement
This repository is built upon the nice framework provided by [TaskPrompter and InvPT](https://github.com/prismformore/Multi-Task-Transformer).

