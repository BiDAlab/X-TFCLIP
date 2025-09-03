#  X-TFCLIP: An Extended TF-CLIP Method Adapted to Aerial-Ground Person ReIdentification

![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2506.22843-b31b1b.svg)](https://arxiv.org/abs/2506.22843)


**X-TFCLIP** is the winner of the [AG-VPReID 2025: The 2nd Aerial-Ground Person ReID competition](https://www.kaggle.com/competitions/agvpreid25/overview) .It is an extended version of the [TF-CLIP](https://github.com/AsuradaYuci/TF-CLIP) framework that leverages temporal and visual-language pretraining (CLIP) for video based aerial-ground person re-identification.

![My SVG](imgs/X_TFCLIP.svg)

X-TFCLIP improves over TF-CLIP across all metrics in the [AG-VPReID](https://openaccess.thecvf.com/content/CVPR2025/html/Nguyen_AG-VPReID_A_Challenging_Large-Scale_Benchmark_for_Aerial-Ground_Video-based_Person_Re-Identification_CVPR_2025_paper.html) based challenge dataset:

| Method         | Aerial→Ground R1 | R5   | R10  | mAP  | Ground→Aerial R1 | R5   | R10  | mAP  | Overall R1 | R5   | R10  | mAP  |
|----------------|-----------------|------|------|------|------------------|------|------|------|------------|------|------|------|
| **X-TFCLIP**   | **72.28**       | **81.94** | **88.81** | **74.45** | **70.77** | **82.59** | **86.08** | **72.67** | **71.56** | **82.25** | **85.94** | **73.60** |
| **TF-CLIP**    | 63.08 | 75.16 | 79.89 | 65.52 | 64.49 | 79.86 | 83.97 | 67.07 | 63.75 | 77.40 | 81.83 | 66.26 |

---
## Environment Setup
```bash
# Create and activate environment
conda create -n xtfclip python=3.12.9
conda activate xtfclip

# Install PyTorch with CUDA 11.3
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Install additional dependencies
pip install yacs timm scikit-image tqdm ftfy regex
```
---
## Training
To train the model, run:
```bash
python train_main.py --output_dir "logs/all"
```
---
## Evaluation

The repository supports two cross-view matching scenarios:

1. **Aerial-to-Ground matching:**

```bash
python eval_main.py --custom_output_dir "results/case1_aerial_to_ground" --output_dir "logs/all"
```
2. **Ground-to-Aerial matching:**
```bash
python eval_main.py --custom_output_dir "results/case2_ground_to_aerial" --output_dir "logs/all"
```
Note: For case 2, you need to modify the dataset path in datasets/set/agreidvid.py to point to case2_ground_to_aerial for query and gallery.

---
## Paper

X-TFCLIP achieved the **1st place** in the [AG-VPReID 2025]((https://agvpreid25.github.io/)): The 2nd Aerial-Ground Person ReID Challenge.

Please consider citing the following article if you found this work helpful. 

```bibtex
@misc{nguyen2025agvpreid2025aerialgroundvideobased,
      title={AG-VPReID 2025: Aerial-Ground Video-based Person Re-identification Challenge Results}, 
      author={Kien Nguyen and Clinton Fookes and Sridha Sridharan and Huy Nguyen and Feng Liu and Xiaoming Liu and Arun Ross and Dana Michalski and Tamás Endrei and Ivan DeAndres-Tame and Ruben Tolosana and Ruben Vera-Rodriguez and Aythami Morales and Julian Fierrez and Javier Ortega-Garcia and Zijing Gong and Yuhao Wang and Xuehu Liu and Pingping Zhang and Md Rashidunnabi and Hugo Proença and Kailash A. Hambarde and Saeid Rezaei},
      year={2025},
      eprint={2506.22843},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.22843}, 
}
```

---

## ✨ Extra features compared to TF-CLIP

- 🔍 **Bicubic CLIP-VIT positional embedding resizing** 
- 🧠 **Lightweight Attention Pooling**
- 🧭 **Online Label Smooth Loss**
- 🎯 **Video Frame Positional Embeddings**
- ⚙️ **Learnable Clip Memory Weighing**
- 💬 **Instance Norm Based BNN-Neck**
- 🔧 **Soft-Biometric Based Distance Matrix Masking** 

Please refer to the original [GitHub repo](https://github.com/agvpreid25/AG-VPReID) for additional code implementations on which this method is based on. 

---
## Acknowledgement

This baseline is based on the work of [TF-CLIP](https://github.com/AsuradaYuci/TF-CLIP). We appreciate the authors for their excellent contribution.

---

[//]: # (## 🛠 Installation)

[//]: # ()
[//]: # (```bash)

[//]: # (git clone https://github.com/BiDAlab/X-TFCLIP.git)

[//]: # (cd X-TFCLIP)

[//]: # (conda env create -f environment.yaml)

[//]: # (conda activate xtfclip)
