# Enhancing Prompt Generation with Adaptive Refinement for Camouflaged Object Detection
📄 [Paper (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Enhancing_Prompt_Generation_with_Adaptive_Refinement_for_Camouflaged_Object_Detection_ICCV_2025_paper.pdf)

## 🌼Requirements
Our environment is built upon [MedSAM](https://github.com/bowang-lab/MedSAM).  
For reference, we also provide our `environment.yml` file.  
You can reproduce the environment used in our paper by running:

```bash
conda env create -f environment.yml
```

## 🚀Getting Started
### 1️⃣ Download Pretrained Weights
Please download the following pretrained weights from [Hugging Face](https://huggingface.co/) and place them in the **same directory** as `train.py`.  
Your folder structure should look like this:
```bash
├── train.py
├── ...
├── blip-image-captioning-large/
├── blip-itm-base-coco/
└── mamba-130m-hf/
```

### 2️⃣ Download COD Dataset
Please download the COD-related datasets and organize them as follows:
```bash
├── train.py
├── ...
├── data/
│   ├── TrainDataset/
│   └── TestDataset/
│       ├── CHAMELEON/
│       ├── CAMO/
│       └── COD10K/
```

### 3️⃣ Pre-save BLIP-related Variables to Reduce GPU Memory Usage During Training:
```bash
python BLIP_infoSave.py
```

### 4️⃣ Model Training:
```bash
python train.py
```

## 🌷Acknowledgments
Part of our implementation builds upon the excellent work of [MedSAM](https://github.com/bowang-lab/MedSAM/tree/main), [CLIP-ES](https://github.com/linyq2117/CLIP-ES?tab=readme-ov-file#clip-is-also-an-efficient-segmenter-a-text-driven-approach-for-weakly-supervised-semantic-segmentation-cvpr-2023) and [ALBEF](https://github.com/salesforce/ALBEF). We sincerely appreciate their contributions to the field.

