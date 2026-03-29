## Hydrology-Aware Contrastive Meta-Learning for Domain Generalization Hyperspectral Wetland Image Classification

<p align='center'>
  <img src='abstract_01.png' width="800px">
</p>

## Abstract

Wetlands represent vital ecosystems requiring precise monitoring for conservation, with hyperspectral imaging serving as a crucial remote sensing technology. However, significant domain shifts caused by hydrological dynamics (e.g., tidal fluctuations and seasonal water variations) and imaging environment disparities critically degrade model generalization in unseen wetland environments. To overcome this limitation, we propose HydroMetaNet—a hierarchical learning framework integrating hydrological prior knowledge with deep domain generalization. Our method lies in a hydrological-algorithmic co-design comprising: 1) A flexible wetland spectral-spatial adaptation network that dynamically encodes state-specific features through adaptive instance-batch normalization hybrid and spectral-spatial attention. 2) Wetland Multi-scale Feature Pyramid capturing hierarchical patterns from textures to large-scale distribution via parallel dilated convolutions; and 3) Hydro-Feature Enhancer disentangling water-impacted components through water-index-inspired saliency gating. These representations are synergistically optimized by invariant contrastive learning and adaptive meta-learning prototypes, establishing dynamic decision boundaries that absorb cross-scene hydrological differences. Evaluated extensively across three cross-domain wetland benchmarks, HydroMetaNet demonstrated overall accuracy improvements of 7.19\% in cross
region, 1.62\% in cross-seasonal, and 0.79\% in cross-sensor generalization tasks compared to the best baselines. The framework exhibits exceptional capability in minority-class discrimination while maintaining leading-average accuracy and Kappa metrics across all scenarios.

## Paper

Please cite our paper if you find the code or dataset useful for your research.

```
@ARTICLE

```



## Requirements

CUDA Version: 11.7

torch: 2.0.0

Python: 3.10

## Dataset

The dataset directory should look like this:

```bash
datasets
├── Wetland
│   ├── ZY_HHK_data108_20200628.mat
│   ├── ZY_HHK_gt108_20200628.mat
│   ├── ZY_HHK_data108_20210929.mat
│   └── ZY_HHK_gt108_20210929.mat
```

## Usageetland

1.You can download dataset: https://drive.google.com/drive/folders/1jREuL_jwmFN65ICFm_h-zbA9LrKirXAE?usp=drive_link.

2.You can change the `source_name` and `target_name` in train.py to set different transfer tasks.

3.Run the following command:

dataset1:
```
python train.py --data_path ./datasets/Wetland/ --source_name ZY1-02D_HHK_2020 --target_name ZY1-02D_Yancheng_A --re_ratio 1 --max_epoch 50 --log_interval 5 --training_sample_ratio  0.8 --batch_size 256 --seed 233
```
dataset2:
```
python train.py --data_path ./datasets/Wetland/ --source_name HHK_20200628 --target_name HHK_20210929 --re_ratio 1 --max_epoch 50 --log_interval 5  --training_sample_ratio 0.8 --batch_size 256 --seed 233
```
dataset3:
```
python train.py --data_path ./datasets/Wetland/ --source_name ZY1-02D_Yancheng_B --target_name GF5_Yancheng --re_ratio 1 --max_epoch 50 --log_interval 5 --dim 512 --training_sample_ratio 0.8 --seed 233 --batch_size 256 --seed 233
```



