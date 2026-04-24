## SpaMV: Interpretable Spatial Multi-Omics Integration

SpaMV is a spatial multi-view representation learning framework for integrating spatial multi-omics data. It is designed
to capture both **shared structure across omics** and **omics-specific signals**, enabling interpretable downstream
analysis such as clustering and visualization.

---

## ✨ Highlights

- Integrates multiple spatial omics modalities
- Preserves omics-specific biological signals
- Produces interpretable low-dimensional embeddings

---

## 🚀 Quick Start

### 1. Create environment
```bash
conda create -n spamv python=3.12 rpy2 scanpy r-mclust=6.1.1
conda activate spamv
```
### 2. Install SpaMV
```bash
pip install spamv
```

### 3. Download sample data
```bash
wget https://zenodo.org/records/16436314/files/Data.zip
unzip Data.zip
```

### 4. Run tutorial
```bash
# to reproduce our results on simulated datasets
python Tutorial_simulation.py
# to reproduce our results on realworld datasets
python Tutorial_realworld.py
```

---

## 📦 Installation

1) Create and activate a conda environment with Python 3.12, rpy2, r-mclust 6.1.1, and scanpy

```
conda create -n spamv python==3.12 rpy2 r-mclust=6.1.1 scanpy
conda activate spamv
```

2) (Optional) If you want to apply our algorithm to large datasets (with more than 10,000 spots), please make sure you have
   installed the pyg-lib package.

```
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

where

- `${TORCH}` should be replaced by either `1.13.0`, `2.0.0`, `2.1.0`, `2.2.0`, `2.3.0`, `2.4.0`, `2.5.0`, `2.6.0`, or
  `2.7.0`
- `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu117`, `cu118`, `cu121`, `cu124`, `cu126`, or `cu128`

3) Then you can install our package as follows:

```
pip install spamv
```
---

## 📊 Data

All datasets used in this project are publicly available:  

https://zenodo.org/records/16436314  

### Data contents
- Simulated datasets  
- Real-world spatial multi-omics datasets
---

## 📚Documentation
Additional tutorials and documentation:  
https://spamv-tutorials.readthedocs.io/  

---

## 📜 License
This project is licensed under the MIT License.