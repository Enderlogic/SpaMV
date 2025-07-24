# SpaMV: An interpretable spatial multi-omics data integration and dimension reduction algorithm

# Installation

1) Create and activate a conda environment with python 3.12

```
conda env create spamv python==3.12
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

# Tutorial

We provide two jupyter notebooks (Tutorial_simulation.ipynb and Tutorial_realworld.ipynb) to reproduce the results in
our paper. Before you run them, please make sure that you have downloaded the simulated data and/or real-world data from
our Zenodo repositoy.