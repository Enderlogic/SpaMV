# SpaMV: An interpretable spatial multi-omics data integration and dimension reduction algorithm

# Installation

1) Create and activate a conda environment with python 3.12

```
conda env create spamv python==3.12
conda activate spamv
```

2) Before you install our package, please make sure you have installed the pyg-lib package.

```
# For CPU users
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
# For GPU users
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.6.0+cuda118.html
```

3) Then you could install our package as follows:

```
pip install spamv
```

# Tutorial

We provide two jupyter notebooks (Tutorial_simulation.ipynb and Tutorial_realworld.ipynb) to reproduce the results in
our paper. Before you run them, please make sure that you have downloaded the simulated data and/or real-world data from
our Zenodo repositoy.