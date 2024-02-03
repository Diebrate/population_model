# Modeling Single Cell Trajectory Using Forward-Backward Stochastic Differential Equations

Research Project by Kevin Zhang
Preprint paper avaiable on [bioarxiv](https://www.biorxiv.org/content/10.1101/2023.08.10.552373v1.abstract)

# Dependency

- numpy
- pandas
- sklearn
- matplotlib
- seaborn
- pytorch
- pot

# Instructions

The repository is able to train 4 different models.

- TrajectoryNet by [(Tong, 2020)](https://github.com/KrishnaswamyLab/TrajectoryNet)
- Waddingtong-OT by [(Schiebinger, 2019)](https://pubmed.ncbi.nlm.nih.gov/30712874/)
- Stationary-OT by [(Zhang, 2021)](https://github.com/zsteve/StationaryOT)
- FBSDE [(preprint)](https://www.biorxiv.org/content/10.1101/2023.08.10.552373v1.abstract)

To train the model, run the [workflow_single.py](workflow_single.py) file.
To perform parameter tuning using validation, run the [workflow_valid.py](workflow_valid.py) file.
To summarize the performance from simulations, run the [workflow_eval.py](workflow_eval.py) file.