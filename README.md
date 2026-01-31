# Virus-human protein-protein interactions predict viral phenotypes

Update ...

2026-1-31

## Workflow for predicting viral phenotypes based on virus-human PPIs
![Workflow of phenotype prediction](./fig_saved/Fig1B_workflow.pdf)

## Requirements
```python
Python==3.8.20
pytorch==2.4.0
scikit-learn==1.3.2
xgboost==2.1.1
pandas==2.0.3
numpy==1.24.4
matplotlib==3.7.5
```


## Installation

```python
conda env create -f environment.yml

OR

conda create --name phenotype_pred python=3.8.20

pip install -r requirements.txt
```

## Reproduce figures in paper
```python
./scripts/reproduce_figures.ipynb
```

## Citation
```

```