# ICE-T
**I**nteractions-aware Cross-column **C**ontrastive **E**mbedding for Heterogeneous **T**abular Datasets

## Description
This repository contains the code and supplementary materials for the **ICE-T** project, along the obtained results. Adjustments can be made based on additional details or features you might want to include in the repository.

## Requirements
The necessary packages and additional dependencies can be installed using `pip` based on the attached `requirements.txt` file:

```console
pip install -r requirements.txt
```

## Adjusting configuration
The current configuration assumes the availability of at least 4 GPUs, CUDA v12.2 or later, and a CPU that supports at least 24 workers. If your hardware configuration does not meet these criteria, it is necessary to modify the `./src/config.yml` file by adjusting the `WHICH_DEVICE` and `N_WORKERS` parameters. To run without GPU support, it is necessary to set `WHICH_DEVICE` to `cpu`.

## Running Experiments
To run experiments on individual datasets, use the following command:

```console
python ./experiment.py --dataset dataset_name
```
To run experiments in batch, you may use the attached `.sh` script. First, make the script executable by running:

```console
chmod +x ./run.sh
```

Then, execute the script:

```console
./run.sh
```

The obtained results are stored in `./results/` folder. 

