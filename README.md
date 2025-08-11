<p align="center">
  <img src="assets/Altimeter.svg" alt="Altimeter Logo" width="256" height="256"/>
</p>

This repository contains the model and training code for Altimeter - a transformer model for peptide spectrum prediction.

## Koina deployment

Altimeter models are hosted on [Koina](https://koina.wilhelmlab.org) and can be queried via their REST API:

- [Altimeter_2024_splines_index](https://koina.wilhelmlab.org/docs#post-/Altimeter_2024_splines_index/infer)
- [Altimeter_2024_splines](https://koina.wilhelmlab.org/docs#post-/Altimeter_2024_splines/infer)
- [Altimeter_2024_intensities](https://koina.wilhelmlab.org/docs#post-/Altimeter_2024_intensities/infer)
- [Altimeter_2024_isotopes](https://koina.wilhelmlab.org/docs#post-/Altimeter_2024_isotopes/infer)

## Model overview

- **Data**: trained on a reprocessed ProteomeTools dataset covering tryptic and non-tryptic peptides (trypsin, LysC, AspN, HLA ligands) acquired on an Orbitrap Fusion Lumos with NCEs from 20–40; only methionine oxidation and cysteine carbamidomethylation were considered.
- **Model**: predicts fragment ion intensities across NCEs using cubic B-splines; an isotopes variant re-creates full fragment isotope patterns based on isolation efficiencies; validated for HCD on Orbitrap instruments only.
- **Performance**: achieves a median normalized spectral angle of 0.941 on a held-out test set and performs consistently across proteases, with slight degradation for long peptides or extreme charge states.
- **Input**: peptides 6–40 amino acids long (6–30 recommended), charge 1–7 (1–4 recommended), NCE between 20–40, with variable methionine oxidation and static carbamidomethylated cysteines.

## Dataset

The training data is available from Zenodo: https://zenodo.org/records/15875054

Download and unpack the archive into a working directory, e.g. `~/altimeter_data`:

```bash
wget https://zenodo.org/records/15875054/files/Altimeter_training_data.tar.gz?download=1 -O Altimeter_training_data.tar.gz
mkdir -p ~/altimeter_data
tar -xzf Altimeter_training_data.tar.gz -C ~/altimeter_data
```

After extraction, update `config/data.yaml` so the paths point to your dataset location:

```yaml
base_path: /path/to/altimeter_data/
ion_dictionary_path: /path/to/altimeter_data/saved_model/ion_dictionary.txt
dataset_path: datasets/
position_path: txt_pos/
label_path: labels/
saved_model_path: saved_model/
```

## Weights & Biases

The training script logs metrics to [Weights & Biases](https://wandb.ai/).

1. Create an account and obtain an API key.
2. Authenticate once on the command line:
   ```bash
   wandb login
   ```
   or set `WANDB_API_KEY` in your environment.
3. The default project name is `Altimeter`. Override it with the
   `WANDB_PROJECT` environment variable or by editing `altimeter/train.py`.
4. To run without logging, set `WANDB_MODE=offline`.

## Training

### Using Docker

```bash
docker pull dennisgoldfarb/pytorch_ris:lightning
docker run --gpus all -v $PWD:/workspace/Altimeter \
    -v /path/to/altimeter_data:/data \
    dennisgoldfarb/pytorch_ris:lightning \
    python altimeter/train.py config/data.yaml
```

### Without Docker

Create a Python environment with PyTorch, PyTorch Lightning, and the
repository's dependencies, then run:

```bash
python altimeter/train.py config/data.yaml
```

## Export to ONNX and TorchScript

The repository provides `export2onnx.py` to serialize the model for
serving. Run the script inside the Docker image used for training:

```bash
docker run --gpus all -v $PWD:/workspace/Altimeter \
    dennisgoldfarb/pytorch_ris:lightning \
    python altimeter/export.py \
         model.ts \
         model.onnx \
       --dic-config path/to/data.yaml \
       --model-config path/to/model_config.yaml \
       --model-ckpt path/to/checkpoint.ckpt
```

The first argument is the output path for the TorchScript model and the
second argument specifies the ONNX file for the spline model. Adjust the
paths to match your environment.


## Exporting

Convert a trained checkpoint to TorchScript and ONNX models:

```bash
python altimeter/export2onnx.py 
```
