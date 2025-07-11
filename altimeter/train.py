import sys
import numpy as np
import os
import yaml
import utils_unispec
import torch
from dataset import AltimeterDataModule
from lightning_model import LitFlipyFlopy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from altimeter.plot import MirrorPlotCallback

torch.set_float32_matmul_precision('medium')

###############################################################################
############################### Configuration #################################
###############################################################################

if len(sys.argv) == 2:
    data_yaml_path = sys.argv[1]
else:
    data_yaml_path = os.path.join(os.path.dirname(__file__), "../config/data.yaml")

with open(os.path.join(os.path.dirname(__file__), "../config/mods.yaml"), 'r') as stream:
    mod_config = yaml.safe_load(stream)
with open(data_yaml_path, 'r') as stream:
    config = yaml.safe_load(stream)
D = utils_unispec.DicObj(config['ion_dictionary_path'], mod_config, config['seq_len'], config['chlim'])

saved_model_path = os.path.join(config['base_path'], config['saved_model_path'])

# Configuration dictionary
if config['config'] is not None:
    # Load model config
    with open(config['config'], 'r') as stream:
        model_config = yaml.safe_load(stream)
else:
    channels = D.seq_channels
    model_config = {
        'in_ch': channels,
        'seq_len': D.seq_len,
        'out_dim': len(D.ion2index),
        **config['model_config']
    }

###############################################################################
############################ Weights and Biases ###############################
###############################################################################

wandb_logger = WandbLogger(project="Altimeter", 
                           config = config, 
                           log_model=False, 
                           save_dir="/scratch1/fs1/d.goldfarb/Backpack/")

###############################################################################
################################## Model ######################################
###############################################################################

# Instantiate model
litmodel = LitFlipyFlopy(config, model_config)


# Load weights
if config['weights'] is not None:
    litmodel.model.load_state_dict(torch.load(config['weights']))

# TRANSFER LEARNING
if config['transfer'] is not None:
    litmodel.model.final = torch.nn.Sequential(torch.nn.Linear(512,D.dicsz), torch.nn.Sigmoid())
    for parm in litmodel.model.parameters(): parm.requires_grad=False
    for parm in litmodel.model.final.parameters(): parm.requires_grad=True
    
sys.stdout.write("Total model parameters: ")
litmodel.model.total_params()
    

###############################################################################
############################# Reproducability #################################
###############################################################################

model_folder_path = os.path.join(config['base_path'], config['saved_model_path'])
if not os.path.exists(model_folder_path): os.makedirs(model_folder_path)
with open(os.path.join(model_folder_path, "model_config.yaml"), "w") as file:
    yaml.dump(model_config, file)
with open(os.path.join(model_folder_path, "data.yaml"), "w") as file:
    yaml.dump(config, file)
with open(os.path.join(model_folder_path, "ion_dictionary.txt"), 'w') as file:
    file.write(open(config['ion_dictionary_path']).read())
    

###############################################################################
########################## Training and testing ###############################
###############################################################################
stopping_criteria = EarlyStopping(monitor="val_SA_mean", mode="max", min_delta=0.00, patience=5)
checkpoint_callback = ModelCheckpoint(dirpath=saved_model_path, save_top_k=5, monitor="val_SA_mean", mode="max", every_n_epochs=1)

dm = AltimeterDataModule(config, D)
mirrorplot_callback = MirrorPlotCallback(dm)
trainer = L.Trainer(default_root_dir=saved_model_path,
                    logger=wandb_logger,
                    callbacks=[stopping_criteria, checkpoint_callback, mirrorplot_callback],
                    strategy="ddp_find_unused_parameters_true",
                    max_epochs=config['epochs'],
                    )

checkpoint_path = (
    os.path.join(config['base_path'], config['saved_model_path'], config['restart'])
    if config['restart'] else None
)

trainer.fit(litmodel, datamodule=dm, ckpt_path=checkpoint_path)
#trainer.test(litmodel, datamodule=dm, ckpt_path=checkpoint_path)
#trainer.predict(litmodel, datamodule=dm, ckpt_path=checkpoint_path)



