import torch
import random
import argparse
import numpy as np
import ipdb as pdb
import os, pwd, yaml
from datetime import datetime
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from CHiLD.modules.instantaneous import InstantaneousProcess
from CHiLD.tools.utils import load_yaml, setup_seed
from CHiLD.datasets.sim_dataset import StationaryDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
wandb.login(key="02f68fdb3367f7ff8ef0fd961bd1758e6e57dd24")

from pytorch_lightning.loggers import WandbLogger

import warnings
warnings.filterwarnings('ignore')

def main(args):

    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    
    current_user = pwd.getpwuid(os.getuid()).pw_name
    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('../CHiLD/configs', 
                            '%s.yaml'%args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)
    cfg = load_yaml(abs_file_path)
    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")

    pl.seed_everything(args.seed)

    # REMEMBER TO DELETE THIS BEFORE SUBMIT
    data = StationaryDataset(args.data)
    print("dataset", args.data)
    # exit()

    num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']
    train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])

    train_loader = DataLoader(train_data, 
                              batch_size=cfg['VAE']['TRAIN_BS'], 
                              pin_memory=cfg['VAE']['PIN'],
                              num_workers=cfg['VAE']['CPU'],
                              drop_last=False,
                              shuffle=True)

    val_loader = DataLoader(val_data, 
                            batch_size=cfg['VAE']['VAL_BS'], 
                            pin_memory=cfg['VAE']['PIN'],
                            num_workers=cfg['VAE']['CPU'],
                            shuffle=False)

    model = InstantaneousProcess(input_dim=cfg['VAE']['INPUT_DIM'],
                               z_dim=cfg['VAE']['LATENT_DIM'], 
                               z_dim_fix=cfg['VAE']['LATENT_DIM_FIX'],
                               z_dim_change=cfg['VAE']['LATENT_DIM_CHANGE'],
                               lag=cfg['VAE']['LAG'],
                               nclass=cfg['VAE']['NCLASS'],
                               hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                               embedding_dim=cfg['VAE']['EMBED_DIM'],
                               lr=cfg['VAE']['LR'],
                               beta=cfg['VAE']['BETA'],
                               gamma=cfg['VAE']['GAMMA'],
                               theta=cfg['VAE']['THETA'],
                               decoder_dist=cfg['VAE']['DEC']['DIST'],
                               correlation=cfg['MCC']['CORR'],
                               enable_flexible_sparsity=cfg['VAE']['FLEXIBLE_SPARTSITY']['ENABLE'],
                               w_hist=cfg['VAE']['FLEXIBLE_SPARTSITY']['HIST'] if cfg['VAE']['FLEXIBLE_SPARTSITY']['ENABLE'] else None,
                               w_inst=cfg['VAE']['FLEXIBLE_SPARTSITY']['INST'] if cfg['VAE']['FLEXIBLE_SPARTSITY']['ENABLE'] else None,
                               z_dim_list=cfg['VAE']['Z_DIM_LIST'],
                               n_mea=cfg['VAE']['N_MEA'],
                               optimizer_type=args.optimizer,
                               )

    log_dir = os.path.join(cfg["LOG"], current_user, args.exp)

    checkpoint_callback = ModelCheckpoint(monitor='val_mcc', 
                                          save_top_k=1, 
                                          mode='max')

    early_stop_callback = EarlyStopping(monitor="val_mcc", 
                                        min_delta=0.00, 
                                        patience=10, 
                                        verbose=False, 
                                        mode="max")

    logger = False
    if not args.disable_wandb:
        wandb_cfg = cfg.get('WANDB', {}) if isinstance(cfg.get('WANDB'), dict) else {}
        base_project = wandb_cfg.get('PROJ_NAME')
        wandb_entity = wandb_cfg.get('ENTITY')
        wandb_mode = wandb_cfg.get('MODE')
        wandb_host = wandb_cfg.get('HOST')

        if wandb_mode:
            os.environ.setdefault("WANDB_MODE", wandb_mode)
        if wandb_host:
            os.environ.setdefault("WANDB_BASE_URL", wandb_host)

        wandb_kwargs = {"project": base_project, "name": f'1_4_{args.seed}_{args.data}'}
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity

        attempt_kwargs = []
        attempt_kwargs.append(dict(wandb_kwargs))
        if "entity" in wandb_kwargs:
            no_entity_kwargs = dict(wandb_kwargs)
            no_entity_kwargs.pop("entity", None)
            attempt_kwargs.append(no_entity_kwargs)
        if base_project:
            auto_project_kwargs = dict(wandb_kwargs)
            auto_project_kwargs.pop("entity", None)
            auto_project_kwargs["project"] = f"{base_project}-auto-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            attempt_kwargs.append(auto_project_kwargs)

        last_exc = None
        for candidate_kwargs in attempt_kwargs:
            if not candidate_kwargs.get("project"):
                continue
            try:
                logger = WandbLogger(**candidate_kwargs)
                _ = logger.experiment
                print(f"[INFO] WandB logging enabled with {candidate_kwargs}.")
                last_exc = None
                break
            except Exception as wandb_exc:
                last_exc = wandb_exc
                print(f"[WARN] WandB init failed with {candidate_kwargs} ({wandb_exc}).")
                try:
                    import wandb as _wandb
                    _wandb.finish()
                except Exception:
                    pass
                logger = False
        if last_exc is not None and logger is False:
            print("[WARN] Falling back to training without WandB. "
                  "Update WandB permissions or provide working ENTITY/PROJECT to enable logging.")

    # trainer = pl.Trainer(default_root_dir=log_dir,
    #                      #accelerator="auto",
    #                      val_check_interval = cfg['MCC']['FREQ'],
    #                      max_epochs=cfg['VAE']['EPOCHS'],
    #                      callbacks=[checkpoint_callback],
    #                      logger=logger,
    #                      #strategy='ddp_find_unused_parameters_true'
    #                      )

    trainer = pl.Trainer(default_root_dir=log_dir,
                         accelerator='cpu' if cfg['VAE']['GPU'] == 0 else 'gpu',
                         devices=cfg['VAE']['GPU'] if cfg['VAE']['GPU'] != 0 else None,
                         strategy='ddp_find_unused_parameters_true' if cfg['VAE']['GPU'] > 1 else 'auto',
                         val_check_interval = cfg['MCC']['FREQ'],
                         max_epochs=cfg['VAE']['EPOCHS'],
                         callbacks=[checkpoint_callback], #, early_stop_callback],
                         logger=logger)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e',
        '--exp',
        type=str,
        default='instantaneous_stationary_link'
    )

    argparser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=770
    )

    argparser.add_argument(
        '-d',
        '--data',
        type=str,
        default="A"
    )

    argparser.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'annealing'],
        default='adam',
        help='Optimizer type: adam (regular) or annealing (cosine annealing)'
    )
    argparser.add_argument(
        '--disable-wandb',
        action='store_true',
        help='Disable WandB logging (useful if credentials or permissions are unavailable).'
    )

    args = argparser.parse_args()
    main(args)
