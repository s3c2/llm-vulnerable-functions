import os
import sys
from pathlib import Path
from decouple import config
import wandb
from utils.model_helper import load_cfg


class WandB:
    """ Log results for Weights & Biases """

    def __init__(self, cfg_basepath: str, cfg_filename: str):
        self.cfg = load_cfg(base_dir=cfg_basepath,
                            filename=cfg_filename, as_namespace=False)
        try:
            self.key = config('WANDB_KEY')
        except Exception as e:
            raise Exception("Your WANDB_KEY is not set in your .env file")

    def login_to_wandb(self):
        # Store WandB key in the environment variables
        if self.key is not None:
            wandb.login(key=self.key)
        else:
            print('Not logging info. in WandB')

    def get_logger(self):
        self.login_to_wandb()
        wb_logger = wandb.init(project=self.cfg['wandb']['project'],
                               #    dir=self.wandb_save_dir,
                               #    id=self.id,
                               #    job_type=self.fold_num,
                               #    name=self.run_name,
                               group=self.cfg['wandb']['group'],
                               config=self.cfg,
                               mode=self.cfg['wandb']['mode'],
                               )

        return wb_logger


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None
