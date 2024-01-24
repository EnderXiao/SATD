from pytorch_lightning import Trainer
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI

from satd.datamodule.datamodule import CROHMEDatamodule
from satd.lit_satd import LitSATD

import os

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


if __name__ == "__main__":
    cli = LightningCLI(
        LitSATD,
        CROHMEDatamodule,
        save_config_overwrite=True,
        trainer_defaults={"plugins": DDPPlugin(find_unused_parameters=False)},
    )
