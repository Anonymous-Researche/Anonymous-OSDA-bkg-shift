import hydra
from pytorch_lightning.utilities import seed
import wandb
import logging
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from src.utils import log_hyperparams
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# log = logging.getLogger(__name__)
log = logging.getLogger("app")
log.setLevel(logging.DEBUG)

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def train(config: DictConfig):
    
    log.info(f"Fixing the seed to <{config.seed}>")
    seed_everything(int(config.seed))

    log.info(f"Instantiating logger <{config.logger._target_}>")
    logger: WandbLogger = hydra.utils.instantiate(config.logger)
    

    checkpoint_callback = ModelCheckpoint(save_last = True)

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    
    if config.mode=="domain_disc":
        early_stop_callback = EarlyStopping(monitor="pred/performance.validation disc cross_entropy", min_delta=0.0, patience=20, mode='min')
        trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback]
        )
    else:
        early_stop_callback = None,
        trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback]
        )
   
    log.info(f"Instantiating model <{config.models._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.models)
    
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    log.info("Logging hyperparameters!")
    log_hyperparams(config=config, trainer=trainer)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)
    # datamodule.setup()
    # trainer.validate(model=model, dataloaders=datamodule.val_dataloader())

    
    # trainer.model.dataselector(unlabeled_data=datamodule.unlabeled_AL_pool,budget_per_AL_cycle=10, batch_size=datamodule.batch_size)
    # datamodule.unlabeled_target.indices = datamodule.unlabeled_AL_pool.indices
    # import pdb; pdb.set_trace()
    log.info("Finished training!")
    wandb.finish()
