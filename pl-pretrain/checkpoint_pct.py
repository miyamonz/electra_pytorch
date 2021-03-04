import pytorch_lightning as pl
from pytorch_lightning import _logger as log


class CheckpointPct(pl.Callback):
    def __init__(
        self,
        total_steps,
        ratio,
        ckpt_path,
    ):
        self.total_steps = total_steps
        self.ratio = ratio
        self.ckpt_path = ckpt_path
        self.stopped_epoch = 0

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        prev_pct = (global_step-1) / self.total_steps
        curr_pct = (global_step) / self.total_steps

        pct = None
        for p in self.ratio:
            if prev_pct <= p < curr_pct:
                pct = p

        if 1 <= curr_pct:
            pct = 1
            self.stopped_epoch = epoch
            trainer.should_stop = True

        if pct is None:
            return

        filename = f"{self.total_steps}_{int(pct*100)}%.ckpt"
        ckpt_path = self.ckpt_path / filename
        print(ckpt_path)
        trainer.save_checkpoint(ckpt_path)

    def on_train_end(self, trainer, pl_module):
        if self.stopped_epoch > 0:
            log.info(
                f'Epoch {self.stopped_epoch:05d}: early stopping triggered.')
