from typing import Sequence

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import nn, Tensor

from ConSSL.utils import _MATPLOTLIB_AVAILABLE
from ConSSL.utils.warnings import warn_missing_pkg

if _MATPLOTLIB_AVAILABLE:
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:  # pragma: no cover
    warn_missing_pkg("matplotlib")
    Axes = object
    Figure = object


class ConfusedLogitCallback(Callback):  # pragma: no cover
    """
    Takes the logit predictions of a model and when the probabilities of two classes are very close, the model
    doesn't have high certainty that it should pick one vs the other class.

    This callback shows how the input would have to change to swing the model from one label prediction
    to the other.

    In this case, the network predicts a 5... but gives almost equal probability to an 8.
    The images show what about the original 5 would have to change to make it more like a 5 or more like an 8.

    For each confused logit the confused images are generated by taking the gradient from a logit wrt an input
    for the top two closest logits.

    Example::

        from ConSSL.callbacks.vision import ConfusedLogitCallback
        trainer = Trainer(callbacks=[ConfusedLogitCallback()])


    .. note:: Whenever called, this model will look for ``self.last_batch`` and ``self.last_logits``
              in the LightningModule.

    .. note:: This callback supports tensorboard only right now.

    Authored by:

        - Alfredo Canziani
    """

    def __init__(
        self,
        top_k: int,
        projection_factor: int = 3,
        min_logit_value: float = 5.0,
        logging_batch_interval: int = 20,
        max_logit_difference: float = 0.1
    ):
        """
        Args:
            top_k: How many "offending" images we should plot
            projection_factor: How much to multiply the input image to make it look more like this logit label
            min_logit_value: Only consider logit values above this threshold
            logging_batch_interval: How frequently to inspect/potentially plot something
            max_logit_difference: When the top 2 logits are within this threshold we consider them confused
        """
        super().__init__()
        self.top_k = top_k
        self.projection_factor = projection_factor
        self.max_logit_difference = max_logit_difference
        self.logging_batch_interval = logging_batch_interval
        self.min_logit_value = min_logit_value

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # show images only every 20 batches
        if (trainer.batch_idx + 1) % self.logging_batch_interval != 0:  # type: ignore[attr-defined]
            return

        # pick the last batch and logits
        x, y = batch
        try:
            logits = pl_module.last_logits
        except AttributeError as err:
            m = """please track the last_logits in the training_step like so:
                def training_step(...):
                    self.last_logits = your_logits
            """
            raise AttributeError(m) from err

        # only check when it has opinions (ie: the logit > 5)
        if logits.max() > self.min_logit_value:  # type: ignore[operator]
            # pick the top two confused probs
            (values, idxs) = torch.topk(logits, k=2, dim=1)  # type: ignore[arg-type]

            # care about only the ones that are at most eps close to each other
            eps = self.max_logit_difference
            mask = (values[:, 0] - values[:, 1]).abs() < eps

            if mask.sum() > 0:
                # pull out the ones we care about
                confusing_x = x[mask, ...]
                confusing_y = y[mask]

                mask_idxs = idxs[mask]

                pl_module.eval()
                self._plot(confusing_x, confusing_y, trainer, pl_module, mask_idxs)
                pl_module.train()

    def _plot(
        self,
        confusing_x: Tensor,
        confusing_y: Tensor,
        trainer: Trainer,
        model: LightningModule,
        mask_idxs: Tensor,
    ) -> None:
        if not _MATPLOTLIB_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                'You want to use `matplotlib` which is not installed yet, install it with `pip install matplotlib`.'
            )

        confusing_x = confusing_x[:self.top_k]
        confusing_y = confusing_y[:self.top_k]

        x_param_a = nn.Parameter(confusing_x)
        x_param_b = nn.Parameter(confusing_x)

        batch_size, c, w, h = confusing_x.size()
        for logit_i, x_param in enumerate((x_param_a, x_param_b)):
            x_param = x_param.to(model.device)  # type: ignore[assignment]
            logits = model(x_param.view(batch_size, -1))
            logits[:, mask_idxs[:, logit_i]].sum().backward()

        # reshape grads
        grad_a = x_param_a.grad.view(batch_size, w, h)
        grad_b = x_param_b.grad.view(batch_size, w, h)

        for img_i in range(len(confusing_x)):
            x = confusing_x[img_i].squeeze(0).cpu()
            y = confusing_y[img_i].cpu()
            ga = grad_a[img_i].cpu()
            gb = grad_b[img_i].cpu()

            mask_idx = mask_idxs[img_i].cpu()

            fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
            self.__draw_sample(fig, axarr, 0, 0, x, f'True: {y}')
            self.__draw_sample(fig, axarr, 0, 1, ga, f'd{mask_idx[0]}-logit/dx')
            self.__draw_sample(fig, axarr, 0, 2, gb, f'd{mask_idx[1]}-logit/dx')
            self.__draw_sample(fig, axarr, 1, 1, ga * 2 + x, f'd{mask_idx[0]}-logit/dx')
            self.__draw_sample(fig, axarr, 1, 2, gb * 2 + x, f'd{mask_idx[1]}-logit/dx')

            trainer.logger.experiment.add_figure('confusing_imgs', fig, global_step=trainer.global_step)

    @staticmethod
    def __draw_sample(fig: Figure, axarr: Axes, row_idx: int, col_idx: int, img: Tensor, title: str) -> None:
        im = axarr[row_idx, col_idx].imshow(img)
        fig.colorbar(im, ax=axarr[row_idx, col_idx])
        axarr[row_idx, col_idx].set_title(title, fontsize=20)
