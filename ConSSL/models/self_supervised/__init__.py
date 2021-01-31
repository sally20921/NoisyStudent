"""
These models have been pre-trained using self-supervised learning. The models can also be used without pre-training
and overwritten for your own research.

Here's an example for using these as pretrained models.

.. code-block ::

    from ConSSL.models.self_supervised import CPCV2

    images = get_imagenet_batch()

    # extract unsupervised representations
    pretrained = CPCV2(pretrained=True)
    representations = pretrained(images)

    # use these in classification or any downstream task
    classifications = classifier(representations)

"""
from ConSSL.models.self_supervised.amdim.amdim_module import AMDIM  # noqa: F401
from ConSSL.models.self_supervised.byol.byol_module import BYOL  # noqa: F401
from ConSSL.models.self_supervised.cpc.cpc_module import CPCV2  # noqa: F401
from ConSSL.models.self_supervised.evaluator import SSLEvaluator  # noqa: F401
from ConSSL.models.self_supervised.moco.moco2_module import MocoV2  # noqa: F401
from ConSSL.models.self_supervised.simclr.simclr_module import SimCLR  # noqa: F401
from ConSSL.models.self_supervised.simsiam.simsiam_module import SimSiam  # noqa: F401
from ConSSL.models.self_supervised.ssl_finetuner import SSLFineTuner  # noqa: F401
from ConSSL.models.self_supervised.swav.swav_module import SwAV  # noqa: F401

__all__ = [
    "AMDIM",
    "BYOL",
    "CPCV2",
    "SSLEvaluator",
    "MocoV2",
    "SimCLR",
    "SimSiam",
    "SSLFineTuner",
    "SwAV",
]
