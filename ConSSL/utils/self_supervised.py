from torch.nn import Module

from ConSSL.models.self_supervised import resnets
from ConSSL.utils.semi_supervised import Identity


def torchvision_ssl_encoder(
    name: str,
    pretrained: bool = False,
    return_all_feature_maps: bool = False,
) -> Module:
    pretrained_model = getattr(resnets, name)(pretrained=pretrained, return_all_feature_maps=return_all_feature_maps)
    pretrained_model.fc = Identity()
    return pretrained_model
