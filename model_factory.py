import global_setting
from models.resnet import get_resnet_model

def get_model(model_architecture):
    assert model_architecture in global_setting.RESNET_MODELS
    if "resnet50" == model_architecture: model_type = 'resnet50'
    elif "resnet101" == model_architecture: model_type = 'resnet101'
    return get_resnet_model(model_type,
                            pretrained=False,
                            num_classes=2)
