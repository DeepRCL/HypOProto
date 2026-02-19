from src.models.ProtoPNet import construct_PPNet
from src.models.XProtoNet import construct_XProtoNet
from src.models.Video_XProtoNet import construct_Video_XProtoNet
from src.models.Video_XProtoNet_EchoPrime import construct_Video_XProtoNet_EchoPrime
from src.models.Hyper_Video_XProtoNet_EchoPrime import construct_Hyper_Video_XProtoNet_EchoPrime
from copy import deepcopy
import logging

MODELS = {
    "ProtoPNet": construct_PPNet,
    "XProtoNet": construct_XProtoNet,
    "Video_XProtoNet": construct_Video_XProtoNet,
    "ProtoFPNet_Video_EchoPrime": construct_Video_XProtoNet_EchoPrime,
    "Hyper_ProtoFPNet_Video_EchoPrime": construct_Hyper_Video_XProtoNet_EchoPrime
}


def build(model_config):
    config = deepcopy(model_config)
    _ = config.pop("checkpoint_path")
    if "prototype_shape" in config.keys():
        config["prototype_shape"] = eval(config["prototype_shape"])

    # Build the model
    model_name = config.pop("name")
    model = MODELS[model_name](**config)
    logging.info(f"Model {model_name} is created.")

    return model
