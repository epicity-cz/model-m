# models

# model = ENGINE + MODEL DEFINITION

# engine is not cofigurable yet
# you can specify your model definition


from models.seirs import SEIRSModel, SEIRSNetworkModel
from models.extended_network_model import ExtendedNetworkModel, ExtendedDailyNetworkModel, ExtendedSequentialNetworkModel
from models.extended_network_model import TGMNetworkModel
from models.seirs_extended import ExtendedNetworkModel as OldExtendedNetworkModel
from models.agent_based_network_model import SimulationDrivenModel

__all__ = [
    ExtendedNetworkModel,          # our extended version
    SEIRSModel,                    # original seirsplus code
    SEIRSNetworkModel,             # original seirsplus code
    OldExtendedNetworkModel        # abandonded implementation
]

# preferable is ExtendedSequentialNetworkModel
# older implementations ExtendedDailyNetworkModel and ExtendedNetworkModel are abandonded
# !!! current only supported choice is SumulationDrivenModel

model_zoo = {
    "ExtendedNetworkModel": ExtendedNetworkModel,
    "ExtendedDailyNetworkModel": ExtendedDailyNetworkModel,
    "ExtendedSequentialNetworkModel": ExtendedSequentialNetworkModel,
    "OldExtendedNetworkModel": OldExtendedNetworkModel,
    "SEIRSNetworkModel": SEIRSNetworkModel,
    "TGMNetworkModel": TGMNetworkModel,
    "SimulationDrivenModel": SimulationDrivenModel
}
