from agent_based_network_model import model_definition, daily_update, testing, change_states, update_plan
from engine_s import EngineS
from model import create_custom_model

# 3. model class
SimulationDrivenModel = create_custom_model("SimulationDrivenModel",
                                            **model_definition,
                                            member_functions={
                                                "daily_update": daily_update,
                                                "testing": testing,
                                                "change_states": change_states,
                                                "update_plan": update_plan
                                            },
                                            engine=EngineS)
