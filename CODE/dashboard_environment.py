"""Evaluation-only hospital-set changes for the dashboard's shared DQN models."""
import copy

SHARED_ARCHITECTURES = ("hospital_shared_v4", "hospital_shared_v5", "hospital_shared_v6")


def resize_inference_agent(source, hospital_count, config):
    """Reuse every learned scorer weight; never resize a fixed output layer."""
    from baseline import state_dimension
    from dqn_agent import DQNAmbulanceAgent

    if hospital_count < 1:
        raise ValueError("평가할 병원이 없습니다.")
    if source.action_dim == hospital_count:
        return source
    if source.architecture not in SHARED_ARCHITECTURES:
        raise ValueError("이전 v3 모델은 병원 수가 고정되어 있습니다. v4 이상 모델이나 다른 병원 선택 정책을 사용하세요.")
    agent = DQNAmbulanceAgent(state_dimension(hospital_count, config.observation_version), hospital_count,
                              architecture=source.architecture, learning_config=source.learning_config)
    agent.model.load_state_dict(source.model.state_dict(), strict=True)
    agent.target_model.load_state_dict(source.target_model.state_dict(), strict=True)
    agent.metadata = copy.deepcopy(source.metadata)
    agent.loaded_episode = source.loaded_episode
    agent.counter_history_complete = source.counter_history_complete
    agent.learning_config_explicit = source.learning_config_explicit
    for name in source.COUNTERS:
        setattr(agent, name, getattr(source, name))
    agent.epsilon = 0.
    agent.model.eval()
    agent.target_model.eval()
    return agent
