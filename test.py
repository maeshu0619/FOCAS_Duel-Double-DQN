from model import DQNAgent
from environment import FoveatedRenderingEnv

# Example configuration
resolution_time = {
    'fovea': 2.0,
    'blend': 1.0,
    'peripheral': 0.5
}

latency_constraints = {
    'high': 50,  # High latency tolerance (ms)
    'medium': 20,  # Medium latency tolerance (ms)
    'low': 10   # Low latency tolerance (ms)
}

transmission_environments = {
    'poor': 20,  # Poor network condition (R_tx)
    'average': 50,  # Average network condition
    'good': 80  # Good network condition
}

def test(latency, transmission):
    env = FoveatedRenderingEnv(
        transmission_rate=transmission_environments[transmission],
        latency_constraint=latency_constraints[latency],
        resolution_time=resolution_time
    )
    agent = DQNAgent(env)

    # Load the trained model
    agent.model.load("dqn_foveated_model")

    # Evaluate the agent
    mean_reward, std_reward = agent.evaluate(env)
    print(f"Testing results: Mean Reward = {mean_reward}, Std Reward = {std_reward}")

if __name__ == "__main__":
    # Set constraints
    selected_latency = 'medium'  # Options: 'high', 'medium', 'low'
    selected_transmission = 'average'  # Options: 'poor', 'average', 'good'

    test(selected_latency, selected_transmission)
