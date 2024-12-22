from environment.environment import MazeGame
from model.model import DqnAgent
import time


maze = [
    ['S', '', '.', '.', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', 'G'],
]



if __name__ == "__main__":
    env = MazeGame(maze)
    agent = DqnAgent(env)

    agent.train(total_timesteps=2000)
    print("Train Done!")

    # obs = env.reset()
    # done = False
    # total_reward = 0
    #
    # while not done:
    #     action = agent.predict(obs)
    #     obs, reward, done, _ = env.step(action)
    #     env.render()
    #     time.sleep(0.1)



