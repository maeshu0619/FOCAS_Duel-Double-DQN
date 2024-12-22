import pygame
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete


class MazeGame(Env):
    """
    A simple maze environment using Pygame for visualization.

    Args:
        maze (list): A list of lists representing the maze layout.
            - "S": Start position
            - "G": Goal position
            - Others: Walls
    """

    def __init__(self, maze: list[list]) -> None:
        super(MazeGame, self).__init__()

        self.maze = np.array(maze)

        self.start_position = np.array(np.where(self.maze == "S")).flatten()

        self.end_position = np.array(np.where(self.maze == "G")).flatten()

        self.current_position = self.start_position.copy()

        self.rows, self.cols = self.maze.shape

        self.action_space = Discrete(4)

        self.observation_space = Box(
            low=0, high=max(self.rows, self.cols), shape=(2,), dtype=np.float32
        )

        self.cell = 100

        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.rows * self.cell, self.cols * self.cell)
        )

    def _is_valid_position(self, new_position: np.ndarray) -> bool:
        """
        Checks if a new position is within the maze boundaries.

        Args:
            new_position (np.ndarray): The new position to check.

        Returns:
            bool: True if the position is valid, False otherwise.
        """

        row, col = new_position

        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False

        return True

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Performs a step in the environment based on the provided action.

        Args:
            action (int): The action to take (0: Up, 1: Down, 2: Left, 3: Right).

        Returns:
            tuple: A tuple containing the new observation, reward, done flag,
                and additional information (empty dict in this case).
        """

        new_position = self.current_position.copy()

        if action == 0:
            new_position[0] -= 1
        elif action == 1:
            new_position[0] += 1
        elif action == 2:
            new_position[1] -= 1
        elif action == 3:
            new_position[1] += 1

        if self._is_valid_position(new_position):
            self.current_position = new_position

        if np.array_equal(self.current_position, self.end_position):
            reward = 1
            done = True

        else:
            reward = 0
            done = False

        return self.current_position.astype(np.float32), reward, done, {}

    def reset(self) -> np.ndarray:
        """
        Resets the environment to the starting position.

        Returns:
            np.ndarray: The initial observation (position).
        """

        self.current_position = self.start_position.copy()
        return self.current_position.astype(np.float32)

    def render(self) -> None:
        """
        Renders the current state of the maze environment.
        """

        self.screen.fill((0, 0, 0))

        for row in range(self.rows):
            for col in range(self.cols):
                cell_left = col * self.cell
                cell_top = row * self.cell

                if self.maze[row, col] == "S":
                    color = (92, 184, 92)
                elif self.maze[row, col] == "G":
                    color = (66, 139, 202)
                else:
                    color = (250, 250, 250)

                pygame.draw.rect(
                    self.screen, color, (cell_left, cell_top, self.cell, self.cell)
                )
                pygame.draw.rect(
                    self.screen,
                    (70, 73, 80),
                    (cell_left, cell_top, self.cell, self.cell),
                    1,
                )

        agent_left = int(self.current_position[1] * self.cell + self.cell // 2)
        agent_top = int(self.current_position[0] * self.cell + self.cell // 2)
        pygame.draw.circle(
            self.screen, (217, 83, 79), (agent_left, agent_top), self.cell // 3
        )

        pygame.display.update()
