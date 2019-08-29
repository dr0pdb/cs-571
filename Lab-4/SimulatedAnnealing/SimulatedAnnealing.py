import random
from math import exp

from Heuristic import Heuristic

from State import State
from output_util import output_result
import sys

# Keeps track of visited states
state_visited = {}

# Keeps track of parent state
parent_state = {}

class simulated_annealing:
    def __init__(self, intial_state, goal_state, temperature):
        self.temperature = temperature
        self.max_temperature = temperature
        self.initial_configuration = intial_state
        self.final_configuration = goal_state
        self.step = 0.0000001
        self.x = -temperature / 5
        self.state_explored = 0
        self.admissible = True
        sys.setrecursionlimit(181440)

    def energy_difference(self, current, new):
        return -1 * (current.hvalue - new.hvalue)

    def cooling_temperature(self, choice):
        return {
            1: self.linear_strategy(),
            2: self.random_strategy(),
            3: self.negative_exponential(),
            4: self.positive_exponential(),
        }[choice]

    def solve_eight_puzzle(self, current_state, heuristic_choice, cooling_choice):
        stack = [current_state]
        while len(stack) != 0:
            current_state = stack.pop()
            print(self.state_explored , current_state)
            if current_state == self.final_configuration:
                self.state_explored += 1
                out = output_result(self.initial_configuration, "simulated_annealing_result.txt", parent_state,
                                    self.state_explored)
                out.write_output_path(current_state, self.admissible)
                print("Goal Achieved....")
                return 0
            elif current_state in state_visited:
                continue
            else:
                self.cooling_temperature(cooling_choice)
                self.state_explored += 1
                state_visited[current_state] = 1
                state = State(current_state, Heuristic(self.final_configuration).getHeuristicEstimation(
                    current_state, heuristic_choice
                ))
                neighbours = state.getAllSuccessor(heuristic_choice)
                neighbours.sort()
                idx, cur = 0, 0
                sz = len(neighbours)
                li = []
                mark = [0]*sz
                cnt = 0
                while cnt < len(neighbours):
                    h = Heuristic(self.final_configuration).getHeuristicEstimation(neighbours[cur].puzzleState, heuristic_choice)
                    if(state.hvalue > h + 1): # Monotonicity implies admissibility
                        self.admissible = False
                    
                    e = self.energy_difference(state, neighbours[cur])
                    if mark[cur] == 1 :
                        cur = (cur + 1) % sz
                        continue
                    if neighbours[current].puzzleState in state_visited :
                        mark[current] = 1
                        count += 1
                    elif e <= 0:
                        mark[current] = 1
                        count += 1
                        parent_state[neighbours[current].puzzleState] = current_state
                        l.append(neighbours[current].puzzleState)
                    elif exp(-e / self.temperature) < random.uniform(0, 1):
                        mark[current] = 1
                        count += 1
                        parent_state[neighbours[current].puzzleState] = current_state
                        l.append(neighbours[current].puzzleState)
                    current = (current+1)%size
                l.reverse()
                stack.extend(l)

        return 1

    def linear_strategy(self):
        self.temperature = abs(self.max_temperature + self.x)
        self.x += self.step

    def random_strategy(self):
        self.temperature = random.uniform(0, 1) * abs(self.max_temperature + self.x)
        self.x += self.step

    def negative_exponential(self):
        self.temperature = exp(-1 * self.x) * self.max_temperature
        self.x += self.step

    def positive_exponential(self):
        self.temperature = exp(self.x) * self.max_temperature
        self.x += self.step
