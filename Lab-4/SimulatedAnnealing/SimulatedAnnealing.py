import random
from math import exp

from Heuristic import Heuristic

from week5.EightPuzzleState import State
from week5.output_util import output_result
import sys

visited_state = {}
state_parent = {}


class simulated_annealing:
    def __init__(self, intial_state, goal_state, temperature):
        self.initial_configuration = intial_state
        self.final_configuration = goal_state
        self.temperature = temperature
        self.temperature_max = temperature
        self.x = -temperature / 5
        self.step = 0.0000001
        self.state_explored = 0
        sys.setrecursionlimit(181440)

    def energy_difference(self, current, new):
        return -1 * (current.hvalue - new.hvalue)

    def cooling_temperature(self, choice):
        return {
            1: self.linear_strategy(),
            2: self.random_strategy(),
            3: self.neg_exponential(),
            4: self.pos_exponential(),
        }[choice]

    def solve_eight_puzzle(self, current_state, heuristic_choice, cooling_choice):
        stack = [current_state]
        while len(stack) != 0:
            current_state = stack.pop()
            print(self.state_explored , current_state)
            if current_state == self.final_configuration:
                self.state_explored += 1
                out = output_result(self.initial_configuration, "simulated_annealing_result.txt", state_parent,
                                    self.state_explored)
                out.write_output_path(current_state)
                print("Goal Achieved....")
                return 0
            elif current_state in visited_state:
                continue
            else:
                self.cooling_temperature(cooling_choice)
                self.state_explored += 1
                visited_state[current_state] = 1
                state = State(current_state, Heuristic(self.final_configuration).getHeuristicEstimation(
                    current_state, heuristic_choice
                ))
                neighbours = state.getAllSuccessor(heuristic_choice)
                neighbours.sort()
                # neighbours.reverse()
                idx, cur = 0, 0
                sz = len(neighbours)
                li = []
                mark = [0]*sz
                cnt = 0
                while cnt < len(neighbours):
                    e = self.energy_difference(state, neighbours[cur])
                    if mark[cur] == 1 :
                        cur = (cur + 1) % sz
                        continue
                    if neighbours[cur].puzzleState in visited_state :
                        mark[cur] = 1
                        cnt += 1
                        # neighbours[cur], neighbours[cur] = neighbours[cur], neighbours[cur + idx]
                        # cur += 1
                    elif e <= 0:
                        mark[cur] = 1
                        cnt += 1
                        state_parent[neighbours[cur].puzzleState] = current_state
                        # success_indicator = self.solve_eight_puzzle(neighbours[cur + idx].puzzleState, heuristic_choice,
                        #                                             cooling_choice)
                        li.append(neighbours[cur].puzzleState)
                        # cur += 1
                    elif exp(-e / self.temperature) < random.uniform(0, 1):
                        mark[cur] = 1
                        cnt += 1
                        state_parent[neighbours[cur].puzzleState] = current_state
                        # success_indicator = self.solve_eight_puzzle(neighbours[cur + idx].puzzleState, heuristic_choice,
                        #                                             cooling_choice)
                        li.append(neighbours[cur].puzzleState)
                        # neighbours[cur + idx], neighbours[cur] = neighbours[cur], neighbours[cur + idx]
                        # idx = cur
                        # cur += 1
                    cur = (cur+1)%sz
                li.reverse()
                stack.extend(li)

        # self.state_explored += 1
        # self.cooling_temperature(cooling_choice)
        # if current_state == self.final_configuration:
        #     out = output_result(self.initial_configuration, "simulated_annealing_result.txt", state_parent,
        #                         self.state_explored)
        #     out.write_output_path(current_state)
        #     print("Goal Achieved....")
        #     return 1
        # else:
        #     visited_state[current_state] = 1
        #     state = State(current_state, Heuristic(self.final_configuration).getHeuristicEstimation(
        #         current_state, heuristic_choice
        #     ))
        #     neighbours = state.getAllSuccessor(heuristic_choice)
        #     neighbours.sort()
        #     success_indicator = 0
        #     idx, cur = 0, 0
        #     sz = len(neighbours)
        #     while cur < len(neighbours):
        #         e = self.energy_difference(state, neighbours[cur + idx])
        #         if neighbours[cur + idx].puzzleState in visited_state:
        #             neighbours[cur + idx], neighbours[cur] = neighbours[cur], neighbours[cur + idx]
        #             cur += 1
        #         elif e <= 0:
        #             state_parent[neighbours[cur + idx].puzzleState] = current_state
        #             success_indicator = self.solve_eight_puzzle(neighbours[cur + idx].puzzleState, heuristic_choice,
        #                                                         cooling_choice)
        #             cur += 1
        #         elif exp(-e / self.temperature) < random.uniform(0, 1):
        #             state_parent[neighbours[cur + idx].puzzleState] = current_state
        #             success_indicator = self.solve_eight_puzzle(neighbours[cur + idx].puzzleState, heuristic_choice,
        #                                                         cooling_choice)
        #             neighbours[cur + idx], neighbours[cur] = neighbours[cur], neighbours[cur + idx]
        #             # idx = cur
        #             cur += 1
        #         if success_indicator == 1:
        #             return success_indicator
        #         if sz == cur:
        #             break
        #         idx = (idx + 1) % (sz - cur)
        return 1

    def linear_strategy(self):
        self.temperature = abs(self.temperature_max + self.x)
        self.x += self.step

    def random_strategy(self):
        self.temperature = random.uniform(0, 1) * abs(self.temperature_max + self.x)
        self.x += self.step

    def neg_exponential(self):
        self.temperature = exp(-1 * self.x) * self.temperature_max
        self.x += self.step

    def pos_exponential(self):
        self.temperature = exp(self.x) * self.temperature_max
        self.x += self.step
