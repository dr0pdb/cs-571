from Heuristic import Heuristic

from State import State
from output_util import output_result

import sys

import random

visited_state = {}
state_parent = {}


class hill_climbing:
    def __init__(self, intial_state, goal_state):
        self.initial_configuration = intial_state
        self.final_configuration = goal_state
        self.state_explored = 0
        self.admissible = True

    def solve_eight_puzzle(self, current_state, heuristic_choice,depth = 0):
        stack = [current_state]
        while len(stack) != 0 :
            print(self.state_explored, current_state, depth)
            current_state = stack.pop()

            if current_state == self.final_configuration: # We got the result!
                self.state_explored += 1
                print("Goal Achieved....")
                out = output_result(self.initial_configuration, "hill_climbing_result.txt", state_parent,
                                    self.state_explored)
                out.write_output_path(current_state, self.admissible)
                return 0
            else:
                self.state_explored += 1
                visited_state[current_state] = 1
                state = State(current_state, Heuristic(self.final_configuration).getHeuristicEstimation(
                    current_state, heuristic_choice
                ))
                neighbours = state.getAllSuccessor(heuristic_choice, self.final_configuration) # Get the successors.
                local_maxima = True
                options = []
                for neighbour in neighbours:
                    if neighbour.hvalue > state.hvalue or neighbour.puzzleState in visited_state:
                        continue
                    else:
                        local_maxima = False
                        options.append(neighbour)

                if local_maxima:
                    print("Stuck in local maxima")
                else:
                    sz = len(options)
                    idx = random.randrange(0, sz)
                    state_parent[options[idx].puzzleState] = current_state
                    stack.append(options[idx].puzzleState)

        return 1
