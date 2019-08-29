from Heuristic import Heuristic

from State import State
from output_util import output_result

import sys

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
            elif current_state in visited_state: # Already visited so skip.
                continue
            else:
                self.state_explored += 1
                visited_state[current_state] = 1
                state = State(current_state, Heuristic(self.final_configuration).getHeuristicEstimation(
                    current_state, heuristic_choice
                ))
                neighbours = state.getAllSuccessor(heuristic_choice) # Get the successors.
                neighbours.sort() # sort in increasing order of H value.
                neighbours.reverse() # reverse so that we iterate in decreasing order of H value.
                for neighbour in neighbours:
                    h = Heuristic(self.final_configuration).getHeuristicEstimation(neighbour.puzzleState, heuristic_choice)
                    if(state.hvalue > h + 1): # Monotonicity implies admissibility
                        self.admissible = False
                    if neighbour.hvalue > state.hvalue:
                        print("stuck in local maxima") # When the neighbour is worse than the current state.
                    if neighbour.puzzleState in visited_state:
                        continue
                    else:
                        state_parent[neighbour.puzzleState] = current_state
                        stack.append(neighbour.puzzleState)
        return 1
