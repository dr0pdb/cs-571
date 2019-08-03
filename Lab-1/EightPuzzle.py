import time

from AStarAlgorithm import *
from PuzzleState import *


def start(startState, goalState):
    goalState = State(goalState, 0)

    # show menu
    print("Enter choice depending on Heuristic: ")
    print("1. h1(n) = 0\n2.h2(n) = number of tiles displaced from their destined position")
    print("3. h3(n) = sum of Manhattan distance of each tiles from the goal position.")
    print("4. h4(n) = heuristics such that h(n) > h∗(n)")

    # retrieve choice.
    HeuristicChoice = int(input("Waiting for Choice"))
    if HeuristicChoice > 4 or HeuristicChoice <= 0:
        print("Invalid Choice.......\n")
        return
    else:
        hInitial = Heuristic(goalState).getHeuristicEstimation(startState, HeuristicChoice)
        initialState = State(startState, hInitial, hInitial)
        EightPuzzleProblem(initialState, goalState).solveEightPuzzle(HeuristicChoice)


if __name__ == '__main__':
    startState = ""
    goalState = ""
    with open("A*StartState") as f:
        for line in f:
            line = line.strip()
            line = line.replace(" ", "")
            startState += line

    with open("A*GoalState") as f:
        for line in f:
            line = line.strip()
            line = line.replace(" ", "")
            goalState += line

    startState = replaceTB(startState)
    goalState = replaceTB(goalState)

    # start the program.
    start(startState, goalState)
