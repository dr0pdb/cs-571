import time

from texttable import Texttable

from AStartAlgorithm import Heuristic, EightPuzzle
import AStartAlgorithm
from PuzzleState import State


def main():
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

    startState = startState.replace("T", "")
    startState = startState.replace("B", "0")

    goalState = goalState.replace("T", "")
    goalState = goalState.replace("B", "0")

    goalState = State(goalState, 0)

    displayMenu()
    HeuristicChoice = int(input("Waiting for Choice"))
    if HeuristicChoice > 4 or HeuristicChoice < 0:
        print("Invalid Choice.......")
        return
    else:
        hInitial = Heuristic(goalState).getHeuristicEstimation(startState, HeuristicChoice)
        initialState = State(startState, hInitial, hInitial)
        EightPuzzle(initialState, goalState).solveEightPuzzle(HeuristicChoice)


def displayMenu():
    print("Enter Choice depending on Heuristic :")
    print("1.h1(n) = 0\n2.h2(n) = number of tiles displaced from their destined position")
    print("3.h3(n) = sum of Manhattan distance of each tiles from the goal position.")
    print("4.h4(n) = heuristics such that h(n) > h∗(n)")

if __name__ == '__main__':
    main()
