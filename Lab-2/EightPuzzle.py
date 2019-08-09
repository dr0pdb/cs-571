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
    print("5. Verify that all the states visited by better heuristics is also visited by inferior heuristics")

    # retrieve choice.
    HeuristicChoice = int(input("Waiting for Choice: "))
    if HeuristicChoice > 5 or HeuristicChoice <= 0:
        print("Invalid Choice.......\n")
        return
    else:
        if HeuristicChoice <= 4:
            start = time.process_time()
            hInitial = Heuristic(goalState).getHeuristicEstimation(startState, HeuristicChoice)
            initialState = State(startState, hInitial, hInitial)
            EightPuzzleProblem(initialState, goalState).solveEightPuzzle(HeuristicChoice)
            print("Time taken by the program in seconds: ")
            print(time.process_time() - start)
        elif HeuristicChoice == 5:
            heuristics_states = []
            for x in range(1,4):
                hInitial = Heuristic(goalState).getHeuristicEstimation(startState, x)
                initialState = State(startState, hInitial, hInitial)
                heuristics_states.append(EightPuzzleProblem(initialState, goalState).solveEightPuzzle(x))
            print("States visited by h1(n) = 0")
            for elem in heuristics_states[0]:
                if elem:
                    print(elem)
            
            print("\n\nStates visited by h2(n) = number of tiles displaced")
            for elem in heuristics_states[1]:
                if elem:
                    print(elem)
            
            print("\n\nStates visited by h3(n) = sum of Manhattan distance")
            for elem in heuristics_states[2]:
                if elem:
                   print(elem)

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
