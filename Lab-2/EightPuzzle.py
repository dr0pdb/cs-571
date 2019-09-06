import time

from AStarAlgorithm import *
from PuzzleState import *


def start(startState, goalState):
    goalState = State(goalState, 0)

    # show menu
    print("Enter choice depending on Heuristic: ")
    print("1. Check monotone for h2(n) = number of tiles displaced from their destined position")
    print("2. Check monotone for h3(n) = sum of Manhattan distance of each tiles from the goal position.")
    print("3. Verify that all the states visited by better heuristics is also visited by inferior heuristics")
    print("4. Check monotone for h2(n) while considering the empty tile's cost")
    print("5. Check monotone for h3(n) while considering the empty tile's cost")

    # retrieve choice.
    HeuristicChoice = int(input("Waiting for Choice: "))
    if HeuristicChoice > 5 or HeuristicChoice <= 0:
        print("Invalid Choice.......\n")
        return
    else:
        if HeuristicChoice == 1:
            start = time.process_time()
            hInitial = Heuristic(goalState).getHeuristicEstimation(startState, 2)
            initialState = State(startState, hInitial, hInitial)
            EightPuzzleProblem(initialState, goalState).solveEightPuzzle(2)
            print("Time taken by the program in seconds: ")
            print(time.process_time() - start)
        elif HeuristicChoice == 2:
            hInitial = Heuristic(goalState).getHeuristicEstimation(startState, 3)
            initialState = State(startState, hInitial, hInitial)
            EightPuzzleProblem(initialState, goalState).solveEightPuzzle(3)
        elif HeuristicChoice == 3:
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
        elif HeuristicChoice == 4:
            hInitial = Heuristic(goalState, True).getHeuristicEstimation(startState, 2)
            initialState = State(startState, hInitial, hInitial)
            EightPuzzleProblem(initialState, goalState).solveEightPuzzle(2)
        else:
            hInitial = Heuristic(goalState, True).getHeuristicEstimation(startState, 3)
            initialState = State(startState, hInitial, hInitial)
            EightPuzzleProblem(initialState, goalState).solveEightPuzzle(3)

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
