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
    if 8 < HeuristicChoice or HeuristicChoice < 0:
        print("Invalid Choice.......")
        return
    if HeuristicChoice == 8:
        AStartAlgorithm.isTileInclude = True
        print("1.h(n)=0\n2.TilesDisplaced Heuristics\n3.ManhattanHeuristic\n4.Overestimated Heuristic")
        Choice = int(input("Waiting for Choice"))
        hInitial = Heuristic(goalState).getHeuristicEstimation(startState, Choice)
        initialState = State(startState, hInitial, hInitial)
        eightPuzzle = EightPuzzle(initialState, goalState)
        eightPuzzle.solveEightPuzzle(Choice)

        if eightPuzzle.isMonotone:
            print("This Heuristic is Monotone")
        else:
            print("This Heuristic is not Monotone")

    elif HeuristicChoice == 7:
        print("1.h(n)=0\n2.TilesDisplaced Heuristics\n3.ManhattanHeuristic\n4.Overestimated Heuristic")
        # For Manhatten
        Choice = int(input("Waiting for Choice"))
        hInitial = Heuristic(goalState).getHeuristicEstimation(startState, Choice)
        initialState = State(startState, hInitial, hInitial)
        eightPuzzle = EightPuzzle(initialState, goalState)
        eightPuzzle.solveEightPuzzle(Choice)
        if eightPuzzle.isMonotone:
            print("This Heuristic is Monotone")
        else:
            print("This Heuristic is not Monotone")
    elif HeuristicChoice == 6:
        # For Manhatten
        hInitial = Heuristic(goalState).getHeuristicEstimation(startState, 3)
        initialState = State(startState, hInitial, hInitial)
        closedList1, temp = EightPuzzle(initialState, goalState).solveEightPuzzle(3)

        # For TilesDisplaced
        hInitial = Heuristic(goalState).getHeuristicEstimation(startState, 2)
        initialState = State(startState, hInitial, hInitial)
        closedList2, temp = EightPuzzle(initialState, goalState).solveEightPuzzle(2)

        if set(closedList2.keys()).issuperset(set(closedList1.keys())):
            print("Nodes explored by Less Powerful heuristic is superset of more Powerful heurisitc")
        else:
            print(set(closedList1.keys()).difference(set(closedList2.keys())))
            print("Something is errorneous ..")

    elif HeuristicChoice == 5:
        tabularList = []
        header = ["Heuristic Name", "StatesExplored", "OptimalState", "OptimalCost", "ExecutionTime"]
        tabularList.append(header)
        for i in range(1, 5):
            start = time.time()
            hInitial = Heuristic(goalState).getHeuristicEstimation(startState, i)
            initialState = State(startState, hInitial, hInitial)
            closedList, temp = EightPuzzle(initialState, goalState).solveEightPuzzle(i)
            end = time.time()
            print("Execution Time : " + str((end - start)))
            print("-" * 50)
            temp.append(str((end - start)))
            tabularList.append(temp)
        print("Comparision in Tabular form")
        t = Texttable()
        t.add_rows(tabularList)
        print(t.draw())
    else:
        hInitial = Heuristic(goalState).getHeuristicEstimation(startState, HeuristicChoice)
        initialState = State(startState, hInitial, hInitial)
        EightPuzzle(initialState, goalState).solveEightPuzzle(HeuristicChoice)


def displayMenu():
    print("Enter Choice depending on Heuristic :")
    print("1.h1(n) = 0\n2.h2(n) = number of tiles displaced from their destined position")
    print("3.h3(n) = sum of Manhattan distance of each tiles from the goal position.")
    print("4.h4(n) = heuristics such that h(n) > h∗(n)")
    print("5.Compare All Heuristic ")
    print("6.Compare Manhattan and TileDisplaced Heuristic")
    print("7.Verify if heuristic is Monotone")
    print("8.Verify if heuristic is Monotone if cost of empty tile is added")


if __name__ == '__main__':
    main()
