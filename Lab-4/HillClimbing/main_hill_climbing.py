from HillClimbing import hill_climbing
from datetime import datetime
from file_utils import *
from Utils import *
from Heuristic import *

def start(startState, goalState, file):
    # Menu
    print("Enter Choice depending on Heuristic :")
    print("1. h1(n) = number of tiles displaced from their destined position")
    print("2. h​2 ​(n)= Total Manhattan distance")
    print("3. h3(n)= 3*h1(n) - 2*h2(n)")

    # Input
    HeuristicChoice = int(input("Waiting for Choice: "))
    write_to_file(file, HeuristicChoice, startState, goalState)
    is_tile_included = int(input("Enter 1 if consider tile as another tile else 0: "))
    if is_tile_included:
        Heuristic.isTileInclude = True

    # Solving the problem.
    start_time = datetime.now()
    puzzle_solver = hill_climbing(startState, goalState)
    status = puzzle_solver.solve_eight_puzzle(startState, HeuristicChoice)
    file = open("hill_climbing_result.txt", "a")
    if status == 1:
        file.write("Search Status : Failed\n")
    file.write("Time Taken : {} ".format(str(datetime.now() - start_time)))
    file.close()    

# The starting point of the program.
if __name__ == '__main__':
    startState = ""
    goalState = ""
    file = open("hill_climbing_result.txt", "w+")

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
    start(startState, goalState, file)
