from SimulatedAnnealing import simulated_annealing
import Heuristic
from datetime import datetime
from Utils import *

def write_to_file(file, heuristic_choice, start_state, goal_state, cooling_function, temp):
    file.write("Heuristic Chosen : ")
    if heuristic_choice == 1:
        file.write("number of tiles displaced from their destined position \n")
    elif heuristic_choice == 2:
        file.write("Total Manhatton distance \n")
    else:
        file.write("Total Manhatton distance * number of tiles displaced from their destined position\n")
    file.write("Start State : \n")
    file.write(start_state[:3] + "\n" + start_state[3:6] + "\n" + start_state[6:] + "\n")
    file.write("Goal State : \n")
    file.write(goal_state[:3] + "\n" + goal_state[3:6] + "\n" + goal_state[6:] + "\n")
    if cooling_function == 1:
        file.write("Cooling Function : Linear Function \n")
    elif cooling_function == 2:
        file.write("Cooling Function : Random Strategy \n")
    elif cooling_function == 3:
        file.write("Cooling Function : Negative exponential Function \n")
    else:
        file.write("Cooling Function : Positive exponential function \n")
    file.write("TMAX : {}\n".format(str(temp)))
    file.close()


def start(startState, goalState, file):
    print("Enter Choice depending on Heuristic :")
    print("1.h1(n) = number of tiles displaced from their destined position")
    print("2.h​2 ​(n)= Total Manhatton distance")
    print("3.h ​3 ​(n)= h ​1 ​(n) * h ​2 ​(n)")

    HeuristicChoice = int(input("Waiting for Choice: "))
    if HeuristicChoice == 4:
    	HeuristicChoice = 3

    is_tile_included = int(input("Enter 1 if consider tile as another tile else 0: "))
    if is_tile_included:
        Heuristic.isTileInclude = True
    print("Enter choice depending on cooling function:\n 1.Linear Function \n 2.Random Strategy \n"
          " 3.Negative exponential Function \n 4.Positive exponential function\n")
    cooling_function_choice = int(input("Waiting for Choice: "))
    initial_temperature = int(input("Enter TMAX: "))

    write_to_file(file, HeuristicChoice, startState, goalState, cooling_function_choice, initial_temperature)

    start_time = datetime.now()
    puzzle_solver = simulated_annealing(startState, goalState, initial_temperature)
    status = puzzle_solver.solve_eight_puzzle(startState, HeuristicChoice, cooling_function_choice)
    file = open("simulated_annealing_result.txt", "a")
    if status == 1:
        file.write("Search Status : Failed\n")
    file.write("Time Taken : {} ".format(str(datetime.now() - start_time)))
    file.close()

# Starting point of the program.
if __name__ == '__main__':
    startState = ""
    goalState = ""

    file = open("simulated_annealing_result.txt", "w+")

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
