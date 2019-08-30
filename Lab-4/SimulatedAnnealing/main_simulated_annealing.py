from SimulatedAnnealing import simulated_annealing
import Heuristic
from datetime import datetime
from Utils import *
from Heuristic import *

# Writes output to the file
def write_to_file(file, heuristic_choice, start_state, goal_state, cooling_function, temp):
    file.write("Chosen Heuristic: ")
    if heuristic_choice == 1:
        file.write("Number of tiles displaced from their final position: \n")
    elif heuristic_choice == 2:
        file.write("Total Manhattan distance: \n")
    else:
        file.write("Total Manhattan distance * Number of tiles displaced from their final position:\n")
    file.write("Start State : \n")
    file.write(start_state[:3] + "\n" + start_state[3:6] + "\n" + start_state[6:] + "\n")
    file.write("Goal State : \n")
    file.write(goal_state[:3] + "\n" + goal_state[3:6] + "\n" + goal_state[6:] + "\n")
    if cooling_function == 1:
        file.write("Cooling Function : Linear Function \n")
    elif cooling_function == 2:
        file.write("Cooling Function : Random Strategy \n")
    elif cooling_function == 3:
        file.write("Cooling Function : Negative Exponential Function \n")
    else:
        file.write("Cooling Function : Positive Exponential function \n")
    file.write("TMAX : {}\n".format(str(temp)))
    file.close()


# Prints the menu for the user
def start(startState, goalState, file):
    print("Enter your choice depending on the heuristic :")
    print("1. h1(n) = Number of tiles displaced from their final position")
    print("2. h2(n) = Total Manhattan distance")
    print("3. h3​(n) = h1​(n) * h2(n)")

    heuristic_choice = int(input("Enter your choice... "))
    if heuristic_choice == 4:
    	heuristic_choice = 3

    is_blank_tile_included = int(input("Enter 1 so as to consider blank tile as another tile else enter 0... "))
    if is_blank_tile_included:
        Heuristic.isTileInclude = True
    print("Enter choice depending on the cooling function:\n 1.Linear Function \n 2.Random Strategy \n"
          " 3.Negative Exponential Function \n 4.Positive Exponential function\n")
    cooling_function_choice = int(input("Enter your choice... "))
    initial_temperature = int(input("Enter the value of TMAX... "))

    write_to_file(file, heuristic_choice, startState, goalState, cooling_function_choice, initial_temperature)

    start_time = datetime.now()
    puzzle_solver = simulated_annealing(startState, goalState, initial_temperature)
    search_status = puzzle_solver.solve_eight_puzzle(startState, heuristic_choice, cooling_function_choice)
    file = open("simulated_annealing_result.txt", "a")
    if search_status == 1:
        file.write("Search Status : Failed\n")
    file.write("Time Taken : {} ".format(str(datetime.now() - start_time)))
    file.close()

# Driver function to run the main program
if __name__ == '__main__':
    startState = ""
    goalState = ""

    # Opens the output file in write mode
    file = open("simulated_annealing_result.txt", "w+")

    # Convert the start state to string form for easy manipulation
    with open("A*StartState") as f:
        for line in f:
            line = line.strip()
            line = line.replace(" ", "")
            startState += line

    # Convert the goal state to string form for easy manipulation
    with open("A*GoalState") as f:
        for line in f:
            line = line.strip()
            line = line.replace(" ", "")
            goalState += line

    startState = replaceTB(startState)
    goalState = replaceTB(goalState)

    # Execute the program
    start(startState, goalState, file)
