# Writes the info about the problem to the file.
def write_to_file(file, heuristic_choice, start_state, goal_state):
    file.write("Heuristic Chosen : ")
    if heuristic_choice == 1:
        file.write("number of tiles displaced from their destined position \n")
    elif heuristic_choice == 2:
        file.write("Total manhattan distance \n")
    else:
        file.write("3 * displace tiles - 2 * manhatten distance\n")
    file.write("Start State : \n")
    file.write(start_state[:3] + "\n" + start_state[3:6] + "\n" + start_state[6:] + "\n")
    file.write("Goal State : \n")
    file.write(goal_state[:3] + "\n" + goal_state[3:6] + "\n" + goal_state[6:] + "\n")
    file.close()