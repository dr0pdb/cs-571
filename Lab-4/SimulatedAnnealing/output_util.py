import sys

class output_result:
    def __init__(self, start_state, file_name, parent_map,state_explored):
        self.start_state = start_state
        self.file_name = file_name
        self.parent = parent_map
        self.state_explored = state_explored
        self.path_length = 0
        sys.setrecursionlimit(181440)

    def write_output_path(self, puzzle_state):
        stack = [puzzle_state]

        while puzzle_state != self.start_state:
            stack.append(self.parent[puzzle_state])
            puzzle_state = self.parent[puzzle_state]
            self.path_length += 1
        stack.pop()
        with open(self.file_name, "a") as f:
            f.write("Total Number of state explored : {}\n".format(str(self.state_explored)))
            f.write("Search Status : Successful\n(sub) optimal Path length: {} \n".format(str(self.path_length)))
            f.write("(Sub) Optimal Path \n")
            f.write(
                puzzle_state[:3] + "\n" + puzzle_state[3:6] + "\n" + puzzle_state[6:] + "\n " + u"\u2304" + "\n")
            f.close()

        while len(stack)!=0 :
            puzzle_state = stack.pop()
            with open(self.file_name, "a") as f:
                f.write(
                    puzzle_state[:3] + "\n" + puzzle_state[3:6] + "\n" + puzzle_state[6:] + "\n " + u"\u2304" + "\n")
                f.close()
