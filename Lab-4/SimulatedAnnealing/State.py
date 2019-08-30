from copy import deepcopy

from Heuristic import Heuristic

from Utils import *


class State:
    def __init__(self, stateInfo, h=0):
        self.puzzleState = stateInfo
        self.hvalue = h

    def getAllSuccessor(self, heuristic_choice, final_configuration):
        x = [1, -1, 0, 0]
        y = [0, -0, 1, -1]
        
        puzzleMatrix = convertStringToMatrix(self.puzzleState)
        for i in range(3):
            for j in range(3):
                if puzzleMatrix[i][j] == 0:
                    blankX = i
                    blankY = j
                    break

        successorState = []
        for (xMove, yMove) in zip(x, y):
            if 0 <= blankX + xMove < 3 and 0 <= blankY + yMove < 3:
                successorPuzzleMat = deepcopy(puzzleMatrix)
                temp = successorPuzzleMat[blankX + xMove][blankY + yMove]
                successorPuzzleMat[blankX + xMove][blankY + yMove] = 0
                successorPuzzleMat[blankX][blankY] = temp
                new_state = convertMatrixToString(successorPuzzleMat)
                successorState.append(State(new_state, Heuristic(final_configuration).getHeuristicEstimation(new_state,heuristic_choice)))

        return successorState

    def __eq__(self, other):
        return self.puzzleState == other.puzzleState

    def __lt__(self, other):
        return self.hvalue < other.hvalue
