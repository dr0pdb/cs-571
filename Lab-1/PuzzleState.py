from copy import deepcopy

from Utils import *


class State:
    def __init__(self, stateInfo, f=0, h=0, g=0):
        self.puzzleState = stateInfo
        self.fvalue = f
        self.gvalue = g
        self.hvalue = h

    def getAllSuccessor(self):
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
                nextSuccessor = deepcopy(puzzleMatrix)
                temp = nextSuccessor[blankX + xMove][blankY + yMove]
                nextSuccessor[blankX + xMove][blankY + yMove] = 0
                nextSuccessor[blankX][blankY] = temp
                successorState.append(convertMatrixToString(nextSuccessor))

        return successorState

    def __eq__(self, other):
        return self.puzzleState == other.puzzleState

    def __lt__(self, other):
        if self.fvalue < other.fvalue:
            return True
        elif self.fvalue == other.fvalue:
            if self.gvalue == other.gvalue:
                return self.puzzleState < other.puzzleState
            elif self.gvalue < other.gvalue:
                return True
            else:
                return False
        else:
            return False
