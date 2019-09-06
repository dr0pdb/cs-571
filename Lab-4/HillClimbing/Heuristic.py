import numpy

from Utils import *

# Specifies whether the blank tile is also considered a tile or not.
isTileInclude = False

class Heuristic:
    def __init__(self, goalState="123456780"):
        self.goalState = goalState

    def tilesDisplacedHeuristic(self, state):
        currentPuzzleState = convertStringToMatrix(state)
        goalPuzzleState = convertStringToMatrix(self.goalState)
        h = 0
        for i in range(3):
            for j in range(3):
                if currentPuzzleState[i][j] != goalPuzzleState[i][j]:
                    h += 1
                if currentPuzzleState[i][j] == 0 and currentPuzzleState[i][j] != goalPuzzleState[i][
                    j] and isTileInclude == False:
                    h -= 1
        return h

    def manhattanHeuristic(self, state):
        currentPuzzleState = convertStringToMatrix(state)
        goalPuzzleState = convertStringToMatrix(self.goalState)
        currentCoordinate = numpy.arange(18).reshape((9, 2))

        for i in range(3):
            for j in range(3):
                currentCoordinate[currentPuzzleState[i][j]][0] = i
                currentCoordinate[currentPuzzleState[i][j]][1] = j

        h = 0
        for i in range(3):
            for j in range(3):
                if goalPuzzleState[i][j] != 0:
                    h += abs(i - currentCoordinate[goalPuzzleState[i][j]][0]) + \
                         abs(j - currentCoordinate[goalPuzzleState[i][j]][1])
                if goalPuzzleState[i][j] == 0 and isTileInclude:
                    h += abs(i - currentCoordinate[goalPuzzleState[i][j]][0]) + \
                         abs(j - currentCoordinate[goalPuzzleState[i][j]][1])
        return h

    def getHeuristicEstimation(self, state, HeuristicChoice):
        return {
            1: self.tilesDisplacedHeuristic(state),
            2: self.manhattanHeuristic(state),
            3: 3*self.tilesDisplacedHeuristic(state) - 2*self.manhattanHeuristic(state),
            4: 2*self.tilesDisplacedHeuristic(state) * self.manhattanHeuristic(state)
        }[HeuristicChoice]
