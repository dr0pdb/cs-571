import queue as Q
import random
import numpy

from PuzzleState import State
from Transform import Transform

isTileInclude = False
closedLlst = {}

class Heuristic:
    def __init__(self, goalState=None):
        self.goalState = goalState

    def zeroHeuristic(self):
        return 0

    def tilesDisplacedHeuristic(self, state):
        transform = Transform()
        currentPuzzleState = transform.convertStringToEightPuzzle(state)
        goalPuzzleState = transform.convertStringToEightPuzzle(self.goalState.puzzleState)
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
        transform = Transform()
        currentPuzzleState = transform.convertStringToEightPuzzle(state)
        goalPuzzleState = transform.convertStringToEightPuzzle(self.goalState.puzzleState)
        currentCoOrdinate = numpy.arange(18).reshape((9, 2))

        for i in range(3):
            for j in range(3):
                currentCoOrdinate[currentPuzzleState[i][j]][0] = i
                currentCoOrdinate[currentPuzzleState[i][j]][1] = j

        h = 0
        for i in range(3):
            for j in range(3):
                if goalPuzzleState[i][j] != 0:
                    h += abs(i - currentCoOrdinate[goalPuzzleState[i][j]][0]) + \
                         abs(j - currentCoOrdinate[goalPuzzleState[i][j]][1])
                if goalPuzzleState[i][j] == 0 and isTileInclude:
                    h += abs(i - currentCoOrdinate[goalPuzzleState[i][j]][0]) + \
                         abs(j - currentCoOrdinate[goalPuzzleState[i][j]][1])
        return h

    def overEstimatedHeuristic(self, state):
        return random.randint(self.manhattanHeuristic(state) ** 2, 181440)

    def getHeuristicEstimation(self, state, HeuristicChoice):
        return {
            1: self.zeroHeuristic(),
            2: self.tilesDisplacedHeuristic(state),
            3: self.manhattanHeuristic(state),
            4: self.overEstimatedHeuristic(state)
        }[HeuristicChoice]

    def getHeuristicName(self, HeuristicChoice):
        return {
            1: "Zero Heuristic          : ",
            2: "tilesDisplacedHeuristic : ",
            3: "manhattanHeuristic      : ",
            4: "overEstimatedHeuristic  : "
        }[HeuristicChoice]


def printStatistics(initialState, finalState, puzzleStateMapWithItsParent, stateExplored, HeuristicChoice):
    temp = []
    temp.append(Heuristic().getHeuristicName(HeuristicChoice))
    print(Heuristic().getHeuristicName(HeuristicChoice))
    print("Valid Path Exist: ")
    printExtremeState(finalState, initialState)

    print("Total number of states explored.")
    temp.append(stateExplored)
    print(stateExplored)

    print("Optimal Path : ")
    statesOnOptimalPath = printOptimalPath(finalState.puzzleState, 0, puzzleStateMapWithItsParent)

    print("Total number of states on optimal path.")
    temp.append(statesOnOptimalPath)
    print(statesOnOptimalPath)

    print("Optimal Cost of the path.")
    temp.append(statesOnOptimalPath - 1)
    print(statesOnOptimalPath - 1)
    return temp


def printExtremeState(initialState, finalState):
    print("Start State : ")
    startState = Transform().convertStringToEightPuzzle(initialState.puzzleState)
    for i in range(3):
        for j in range(3):
            if startState[i][j] == 0:
                print("B", end=" ")
            else:
                print("T" + str(startState[i][j]), end=" ")
        print()
    print("Goal State : ")
    goalState = Transform().convertStringToEightPuzzle(finalState.puzzleState)
    for i in range(3):
        for j in range(3):
            if goalState[i][j] == 0:
                print("B", end=" ")
            else:
                print("T" + str(goalState[i][j]), end=" ")
        print()


def printOptimalPath(state, depth, puzzleStateMapWithItsParent):
    if state is None:
        return depth
    else:
        totalState = printOptimalPath(puzzleStateMapWithItsParent[state], depth + 1, puzzleStateMapWithItsParent)
        eightPuzzleConfiguration = Transform().convertStringToEightPuzzle(state)
        for i in range(3):
            for j in range(3):
                if eightPuzzleConfiguration[i][j] == 0:
                    print("B", end=" ")
                else:
                    print("T" + str(eightPuzzleConfiguration[i][j]), end=" ")
            print()
        print("###########################")
        return totalState


class EightPuzzle:
    def __init__(self, initialState, goalState):
        self.initialPuzzleConfiguration = initialState
        self.finalPuzzleConfiguration = goalState

    def solveEightPuzzle(self, HeuristicChoice):
        global closedLlst
        openList = Q.PriorityQueue() # state, g, h
        openList.put(self.initialPuzzleConfiguration, 0, Heuristic(
            self.finalPuzzleConfiguration).getHeuristicEstimation(
            self.initialPuzzleConfiguration.puzzleState, HeuristicChoice))
        puzzleStateMapWithItsParent = {} # stores the parent of a node.
        puzzleStateMapWithGvalue = {} # stores g value of a node.
        closedList = {} # stores whether a node is explored or not.
        stateExplored = 0
        puzzleStateMapWithGvalue[self.initialPuzzleConfiguration.puzzleState] = 0
        puzzleStateMapWithItsParent[self.initialPuzzleConfiguration.puzzleState] = None
        closedLlst[self.initialPuzzleConfiguration.puzzleState] = 1 # mark initial node as explored.
        currentNode = None
        print("State : h(State) , ChildState : h(ChildState) ")
        while not openList.empty():
            currentNode = openList.get()
            if currentNode.puzzleState in closedList:
                continue
            if currentNode == self.finalPuzzleConfiguration:
                return closedList, printStatistics(self.initialPuzzleConfiguration, self.finalPuzzleConfiguration,
                                                   puzzleStateMapWithItsParent, stateExplored, HeuristicChoice)

            successorState = currentNode.getAllSuccessor()
            for stateStringRepresentation in successorState:
                h = Heuristic(self.finalPuzzleConfiguration).getHeuristicEstimation(stateStringRepresentation,
                                                                                    HeuristicChoice)
                print(currentNode.puzzleState + ": " +str(currentNode.hvalue), stateStringRepresentation + " " +str(h))
                closedLlst[stateStringRepresentation] = 1
                successorStateGvalue = puzzleStateMapWithGvalue[currentNode.puzzleState] + 1
                if stateStringRepresentation in closedList:
                    if successorStateGvalue <= puzzleStateMapWithGvalue[stateStringRepresentation]:
                        puzzleStateMapWithGvalue[stateStringRepresentation] = successorStateGvalue
                        puzzleStateMapWithItsParent[stateStringRepresentation] = currentNode.puzzleState
                        openList.put(State(stateStringRepresentation,
                                           successorStateGvalue + h, h, successorStateGvalue))
                elif stateStringRepresentation in puzzleStateMapWithGvalue:
                    if successorStateGvalue < puzzleStateMapWithGvalue[stateStringRepresentation]:
                        puzzleStateMapWithGvalue[stateStringRepresentation] = successorStateGvalue
                        puzzleStateMapWithItsParent[stateStringRepresentation] = currentNode.puzzleState
                        openList.put(State(stateStringRepresentation,
                                           successorStateGvalue + h, h,successorStateGvalue))
                else:
                    puzzleStateMapWithGvalue[stateStringRepresentation] = successorStateGvalue
                    puzzleStateMapWithItsParent[stateStringRepresentation] = currentNode.puzzleState
                    openList.put(State(stateStringRepresentation, successorStateGvalue + h, h,successorStateGvalue))
            if currentNode.puzzleState not in closedList:
                #print(currentNode.fvalue)
                closedList[currentNode.puzzleState] = 1
                stateExplored += 1

        if currentNode != self.finalPuzzleConfiguration:
            print("Failed Search")
            printExtremeState(self.initialPuzzleConfiguration, self.finalPuzzleConfiguration)
            print("Total number of states explored.")
            print(stateExplored)
