import queue
import random
import numpy

from PuzzleState import State
from Utils import *

isTileInclude = False
cList = {} # closed list.

# Heuristic class for storing the used heuristic and goal state.
class Heuristic:
    def __init__(self, goalState=None):
        self.goalState = goalState

    def zeroHeuristic(self):
        return 0

    def tilesDisplacedHeuristic(self, state):
        currentPuzzleState = convertStringToMatrix(state)
        goalPuzzleState = convertStringToMatrix(self.goalState.puzzleState)
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
        goalPuzzleState = convertStringToMatrix(self.goalState.puzzleState)
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

    def getHeuristicEstimation(self, state, chosenHeuristic):
        return {
            1: self.zeroHeuristic(),
            2: self.tilesDisplacedHeuristic(state),
            3: self.manhattanHeuristic(state),
            4: self.overEstimatedHeuristic(state)
        }[chosenHeuristic]

    def getHeuristicName(self, chosenHeuristic):
        return {
            1: "Zero Heuristic          : ",
            2: "tilesDisplacedHeuristic : ",
            3: "manhattanHeuristic      : ",
            4: "overEstimatedHeuristic  : "
        }[chosenHeuristic]


def printStatistics(initialState, finalState, puzzleStateParent, stateExplored, HeuristicChoice):
    print("SUCCESS!")
    print(Heuristic().getHeuristicName(HeuristicChoice))
    print("Valid Path Exists: ")
    printExtremeState(finalState, initialState)

    print("Total number of states explored.")
    print(stateExplored)

    print("Optimal Path : ")
    statesOnOptimalPath = printOptimalPath(finalState.puzzleState, 0, puzzleStateParent)

    print("Total number of states on optimal path.")
    print(statesOnOptimalPath)

    print("Optimal Cost of the path.")
    print(statesOnOptimalPath - 1)


def printExtremeState(initialState, finalState):
    print("Start State : ")
    startState = convertStringToMatrix(initialState.puzzleState)
    for i in range(3):
        for j in range(3):
            if startState[i][j] == 0:
                print("B", end=" ")
            else:
                print("T" + str(startState[i][j]), end=" ")
        print()
    print("Goal State : ")
    goalState = convertStringToMatrix(finalState.puzzleState)
    for i in range(3):
        for j in range(3):
            if goalState[i][j] == 0:
                print("B", end=" ")
            else:
                print("T" + str(goalState[i][j]), end=" ")
        print()


def printOptimalPath(state, depth, puzzleStateParent):
    if state is None:
        return depth
    else:
        totalState = printOptimalPath(puzzleStateParent[state], depth + 1, puzzleStateParent)
        eightPuzzleConfiguration = convertStringToMatrix(state)
        for i in range(3):
            for j in range(3):
                if eightPuzzleConfiguration[i][j] == 0:
                    print("B", end=" ")
                else:
                    print("T" + str(eightPuzzleConfiguration[i][j]), end=" ")
            print()
        print("###########################")
        return totalState


def printFailure(initialPuzzleConfiguration, finalPuzzleConfiguration, stateExplored):
    print("Failed Search")
    printExtremeState(initialPuzzleConfiguration, finalPuzzleConfiguration)
    print("Total number of states explored.")
    print(stateExplored)


class EightPuzzleProblem:
    def __init__(self, initialState, goalState, statesVisited = None):
        self.initialPuzzleConfiguration = initialState
        self.finalPuzzleConfiguration = goalState
        self.statesVisited = set()

    def solveEightPuzzle(self, HeuristicChoice):
        global cList
        olist = queue.PriorityQueue() # state, g, h
        
        heuristic_value = Heuristic(self.finalPuzzleConfiguration).getHeuristicEstimation(self.initialPuzzleConfiguration.puzzleState, HeuristicChoice)
        olist.put(self.initialPuzzleConfiguration, 0, heuristic_value)
        
        puzzleStateParent = {} # stores the parent of a node.
        puzzleStateG = {} # stores g value of a node.
        closedList = {} # stores whether a node is explored or not.
        
        stateExplored = 0
        puzzleStateG[self.initialPuzzleConfiguration.puzzleState] = 0
        puzzleStateParent[self.initialPuzzleConfiguration.puzzleState] = None
        cList[self.initialPuzzleConfiguration.puzzleState] = 1 # mark initial node as explored.
        currentNode = None
        
        print("State : h(State) , ChildState : h(ChildState) ")
        while not olist.empty():
            currentNode = olist.get()
            if currentNode.puzzleState in closedList:
                continue
            if currentNode == self.finalPuzzleConfiguration:
                self.statesVisited.add(currentNode.puzzleState)
                return self.statesVisited, printStatistics(self.initialPuzzleConfiguration, self.finalPuzzleConfiguration,
                                                   puzzleStateParent, stateExplored, HeuristicChoice)

            successorState = currentNode.getAllSuccessor()
            for successorRep in successorState:
                h = Heuristic(self.finalPuzzleConfiguration).getHeuristicEstimation(successorRep,
                                                                                    HeuristicChoice)
                print(currentNode.puzzleState + ": " +str(currentNode.hvalue), successorRep + " " +str(h))
                cList[successorRep] = 1
                successorStateGvalue = puzzleStateG[currentNode.puzzleState] + 1
                
                if successorRep in closedList:
                    if successorStateGvalue <= puzzleStateG[successorRep]:
                        puzzleStateG[successorRep] = successorStateGvalue
                        puzzleStateParent[successorRep] = currentNode.puzzleState
                        olist.put(State(successorRep,
                                           successorStateGvalue + h, h, successorStateGvalue))
                elif successorRep in puzzleStateG:
                    if successorStateGvalue < puzzleStateG[successorRep]:
                        puzzleStateG[successorRep] = successorStateGvalue
                        puzzleStateParent[successorRep] = currentNode.puzzleState
                        olist.put(State(successorRep,
                                           successorStateGvalue + h, h,successorStateGvalue))
                else:
                    puzzleStateG[successorRep] = successorStateGvalue
                    puzzleStateParent[successorRep] = currentNode.puzzleState
                    olist.put(State(successorRep, successorStateGvalue + h, h,successorStateGvalue))
            if currentNode.puzzleState not in closedList:
                closedList[currentNode.puzzleState] = 1
                self.statesVisited.add(currentNode.puzzleState)
                stateExplored += 1

        if currentNode != self.finalPuzzleConfiguration:
            printFailure(self.initialPuzzleConfiguration, self.finalPuzzleConfiguration, stateExplored)
