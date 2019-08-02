class Transform:
    def __init__(self):
        pass

    def convertStringToEightPuzzle(self, state):
        puzzleMatrix = [[0 for i in range(3)] for j in range(3)]
        puzzleState = state
        for i in range(3):
            for j in range(3):
                puzzleMatrix[i][j] = int(puzzleState[i * 3 + j])
        return puzzleMatrix

    def convertEightPuzzleToString(self, puzzleMatrix):
        stringRepresentationOfState = ""
        for i in range(3):
            for j in range(3):
                stringRepresentationOfState += str(puzzleMatrix[i][j])
        return stringRepresentationOfState
