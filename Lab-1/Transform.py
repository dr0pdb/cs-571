def convertStringToEightPuzzle(state):
    puzzleMatrix = [[0 for i in range(3)] for j in range(3)]
    puzzleState = state
    for i in range(3):
        for j in range(3):
            puzzleMatrix[i][j] = int(puzzleState[i * 3 + j])
    return puzzleMatrix

def convertEightPuzzleToString(puzzleMatrix):
    stringRepresentationOfState = ""
    for i in range(3):
        for j in range(3):
            stringRepresentationOfState += str(puzzleMatrix[i][j])
    return stringRepresentationOfState
