def convertStringToMatrix(state):
    puzzleMatrix = [[0 for i in range(3)] for j in range(3)]
    puzzleState = state
    for i in range(3):
        for j in range(3):
            puzzleMatrix[i][j] = int(puzzleState[i * 3 + j])
    return puzzleMatrix

def convertMatrixToString(puzzleMatrix):
    stringRep = ""
    for i in range(3):
        for j in range(3):
            stringRep += str(puzzleMatrix[i][j])
    return stringRep

def replaceTB(txt):
    txt = txt.replace("T", "")
    txt = txt.replace("B", "0")
    return txt