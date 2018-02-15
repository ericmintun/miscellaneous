'''
Generates a loopless maze by drawing a random, non-intersecting path from one point to another, then connecting random points via random paths to that path until all points are connected.
'''

import Maze
import numpy as np
import random

def getNextCell(c, touchedCells, sRatio = None, sVector = None):
    #If no difference between straight and turns
    if sVector is None or sRatio is None:
        allowedCells = []
        if c[0] != 0 and touchedCells[(c[0]-1,c[1])] == False:
            allowedCells.append((c[0]-1, c[1]))
        if c[0] != touchedCells.shape[0]-1 and touchedCells[(c[0]+1, c[1])] == False:
            allowedCells.append((c[0]+1, c[1]))
        if c[1] != 0 and touchedCells[(c[0], c[1]-1)] == False:
            allowedCells.append((c[0],c[1]-1))
        if c[1] != touchedCells.shape[1]-1 and touchedCells[(c[0],c[1]+1)] == False:
            allowedCells.append((c[0],c[1]+1))
    
        if allowedCells == []:
            return None
        else:
            return random.choice(allowedCells)
    else: #If straight is preferred
        sCell = (c[0] + sVector[0], c[1] + sVector[1])
        if sCell[0] != -1 and sCell[0] != touchedCells.shape[0] and sCell[1] != -1 and sCell[1] != touchedCells.shape[1] and touchedCells[sCell] == False:
            sCellAllowed = True
        else:
            sCellAllowed = False

        turnVector = (sVector[1], sVector[0])
        tCellOne = (c[0] + turnVector[0], c[1] + turnVector[1])
        tCellTwo = (c[0] - turnVector[0], c[1] - turnVector[1])
        tCellsAllowed = []
        if tCellOne[0] != -1 and tCellOne[0] != touchedCells.shape[0] and tCellOne[1] != -1 and tCellOne[1] != touchedCells.shape[1] and touchedCells[tCellOne] == False:
            tCellsAllowed.append(tCellOne)
        if tCellTwo[0] != -1 and tCellTwo[0] != touchedCells.shape[0] and tCellTwo[1] != -1 and tCellTwo[1] != touchedCells.shape[1] and touchedCells[tCellTwo] == False:
            tCellsAllowed.append(tCellTwo)

        if sCellAllowed is False:
            if tCellsAllowed == []:
                return None
            else:
                return random.choice(tCellsAllowed)
        else:
            if tCellsAllowed == []:
                return sCell
            elif len(tCellsAllowed) == 1:
                baseP = 1.0/2.0
            else:
                baseP = 1.0/3.0

            if random.random() < baseP * sRatio:
                return sCell
            else:
                return random.choice(tCellsAllowed)


#This seems slow but I don't have a better method atm.  Also the mazes are small so it won't matter    
def randomCell(touchedCells, qTouched = False):
    validCells = []
    it = np.nditer(touchedCells, flags=['multi_index'])
    while not it.finished:
        if it[0] == qTouched:
            validCells.append(it.multi_index)
        it.iternext()
    if validCells == []:
        return None
    else:
        return random.choice(validCells)

def allWallsMaze(size):
    walls = []
    for i in range(size[0]):
        walls.append([])
        for j in range(size[1]):
            walls[i].append((True, True))
    return Maze.Maze(walls)

def cutNewPath(maze, cellsTouched, start, straightRatio = None, maxLength = None, protectedCells = None):
    currentCell = start
    sVector = None
    qCellTouchedThisPath = np.full(cellsTouched.shape, False, dtype=bool)
    qCellTouchedThisPath[start] = True
    if protectedCells != None:
        for c in protectedCells:
            qCellTouchedThisPath[c] = True
    pathLength = 0
    pathList = [start]
    while True:
        nextCell = getNextCell(currentCell, qCellTouchedThisPath, straightRatio, sVector)
        pathLength += 1
        #The path wandered into itself unavoidably, so pick a random place on the path and try a new direction
        if nextCell is None:
            #We somehow filled the whole grid with this one path and should stop
            if qCellTouchedThisPath.all():
                return nextCell, cellsTouched | qCellTouchedThisPath
            #if protectedCells != None:
            #    for c in protectedCells:
            #        qCellTouchedThisPath[c] = False
            #currentCell = randomCell(qCellTouchedThisPath, True)
            #if protectedCells != None:
            #    for c in protectedCells:
            #        qCellTouchedThisPath[c] = True
            currentCell = pathList[-1]
            pathList.remove(currentCell)
        else:
            maze.changeWall(currentCell, nextCell, False)
            sVector = (nextCell[0] - currentCell[0], nextCell[1] - currentCell[1])
            qCellTouchedThisPath[nextCell] = True
            pathList.append(nextCell)
            #The path hit a previous path, so we should add the path to the set of touched cells and stop.
            if cellsTouched[nextCell] == True:
                return nextCell, cellsTouched | qCellTouchedThisPath
            #This is the main path and is length limited instead of intersection limited.  Stop after a certain number of steps.
            if maxLength is not None and pathLength >= maxLength:
                return nextCell, cellsTouched | qCellTouchedThisPath
            #Otherwise keep going
            currentCell = nextCell


def randomPathMaze(size, attemptedMainPathLength = None, straightRatio = None):
    if attemptedMainPathLength is None:
        attemptedMainPathLength = (size[0] + size[1])

    maze = allWallsMaze(size)
    qCellTouched = np.full(size, False, dtype=bool)

    start = randomCell(qCellTouched)
    qCellTouched[start] = True

    #Main path
    end, qCellTouched = cutNewPath(maze, qCellTouched, start, straightRatio, attemptedMainPathLength)
    maze.start = start
    maze.end = end
    maze.printASCII()

    while not qCellTouched.all():
        newPathStart = randomCell(qCellTouched)
        endCell, qCellTouched = cutNewPath(maze, qCellTouched, newPathStart, straightRatio, None, [start, end])
    return maze

if __name__ == "__main__":

    testMaze = randomPathMaze((50,50), 750, 1.3)
    testMaze.printASCII()


