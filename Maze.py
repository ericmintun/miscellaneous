'''
A class for cartesian mazes.  A maze is a rectangular grid, where every edge between grid points may or may not contain a wall.  Walls are stored in an array of tuples, where each tuple is a boolean answering the question 'is there a wall to the (right, down) of this grid point?'  The rightmost and bottommost grid points ignore the results for the edge of the maze.  Also includes start and end grid points.
'''

import random


class Maze:

    
    def __init__(self, walls, start = None, end = None):
        self.walls = walls
        self.size = (len(walls), len(walls[0]))
        self.start = start
        self.end = end
        
        self.horzChar = "\u2500"
        self.vertChar = "\u2502"
        self.downRightChar = "\u250C"
        self.downLeftChar = "\u2510"
        self.upRightChar = "\u2514"
        self.upLeftChar = "\u2518"
        self.tRightChar = "\u251C"
        self.tLeftChar = "\u2524"
        self.tDownChar = "\u252C"
        self.tUpChar = "\u2534"
        self.downChar = "\u2577"
        self.rightChar = "\u2576"
        self.upChar = "\u2575"
        self.leftChar = "\u2574"
        self.plusChar = "\u253C"
        self.emptyChar = " "
        self.startChar = "S"
        self.endChar = "E"


    def changeWall(self, cellOne, cellTwo, newValue):
        diff = (cellTwo[0] - cellOne[0], cellTwo[1] - cellOne[1])
        if diff[1] == 0:
            if diff[0] == 1:
                self.walls[cellOne[0]][cellOne[1]] = (newValue, self.walls[cellOne[0]][cellOne[1]][1])
            elif diff[0] == -1:
                self.walls[cellTwo[0]][cellTwo[1]] = (newValue, self.walls[cellTwo[0]][cellTwo[1]][1])
            else:
                raise ValueError("Non-adjacent cells provided.")
        elif diff[0] == 0:
            if diff[1] == 1:
                self.walls[cellOne[0]][cellOne[1]] = (self.walls[cellOne[0]][cellOne[1]][0], newValue)
            elif diff[1] == -1:
                self.walls[cellTwo[0]][cellTwo[1]] = (self.walls[cellTwo[0]][cellTwo[1]][0], newValue)
            else:
                raise ValueError("Non-adjacent cells provided.")
        else:
            raise ValueError("Non-adjacent cells provided.")

    def adjCells(self, cell):
        adj = []
        if cell[0] is not 0 and self.walls[cell[0]-1][cell[1]][0] is not True:
            adj.append((cell[0]-1, cell[1]))
        if cell[0] is not self.size[0]-1 and self.walls[cell[0]][cell[1]][0] is not True:
            adj.append((cell[0]+1, cell[1]))
        if cell[1] is not 0 and walls[cell[0]][cell[1]-1][1] is not True:
            adj.append((cell[0],cell[1]-1))
        if cell[1] is not self.size[1]-1 and self.walls[cell[0]][cell[1]][1] is not True:
            adj.append((cell[0],cell[1]+1))

        return adj

    def printSpace(self, p):
        if p[0] == -1 or p[1] == -1:
            return
        elif p == self.start:
            print(self.startChar, end='')
        elif p == self.end:
            print(self.endChar, end='')
        else:
            print(self.emptyChar, end='')

    def printVertWall(self, p):
        if p[1] == -1:
            return
        if p[0] == self.size[0]-1 or p[0] == -1 or self.walls[p[0]][p[1]][0] is True:
            print(self.vertChar, end='')
        else:
            print(self.emptyChar, end='')

    def printHorzWall(self, p):
        if p[0] == -1:
            return
        if p[1] == self.size[1]-1 or p[1] == -1 or self.walls[p[0]][p[1]][1] is True:
            print(self.horzChar, end='')
        else:
            print(self.emptyChar, end='')

    def printVertex(self, p):
        i = p[1]
        j = p[0]

        if i == -1 and j == -1: #Top left corner
            print(self.downRightChar, end='')
            return
        elif i == -1 and j == self.size[0]-1: #Top right corner
            print(self.downLeftChar, end='')
            return
        elif i == self.size[1]-1 and j == -1: #Bottom left corner
            print(self.upRightChar, end='')
            return
        elif i == self.size[1]-1 and j == self.size[0]-1: #Bottom right corner
            print(self.upLeftChar, end='')
            return
        elif i == -1:
            if self.walls[j][0][0] is True: #Top edge
                print(self.tDownChar, end='')
            else:
                print(self.horzChar, end='')
            return
        elif i == self.size[1]-1: #Bottom edge
            if self.walls[j][self.size[1]-1][0] is True:
                print(self.tUpChar, end='')
            else:
                print(self.horzChar, end='')
            return
        elif j == -1: #Left edge
            if self.walls[0][i][1] is True:
                print(self.tRightChar, end='')
            else:
                print(self.vertChar, end='')
            return
        elif j == self.size[0]-1: #Right edge
            if self.walls[self.size[0]-1][i][1] is True:
                print(self.tLeftChar, end='')
            else:
                print(self.vertChar, end='')
            return

        if self.walls[j][i][0] is True and self.walls[j][i][1] is True and self.walls[j+1][i][1] is True and self.walls[j][i+1][0] is True:
            print(self.plusChar, end='')
        elif self.walls[j][i][0] is False and self.walls[j][i][1] is True and self.walls[j+1][i][1] is True and self.walls[j][i+1][0] is True:
            print(self.tDownChar, end='')
        elif self.walls[j][i][0] is True and self.walls[j][i][1] is False and self.walls[j+1][i][1] is True and self.walls[j][i+1][0] is True:
            print(self.tRightChar, end='')
        elif self.walls[j][i][0] is True and self.walls[j][i][1] is True and self.walls[j+1][i][1] is False and self.walls[j][i+1][0] is True:
            print(self.tLeftChar, end='')
        elif self.walls[j][i][0] is True and self.walls[j][i][1] is True and self.walls[j+1][i][1] is True and self.walls[j][i+1][0] is False:
            print(self.tUpChar, end='')
        elif self.walls[j][i][0] is False and self.walls[j][i][1] is False and self.walls[j+1][i][1] is True and self.walls[j][i+1][0] is True:
            print(self.downRightChar, end='')
        elif self.walls[j][i][0] is False and self.walls[j][i][1] is True and self.walls[j+1][i][1] is False and self.walls[j][i+1][0] is True:
            print(self.downLeftChar, end='')
        elif self.walls[j][i][0] is False and self.walls[j][i][1] is True and self.walls[j+1][i][1] is True and self.walls[j][i+1][0] is False:
            print(self.horzChar, end='')
        elif self.walls[j][i][0] is True and self.walls[j][i][1] is False and self.walls[j+1][i][1] is False and self.walls[j][i+1][0] is True:
            print(self.vertChar, end='')
        elif self.walls[j][i][0] is True and self.walls[j][i][1] is False and self.walls[j+1][i][1] is True and self.walls[j][i+1][0] is False:
            print(self.upRightChar, end='')
        elif self.walls[j][i][0] is True and self.walls[j][i][1] is True and self.walls[j+1][i][1] is False and self.walls[j][i+1][0] is False:
            print(self.upLeftChar, end='')
        elif self.walls[j][i][0] is False and self.walls[j][i][1] is False and self.walls[j+1][i][1] is False and self.walls[j][i+1][0] is True:
            print(self.downChar, end='')
        elif self.walls[j][i][0] is False and self.walls[j][i][1] is False and self.walls[j+1][i][1] is True and self.walls[j][i+1][0] is False:
            print(self.rightChar, end='')
        elif self.walls[j][i][0] is False and self.walls[j][i][1] is True and self.walls[j+1][i][1] is False and self.walls[j][i+1][0] is False:
            print(self.leftChar, end='')
        elif self.walls[j][i][0] is True and self.walls[j][i][1] is False and self.walls[j+1][i][1] is False and self.walls[j][i+1][0] is False:
            print(self.upChar, end='')
        elif self.walls[j][i][0] is False and self.walls[j][i][1] is False and self.walls[j+1][i][1] is False and self.walls[j][i+1][0] is False:
            print(self.emptyChar, end='')
        else:
            raise ValueError("Somehow got to the end of a complete truth table????")


    def printASCII(self):
        for i in range(-1, self.size[1]):
            for j in range(-1, self.size[1]):
                self.printSpace((j,i))
                self.printVertWall((j,i))
            print("\n",end='')
            for j in range(-1, self.size[1]):
                self.printHorzWall((j,i))
                self.printVertex((j,i))
            print("\n",end='')

if __name__ == "__main__":
    
    maze1Booleans = [[(False, True), (True, False), (False, True)],[(True, False), (True, False), (False, True)],[(True, False), (True, False), (True, True)]]
    maze1 = Maze(maze1Booleans)
    maze1.printASCII()

    maze2Booleans = []
    for i in range(20):
        maze2Booleans.append([])
        for j in range(20):
            maze2Booleans[i].append((random.choice([True, False]), random.choice([True, False])))

    maze2 = Maze(maze2Booleans, (0,0), (19,19))
    maze2.printASCII()

