import math
from heapq import *
import heapq
import time


#Different Heuristics
UNINFORMED = 0
DISPLACED = 1
MANHATTAN = 2
PESSIMISTIC = 3
heuristicType = 0 #it will tell which heuristic we will use

#class to define a state of tiles
#dist 	: distance from the start state
#tiles 	: Matrix of tiles
#cost 	: value of f(n)
#parent : parent of the state, through which we can reach this(in optimal path)
class state:
	def __init__(self, position, dist = math.inf, parent = None):
		self.dist = dist
		self.tiles = [list(map(int, position[0:3])), list(map(int, position[3:6])), list(map(int, position[6:9]))]
		self.cost = costCalculate(dist, self.tiles)
		self.parent = parent
	
	#overriding 'less than' operator to compare two 'state' class objects
	def __lt__(self, other):
		return self.cost < other.cost

		
#to calculate f(n)
def costCalculate(dist, tiles):
	if (dist == math.inf):
		return math.inf
	if(heuristicType == UNINFORMED):
		return dist
	elif(heuristicType == DISPLACED):
		return dist + noOfDisplacedTiles(tiles)
	elif(heuristicType == MANHATTAN):
		return dist + manhattanDistance(tiles)
	else:
		return dist + pessimisticValue(tiles)

#convert matrix of tiles to string
def toString(values):
	s = ""
	s = "".join(map(str, values[0])) + "".join(map(str, values[1])) + "".join(map(str, values[2]))
	return s

#4th heuristic
def pessimisticValue(val):
	totalDistance = 0
	a = toString(val)
	for i in range(9):
		for j in range(9):
			if(a[i] == g[j]):
				totalDistance += (abs((i//3) - (j//3)))**5 + (abs((i%3) - (j%3)))**5
	return totalDistance

#calculate manhattanDistance
def manhattanDistance(val):
	totalDistance = 0
	a = toString(val)
	for i in range(9):
		for j in range(9):
			if(a[i] != 0 and a[i] == g[j]):
				totalDistance += abs((i//3) - (j//3)) + abs((i%3) - (j%3))
	return totalDistance

#calculate number of displaced tiles
def noOfDisplacedTiles(val):
	a = toString(val)
	count = 0;
	for i in range(9):
		if(g[i] != 0 and g[i] != a[i]):
			count += 1
	return count

def printMatrix(a):
	for i in range(3):
		for j in range(3):
			if(a[i][j] != 0):
				print(a[i][j], end = "|")
			else:
				print(" ", end = "|")
		print()
	return

#function to use parent pointers to return final optimal path
def optimalPath(state):
	if(state.parent == None):
		return
	optimalPath(state.parent)
	printMatrix(state.tiles)
	print("")
	return 

#fuction to run the A* algorithm
def steps():
	mapStateToAddress = {} #hash map, state(string form) -> 'state' class object
	openList = []
	closedList = set()
	mapStateToAddress[s] = state(s, 0)	#storing start state object in map
	heappush(openList, mapStateToAddress[s])
	while(len(openList) != 0):
		current = heappop(openList)
		closedList.add(toString(current.tiles))

		#if goal state is transferred to closed list, then print optimal path
		if(toString(current.tiles) == g):
			print("**********Success**********")
			print("Start State:")
			printMatrix(mapStateToAddress[s].tiles)
			print("\nGoal State:")
			printMatrix(mapStateToAddress[g].tiles)
			print("\nTotal number of states explored:", len(mapStateToAddress))
			print("Total number of states on optimal path:", current.dist+1)
			print("\nOptimal Path:")
			optimalPath(current)
			print("Optimal cost of the path:", current.dist)
			break

		#finding (x, y) :coordinates of blank tile
		x = 0
		y = 0
		for i in range(3):
			for j in range(3):
				if(current.tiles[i][j] == 0):
					x = i
					y = j
					break

		#blank tile can move either up/down/left/right, so checking all possibilities
		for i in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
			xd = x+i[0]
			yd = y+i[1]
			if(xd >= 0 and xd < 3 and yd >= 0 and yd < 3):
				newStateTiles = [row[:] for row in current.tiles]	#matrix of tiles of new state
				newStateTiles[x][y] = newStateTiles[xd][yd]			#moving the blank tile
				newStateTiles[xd][yd] = 0
				stringForm = toString(newStateTiles)
				newDist = current.dist+1;

				#if new state is already in closed list, then check other reachable states, ignore this
				if(stringForm in closedList):
					continue
				
				#if new state is in open list, check if its cost can be updated
				elif(stringForm in mapStateToAddress.keys()):
					existingState = mapStateToAddress[stringForm]
					
					#if new distance is less than the earlier reachable distance, then update it
					if(newDist < existingState.dist):
						existingState.cost = existingState.cost - existingState.dist + newDist
						existingState.dist = newDist
						indexInHeap = openList.index(existingState)
						existingState.parent = current
						heapq._siftdown(openList, 0, indexInHeap)	#open list heap is modified according to updated distance and cost
				
				#else if this state is reached first time, then add it to openlist
				else:
					mapStateToAddress[stringForm] = state(stringForm, newDist, current)
					heappush(openList, mapStateToAddress[stringForm])
	if(g not in closedList):
		print("**********Failure**********")
		print("Start State:")
		printMatrix([list(map(int, s[0:3])), list(map(int, s[3:6])), list(map(int, s[6:9]))])
		print("\nGoal State:")
		printMatrix([list(map(int, g[0:3])), list(map(int, g[3:6])), list(map(int, g[6:9]))])
		print("\nTotal number of states explored:", len(mapStateToAddress))
	return mapStateToAddress.keys()

#'a' is the set of states expanded by inferior heuristic
#'b' is the set of states expanded by superior heuristic
#function is checking if all elements of 'b' are present in 'a'
def ifAllExpanded(a, b):
	for i in b:
		if i not in a:
			print("Wrongly Expanded")
			return
	print("All the states expanded by better heuristics are also expanded by the inferior heuristics")
	return

if __name__ == "__main__":
	global s, g
	print("If the matix is of the form:\n1 2 3\n4 5 6\n7 8 \nthen input format: 123456780")
	print("Here, '0' stands for the blank tile")
	s = input("\nEnter the start state: ")
	g = input("Enter the goal state: ")
	print(s, g)
	for i in range(4):
		heuristicType = i
		if(i == 0):
			start_time = time.time()
			print("Uninformed Heuristics")
			print("----------------------------------")
			statesExpanded1 = steps()
			print("Exection Time:", time.time()-start_time)
		elif(i == 1):
			start_time = time.time()
			print("\n\nDisplaced Tiles Heuristics")
			print("----------------------------------")
			statesExpanded2 = steps()
			print("Exection Time:", time.time()-start_time)
		elif(i == 2):
			start_time = time.time()
			print("\n\nManhattan Distance Heuristics")
			print("----------------------------------")
			statesExpanded3 = steps()
			print("Exection Time:", time.time()-start_time)
		else:
			start_time = time.time()
			print("\n\nPessimistic Heuristics")
			print("----------------------------------")
			steps()
			print("Exection Time:", time.time()-start_time)
			
	print("\n\nChecking if all states of Displaced Tiles Heuristics are expanded by Uninformed Heuristic:")
	ifAllExpanded(statesExpanded1, statesExpanded2)
	print("\nChecking if all states of Manhattan Distance Heuristics are expanded by Displaced Tiles Heuristic:")
	ifAllExpanded(statesExpanded2, statesExpanded3)
	print("\nChecking if all states of Manhattan Distance Heuristics are expanded by Uninformed Heuristic:")
	ifAllExpanded(statesExpanded1, statesExpanded3)
		