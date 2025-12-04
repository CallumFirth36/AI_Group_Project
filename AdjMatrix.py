import numpy as np

# Number of Nodes is used to create an adjacency matrix

# use add node to connect two nodes and add the weight between them

# The same weight is added for both directions e.g. 1 to 2 = 10 and also 2 to 1 = 10

# dijkstra algorithm may have some problems but i haven't been able to fully test it yet!


class dijkstraTableEntry:
    def __init__(self, weight, predecessor):
        self.weight = weight
        self.visited = False
        self.predecessor = predecessor

class adjacencyMatrix:

    def __init__(self, numberOfNodes):
        self.table = np.zeros((numberOfNodes,numberOfNodes))

    def addNode(self, nodeFrom, nodeTo, weight):

        self.table[nodeFrom][nodeTo] = weight
        self.table[nodeTo][nodeFrom] = weight

    def dijkstra(self, startNode):

        numberOfNodes = len(self.table)
        nextNode = 0
        currentNode = -1

        currentNode = startNode

        self.dijkstraTable = [dijkstraTableEntry(0,-1) for i in range(numberOfNodes)]
        
        while nextNode != -1:

            nextNode = -1

            for x in range(numberOfNodes):
                if self.table[currentNode][x] != 0 and not self.dijkstraTable[x].visited:

                    if self.dijkstraTable[x].weight > self.table[currentNode][x] + self.dijkstraTable[currentNode].weight:
                        self.dijkstraTable[x].weight = self.table[currentNode][x] + self.dijkstraTable[currentNode].weight
                        self.dijkstraTable[x].predecessor = currentNode

                    if nextNode == -1 or self.dijkstraTable[x].weight < self.dijkstraTable[nextNode].weight:
                        nextNode = x
            
            self.dijkstraTable[currentNode].visited = True
            currentNode = nextNode
        
        return self.dijkstraTable



        



