class dijkstraTableEntry:
    def __init__(self):
        self.weight = 99999999
        self.visited = False
        self.predecessor = -1

class adjacencyMatrix:

    def __init__(self, numberOfNodes):
        self.table = np.zeros((numberOfNodes,numberOfNodes))
        self.dijkstraTable = [dijkstraTableEntry() for i in range(numberOfNodes)]

    def addNode(self, nodeFrom, nodeTo, weight):

        self.table[nodeFrom][nodeTo] = weight
        self.table[nodeTo][nodeFrom] = weight

    def dijkstra(self, startNode, numberOfNodes):
        nodeQueue = []
        self.dijkstraTable[startNode].weight = 0

        nodeQueue.append(startNode)

        for x in range(numberOfNodes):
            if x != startNode:
                self.dijkstraTable[x].predecessor = -1
                self.dijkstraTable[x].weight = 9999999
                nodeQueue.insert(0,x)
        
        while len(nodeQueue) > 0:
            currentNode = nodeQueue.pop()
            print(currentNode)
            for x in range(numberOfNodes):
                if self.table[currentNode][x] != 0:
                    if self.table[currentNode][x] + self.dijkstraTable[currentNode].weight < self.dijkstraTable[x].weight:
                        self.dijkstraTable[x].weight = self.table[currentNode][x] + self.dijkstraTable[currentNode].weight
                        self.dijkstraTable[x].predecessor = currentNode
                        print(currentNode ," ",x, " ", self.dijkstraTable[x].predecessor)
        



