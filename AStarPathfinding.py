import numpy as np
import heapq

class AStarPathfinding:
    def __init__(self, env):
        self.env = env
    
    def heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def find_path(self, start, goal):
        """
        A* pathfinding algorithm
        Returns the path from start to goal, or None if no path exists
        """
        start = tuple(start)
        goal = tuple(goal)
        
        # Priority queue: (f_score, g_score, position, path)
        open_set = [(0, 0, start, [start])]
        closed_set = set()
        
        # g_score: cost from start to current position
        g_scores = {start: 0}
        
        while open_set:
            current_f, current_g, current_pos, current_path = heapq.heappop(open_set)
            
            if current_pos in closed_set:
                continue
            
            closed_set.add(current_pos)
            
            # Check if we reached the goal
            if current_pos == goal:
                return current_path
            
            # Explore neighbors
            neighbors = self.env.getNeighbors(current_pos)
            
            for neighbor in neighbors:
                neighbor = tuple(neighbor)
                
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = current_g + 1
                
                # If this path to neighbor is better, record it
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h_score = self.heuristic(neighbor, goal)
                    f_score = tentative_g + h_score
                    
                    new_path = current_path + [neighbor]
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, new_path))
        
        # No path found
        return None