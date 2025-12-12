# AI_Group_Project
This project uses both a search algorithm and machine learning (Reinforcement Learning) to find the shortest distance between two points on a map to aid with campus navigation.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Structure](#structure)
- [Enviromnent](#enviromnent)
- [Search Algorithm](#search-algorithm)
- [Contributers](#contributers)

## Requirements
- Python 3.10+
- Gymnasium
- Numpy 
- Pygame

## Installation
Install dependencies:
```bash
pip install gymnasium numpy pygame
```

## Structure
- `AStarPathfinding.py` – search algorithm
- `CampusMapRun.py` – runs the program
- `CampusTraversalEnvironment.py` – custom environment
- `NewRLModel.py` – reinforcement learning

## Enviromnent
Action Space
- Right - move right
- Up - move up
- Left - move left
- Down - move down

Observation Space
- Current location
- Target location

## Search Algorithm
- Implements the A (A-star) search algorithm* to find the shortest path on the campus map grid using Manhattan distance as an admissible heuristic.
- Treats walkable (green) pixels as nodes with 4-directional movement; guarantees optimal and complete paths.
- Runs instantly on the large map and draws the path as a smooth red line.
- In the program: Choose option 2 after selecting start (red dot) and end (blue dot) to see the optimal path displayed.

## Reinforcement learning Algorithm
- Implements tabular Q-learning, a model-free reinforcement learning method that trains a Q-table to learn an optimal policy for navigating the campus map.
- Trains on random episodes (default 250) with ε-greedy exploration; rewards encourage reaching the goal quickly while penalising invalid moves.
- Uses a hybrid path extraction: BFS guided by Q-values for shorter paths, with greedy fallback; draws the learned path as a red line.
- In the program: Choose option 1 after selecting start (red dot) and end (blue dot)—it trains quickly then displays the learned path.

## Contributers
- Callum Firth 2635930
- Jonny Forbes 2643497
- Bailey Clark 2636229
- Logan Howie 2639383


