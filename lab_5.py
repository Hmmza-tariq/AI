class PriorityQueue:
    def __init__(self):
        self.elements = []

    def is_empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        self.elements.append((priority, item))

    def get(self):
        min_index = 0
        for i in range(1, len(self.elements)):
            if self.elements[i][0] < self.elements[min_index][0]:
                min_index = i
        item = self.elements[min_index]
        del self.elements[min_index]
        return item[1]

def a_star_search(graph, heuristics, start, goal):
    q = PriorityQueue()
    q.put(start, 0)
    came_from = {start: None}
    cost_so_far = {start: 0}
    while not q.is_empty():
        current_node = q.get()
        if current_node == goal:
            break
        for neighbor, cost in graph[current_node].items():
            new_cost = cost_so_far[current_node] + cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristics[neighbor]
                q.put(neighbor, priority)
                came_from[neighbor] = current_node

    return reconstruct_path(came_from, start, goal)

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

graph = {
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Rimnicu Vilcea': {'Craiova': 146, 'Sibiu': 80, 'Pitesti': 97},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

heuristics = {
    'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242,
    'Eforie': 161, 'Fagaras': 178, 'Giurgiu': 77, 'Hirsova': 151,
    'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234, 'Oradea': 380,
    'Pitesti': 98, 'Rimnicu Vilcea': 193, 'Sibiu': 253, 'Timisoara': 329,
    'Urziceni': 80, 'Vaslui': 199, 'Zerind': 374
}

start = 'Arad'
end = 'Bucharest'

path = a_star_search(graph, heuristics, start, end)
print("\n\nTask#1")
print("Optimal Path from Arad to Bucharest:", path)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_maze(maze, start, goal):
    q = PriorityQueue()
    q.put(start, 0)
    came_from = {start: None}
    cost_so_far = {start: 0}
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while not q.is_empty():
        current_node = q.get()
        
        if current_node == goal:
            break

        for move in moves:
            next_node = (current_node[0] + move[0], current_node[1] + move[1])

            if 0 <= next_node[0] < len(maze) and 0 <= next_node[1] < len(maze[0]) and maze[next_node[0]][next_node[1]] == 0:
                new_cost = cost_so_far[current_node] + 1 
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(goal, next_node)
                    q.put(next_node, priority)
                    came_from[next_node] = current_node

    return reconstruct_path(came_from, start, goal)


maze = [
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0]   
]

start = (4, 0)  
goal = (4, 4)   

path = a_star_maze(maze, start, goal)
print("\n\nTask#2")
print("Optimal Path:", path)