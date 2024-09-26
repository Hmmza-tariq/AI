from collections import deque

graph = {
    'A': ['B', 'D', 'E'],
    'B': ['A', 'D', 'E'],
    'C': ['D'],
    'D': ['A', 'B', 'C'],
    'E': ['A', 'B'],
}

def bfs(graph, start):
    visited = [] 
    queue = deque([start]) 
    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.append(current)
            for node in graph[current]:
                if node not in visited:
                    queue.append(node)

    print(' -> '.join(visited))

print("\n\nTask#1\nBFS traversal starting from 'C':")
bfs(graph, 'C')

class Tree:
    def __init__(self, data):
        self.left = None
        self.center = None
        self.right = None
        self.data = data

root = Tree('1')
root.left = Tree('2')
root.center = Tree('3')
root.right = Tree('4')
root.left.left = Tree('5')
root.left.center = Tree('6')
root.left.right = None
root.center.left = Tree('7')
root.center.center = Tree('8')
root.center.right = None
root.right.left = None
root.right.center = None
root.right.right = None
root.left.left.left = Tree('9')
root.left.left.center = None
root.left.left.right = Tree('10')
root.center.left.left = Tree('11')
root.center.left.center = None
root.center.left.right = Tree('12')


def bfs_tree(root):
    queue = deque([root])
    visited = []
    while queue:
        current = queue.popleft()
        if current:
            visited.append(current.data)
            if current.left:
                queue.append(current.left)
            if current.center:
                queue.append(current.center)
            if current.right:
                queue.append(current.right)

    print(' -> '.join(visited))

print("\n\nTask#2\nBFS traversal of the tree:")
bfs_tree(root)
 
def bfs_maze(maze):
    rows, cols = len(maze), len(maze[0])
    queue = deque([(0, 0, 0)])   
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  
    visited = set((0, 0))

    while queue:
        row, col, dist = queue.popleft()
        if row == rows - 1 and col == cols - 1:
            return dist  
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols and (new_row, new_col) not in visited and maze[new_row][new_col] == 0:
                visited.add((new_row, new_col))
                queue.append((new_row, new_col, dist + 1))

    return -1   

maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0]
]

print("\n\nTask#3\nShortest path length:", bfs_maze(maze))