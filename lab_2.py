def print_graph(graph):
    for node, neighbors in graph.items():
        print(f"Node {node} -> {' -> '.join(str(neighbor) for neighbor in neighbors)}")
 
graph_1 = {
    1: [2,5],
    2: [1,3,5],
    3: [2,4],
    4: [3,5, 6],
    5: [1,2,4],
    6: [4]
}

graph_2 = {
    'A': ['B'],
    'B': ['C', 'D', 'E'],
    'C': ['E'],
    'D': ['E'],
    'E': ['F'],
    'F': [],
    'G': ['D']
}
print("Graph 1:")
print_graph(graph_1)
print('\n')

print("Graph 2:")
print_graph(graph_2)
print('\n')

# task 1.a
def print_degrees(graph):
    for node, neighbors in graph.items():
        print(f"Node {node}: Degree {len(neighbors)}")

print("Task 1.a:")
print_degrees(graph_1)
print('\n\n')

# task 1.b

def print_in_out_degree(graph):
    for node, neighbors in graph.items():
        in_degree = 0
        out_degree = len(neighbors)
        for myNode, myNeighbors in graph.items():
            if node in myNeighbors:
                in_degree += 1
        print(f"Node {node}: In-degree {in_degree}, Out-degree {out_degree}")

print("Task 1.b:")
print_in_out_degree(graph_2)
print('\n\n')

# task 1.c

def find_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    for node in graph[start]:
        if node not in path:
            new_path = find_path(graph, node, end, path)
            if new_path:
                return new_path
    return None

print("Task 1.c:")
print(f"Any path from node 6 to node 1: {find_path(graph_1, 6, 1)}")
print("\n\n")

# task 1.d

print("Task 1.d:")
print(f"Any path from node A to node F: {find_path(graph_2, 'A', 'F')}")
print("\n\n")

# task 1.e
def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    paths = []
    for node in graph[start]:
        if node not in path:
            new_paths = find_all_paths(graph, node, end, path)
            for new_path in new_paths:
                paths.append(new_path)
    return paths

print("Task 1.e:")
print("All paths from node 6 to node 1: \n" + '\n'.join(str(path) for path in find_all_paths(graph_1, 6, 1)))
print("\n\n")

# task 1.f
print("Task 1.f:")
print("All paths from node A to node F: \n" + '\n'.join(str(path) for path in find_all_paths(graph_2, 'A', 'F')))
print("\n\n")


# task 1.g
 
def print_adj_list(graph):
    for node, neighbors in graph.items():
      print(f"{node}: {neighbors}")

print("Task 1.g:")
print("Adjacency list for graph 1:")
print_adj_list(graph_1)
print('\n')
print("Adjacency list for graph 2:")
print_adj_list(graph_2)
print('\n\n')    

# task 2.a

image_pixels = [
    [100, 110, 120, 130],
    [140, 145, 45, 135],
    [220, 10, 165, 80],
    [180, 200, 191, 118]
]

def create_graph_from_image(pixels):
    rows = len(pixels)
    cols = len(pixels[0])
    graph = {}
    
    for i in range(rows):
        for j in range(cols):
            node = pixels[i][j]
            neighbors = []
            if i > 0: neighbors.append(pixels[i-1][j]) 
            if i < rows-1: neighbors.append(pixels[i+1][j]) 
            if j > 0: neighbors.append(pixels[i][j-1]) 
            if j < cols-1: neighbors.append(pixels[i][j+1]) 
            graph[node] = neighbors
    return graph

image_graph = create_graph_from_image(image_pixels)
print("Task 2.a:")
print("Graph created from image pixels:" + '\n' + '\n'.join(f"{node}: {neighbors}" for node, neighbors in image_graph.items()))
print("\n\n")

# task 2.b
all_paths_100_to_118 = find_all_paths(image_graph, 100, 118)
print("Task 2.b:")
print("All paths from pixel 100 to pixel 118: \n" + '\n'.join(str(path) for path in all_paths_100_to_118))
print("\n\n")

# task 3
def find_shortest_path(graph, start, end, path=[], cost=0):
    path = path + [start]

    if start == end:
        return path, cost
    shortest = None
    min_cost = float('inf')   
    for node in graph[start]:
        if node not in path:
            edge_cost = abs(start - node) 
            new_path, new_cost = find_shortest_path(graph, node, end, path, cost + edge_cost)
            
            if new_path:
                if new_cost < min_cost:
                    shortest = new_path
                    min_cost = new_cost

    return shortest, min_cost

print("Task 3:")
shortest_path_100_to_118, min_cost = find_shortest_path(image_graph, 100, 118)
print(f"Shortest path from pixel 100 to pixel 118: {shortest_path_100_to_118} with cost {min_cost}")
print("\n\n")