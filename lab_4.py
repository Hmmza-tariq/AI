#Task 1
class Stack:
    def __init__(self):
        self.items = []


    def push(self, item):
        self.items.append(item)

    def pop(self):
        if len(self.items) != 0:
            return self.items.pop()
        raise IndexError("pop from empty stack")

def dfs_recursive(graph, start, stack = None):
    if stack is None:
        stack = Stack()
    stack.push(start)
    print(start, end=' ')
    for neighbor in graph[start]:
        if neighbor not in stack.items:
            dfs_recursive(graph, neighbor, stack)

def dfs_iterative(graph, start):
    viewed = []
    stack = Stack()
    stack.push(start)

    while stack.items != []:
        vertex = stack.pop()
        if vertex not in viewed:
            print(vertex, end=' ')
            viewed.append(vertex)
            for neighbor in reversed(graph[vertex]):  
                if neighbor not in viewed:
                    stack.push(neighbor)


graph1 = {
    '1': ['2', '5'],
    '2': ['1', '3' ,'5'],
    '3': ['2', '4'],
    '4': ['3', '5', '6'],
    '5': ['1', '2' ,'4'],
    '6': ['4'],
}

graph2 = {
    'A': ['B', 'D' ,'E'],
    'B': ['A', 'D' ,'E'],
    'C': ['D'],
    'D': ['A', 'B', 'C'],
    'E': ['A', 'B'],
}

print("\n\nTask#1")
print("DFS Recursive on Graph 1 from node 6:")
dfs_recursive(graph1, '6')
print("\nDFS Iterative on Graph 1 from node 6:")
dfs_iterative(graph1, '6')

print("\nDFS Recursive on Graph 2 from node E:")
dfs_recursive(graph2, 'E')
print("\nDFS Iterative on Graph 2 from node E:")
dfs_iterative(graph2, 'E')

class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def tree1():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.left.left.left = TreeNode(6)
    root.left.right.left = TreeNode(7)
    root.left.right.right = TreeNode(8)
    return root

def tree2():
    root = TreeNode(50)
    root.left = TreeNode(17)
    root.right = TreeNode(76)
    root.left.left = TreeNode(9)
    root.left.right = TreeNode(23)
    root.right.left = TreeNode(54)
    root.left.left.left = TreeNode(14)
    root.left.right.left = TreeNode(19)
    root.right.left.left = TreeNode(72)
    root.left.left.left.left = TreeNode(12)
    root.right.left.left.left = TreeNode(67)
    return root

def pre_order(node):
    if node is None:
        return
    print(node.data, end=" ")
    pre_order(node.left)
    pre_order(node.right)

def in_order(node):
    if node is None:
        return
    in_order(node.left)
    print(node.data, end=" ")
    in_order(node.right)

def post_order(node):
    if node is None:
        return
    post_order(node.left)
    post_order(node.right)
    print(node.data, end=" ")

print("\n\nTask#2")
root1 = tree1()
print("Pre-Order Traversal for Tree 1:")
pre_order(root1)
print("\nIn-Order Traversal for Tree 1:")
in_order(root1)
print("\nPost-Order Traversal for Tree 1:")
post_order(root1)

root2 = tree2()
print("\nPre-Order Traversal for Tree 2:")
pre_order(root2)
print("\nIn-Order Traversal for Tree 2:")
in_order(root2)
print("\nPost-Order Traversal for Tree 2:")
post_order(root2)


