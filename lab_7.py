import time

class Node:
    def __init__(self, value=None):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

def minimax(node, depth, maximizingPlayer):
    if depth == 0 or not node.children:
        return node.value

    if maximizingPlayer:
        bestValue = float('-inf')
        for child in node.children:
            value = minimax(child, depth - 1, False)
            bestValue = max(bestValue, value)
        return bestValue
    else:
        bestValue = float('inf')
        for child in node.children:
            value = minimax(child, depth - 1, True)
            bestValue = min(bestValue, value)
        return bestValue

def alphabeta(node, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or not node.children:
        return node.value

    if maximizingPlayer:
        value = float('-inf')
        for child in node.children:
            value = max(value, alphabeta(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value
    else:
        value = float('inf')
        for child in node.children:
            value = min(value, alphabeta(child, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value


root = Node(-7)
root.left = Node(-10)
root.right = Node(-7)
root.left.left = Node(10)
root.left.right = Node(-10)
root.right.left = Node(5)
root.right.right = Node(-7)

root.left.left.left = Node(10)
root.left.left.right = Node(5)
root.left.right.left = Node(-10)

root.right.left.left = Node(5)
root.right.left.right = Node(float('-inf'))
root.right.right.left = Node(-7)

root.left.left.left.left = Node(10)
root.left.left.left.right = Node(float('inf'))
root.left.left.right.left = Node(5)
root.left.right.left.left = Node(-10)

root.right.left.left.left = Node(7)
root.right.left.left.right = Node(5)
root.right.left.right.left = Node(float('-inf'))
root.right.right.left.left = Node(-7)
root.right.right.left.right = Node(-5)


start_time = time.time()
minimax_value = minimax(root, 3, True)
end_time = time.time()
print(f"Minimax value: {minimax_value}, Time taken: {end_time - start_time} seconds")


start_time = time.time()
alphabeta_value = alphabeta(root, 3, float('-inf'), float('inf'), True)
end_time = time.time()
print(f"Alpha-Beta value: {alphabeta_value}, Time taken: {end_time - start_time} seconds")
