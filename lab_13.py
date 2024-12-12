import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_xor():
    # XOR Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Network parameters
    input_neurons = 2
    hidden_neurons = 2
    output_neurons = 1
    learning_rate = 0.1
    epochs = 10000

    # Initialize weights and biases
    W1 = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
    b1 = np.zeros((1, hidden_neurons))
    W2 = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
    b2 = np.zeros((1, output_neurons))

    for epoch in range(epochs):
        # Forward pass
        Z1 = np.dot(X, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)

        # Compute error
        error = A2 - y

        # Backward pass
        dA2 = error * sigmoid_derivative(A2) # output layer error
        dA1 = np.dot(dA2, W2.T) * sigmoid_derivative(A1) # hidden layer error

        # Gradients
        dW2 = np.dot(A1.T, dA2)
        db2 = np.sum(dA2, axis=0, keepdims=True)
        dW1 = np.dot(X.T, dA1)
        db1 = np.sum(dA1, axis=0, keepdims=True)

        # Update weights and biases
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        if epoch % 1000 == 0:
            predictions = (A2 > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            print(f"Epoch {epoch}: Accuracy = {accuracy:.2f}")

    print("Training completed!")
    print(f"Final predictions: {A2}")
    print(f"Actual outputs: {y}")

train_xor()

def train_iris():
    # Load the Iris dataset
    data = pd.read_csv("files/iris.csv")
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values  # Target

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # One-hot encode the target
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Network parameters
    input_neurons = X_train.shape[1]
    hidden_neurons1 = 4
    hidden_neurons2 = 4
    output_neurons = y_train.shape[1]
    learning_rate = 0.1
    epochs = 10000

    # Initialize weights and biases
    W1 = np.random.uniform(-1, 1, (input_neurons, hidden_neurons1))
    b1 = np.zeros((1, hidden_neurons1))
    W2 = np.random.uniform(-1, 1, (hidden_neurons1, hidden_neurons2))
    b2 = np.zeros((1, hidden_neurons2))
    W3 = np.random.uniform(-1, 1, (hidden_neurons2, output_neurons))
    b3 = np.zeros((1, output_neurons))

    for epoch in range(epochs):
        
        # Forward pass
        Z1 = np.dot(X_train, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)
        Z3 = np.dot(A2, W3) + b3
        A3 = sigmoid(Z3)

        # Compute error
        error = A3 - y_train

        # Backward pass
        dA3 = error * sigmoid_derivative(A3)
        dA2 = np.dot(dA3, W3.T) * sigmoid_derivative(A2)
        dA1 = np.dot(dA2, W2.T) * sigmoid_derivative(A1)

        # Gradients
        dW3 = np.dot(A2.T, dA3)
        db3 = np.sum(dA3, axis=0, keepdims=True)
        dW2 = np.dot(A1.T, dA2)
        db2 = np.sum(dA2, axis=0, keepdims=True)
        dW1 = np.dot(X_train.T, dA1)
        db1 = np.sum(dA1, axis=0, keepdims=True)

        # Update weights and biases
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        if epoch % 1000 == 0:
            predictions = np.argmax(A3, axis=1)
            actual = np.argmax(y_train, axis=1)
            accuracy = np.mean(predictions == actual)
            print(f"Epoch {epoch}: Training Accuracy = {accuracy:.2f}")

    print("Training completed!")

    # Testing the model
    Z1 = np.dot(X_test, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)

    test_predictions = np.argmax(A3, axis=1)
    test_actual = np.argmax(y_test, axis=1)
    test_accuracy = np.mean(test_predictions == test_actual)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    
train_iris()
