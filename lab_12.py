import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def perceptron_training(data, labels, learning_rate=0.1, threshold=0.5, max_iterations=1000):    
    weights = np.zeros(data.shape[1])
    for _ in range(max_iterations):
        errors = 0
        correct_predictions = 0
        for i in range(len(data)):
            weighted_sum = np.dot(data[i], weights) 
            if weighted_sum >= threshold:
                output = 1
            else:
                output = 0         
            error = labels[i] - output
            if error != 0:
                errors += 1
            else:
                correct_predictions += 1
            weights += learning_rate * error * data[i]
        
        
        training_accuracy = correct_predictions / len(data)
        if training_accuracy >= 0.75:
            break

    return weights

def predict(data, weights, threshold=0.5):
    predictions = []
    for i in range(len(data)):
        weighted_sum = np.dot(data[i], weights)
        output = 1 if weighted_sum >= threshold else 0
        predictions.append(output)
    return np.array(predictions)


data = np.array([
    [1, 0, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
])
inputs = data[:, :-1]
labels = data[:, -1]

weights = perceptron_training(inputs, labels)
print("Task 1")
print("Trained weights:", weights)

print("Task 2")
sonar_data = pd.read_csv('files/sonar.all-data.csv', header=None)
features = sonar_data.iloc[:, :-1].values
labels = np.where(sonar_data.iloc[:, -1] == 'R', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
weights = perceptron_training(X_train, y_train)
y_pred = predict(X_test, weights)
test_accuracy_score = accuracy_score(y_test, y_pred)

print("Testing Accuracy:", test_accuracy_score)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

plt.figure(figsize=(5, 5))
plt.imshow(conf_matrix, cmap='Greens', interpolation='nearest')
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['M', 'R'])
plt.yticks([0, 1], ['M', 'R'])
plt.show()

print("Trained weights:", weights)
