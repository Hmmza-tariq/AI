import math

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):  
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

def get_neighbors(trainingSet, testRow, k):
    distances = []
    for train_row in trainingSet:
        dist = euclidean_distance(testRow, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])  
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def predict_classification(trainingSet, testRow, k):
    neighbors = get_neighbors(trainingSet, testRow, k)
    output_values = [row[-1] for row in neighbors]  
    prediction = max(set(output_values), key=output_values.count)
    return prediction

dataset = [
    [2.7810836, 2.550537003, 0],
    [1.465489372, 2.362125076, 0],
    [3.396561688, 4.400293529, 0],
    [1.38807019, 1.850220317, 0],
    [3.06407232, 3.005305973, 0],
    [7.627531214, 2.759262235, 1],
    [5.332441248, 2.088626775, 1],
    [6.922596716, 1.77106367, 1],
    [8.675418651, -0.242068655, 1],
    [7.673756466, 3.508563011, 1]
]

# task 1
test_row = dataset[0]
train_set = dataset[1:]
k = 3
prediction = predict_classification(train_set, test_row, k)
print(f'Predicted Class for the first row: {prediction}')

# task 2
correct = 0
for row in dataset:
    predicted = predict_classification([x for x in dataset if x != row], row, k) 
    print(f'True Class: {row[-1]}, Predicted Class: {predicted}')
    if predicted == row[-1]:
        correct += 1
accuracy = correct / len(dataset) * 100
print(f'Accuracy of KNN: {accuracy:.2f}%')


# task 3
import csv
import numpy as np

file = open('files/fruit_data_with_colours.csv')
csvreader = csv.reader(file)
csv_dataset = []
for row in csvreader:
    csv_dataset.append(row)

csv_dataset.remove(csv_dataset[0])  
csv_dataset = np.array(csv_dataset, dtype=float)
first_csv_row = csv_dataset[0]
csv_train_set = csv_dataset[1:]
predicted_csv = predict_classification(csv_train_set, first_csv_row, k)
print(f'Predicted Class for the first CSV row: {predicted_csv}')

random_csv_row = csv_dataset[3]
predicted_csv = predict_classification(csv_train_set, random_csv_row, k)
print(f'Predicted Class for the random CSV row: {predicted_csv}')
