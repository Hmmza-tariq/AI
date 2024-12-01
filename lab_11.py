import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#task 1
data = np.array([
    [25, 40000, 0],
    [35, 60000, 0],
    [45, 80000, 0],
    [20, 20000, 0],
    [35, 120000, 0],
    [52, 18000, 0],
    [23, 95000, 1],
    [40, 62000, 1],
    [60, 100000, 1],
    [48, 220000, 1],
    [33, 150000, 1]
])
test_sample = np.array([48, 142000])

X = data[:, :2]
y = data[:, 2]

def gaussian_pdf(x, mean, std):
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

classes, class_counts = np.unique(y, return_counts=True)
priors = class_counts / len(y)
data_by_class = {c: X[y == c] for c in classes}
means_stds = {
    c: [(feature.mean(), feature.std()) for feature in data_by_class[c].T]
    for c in classes
}
posteriors = []
for c in classes:
    likelihood = 1
    for i, (mean, std) in enumerate(means_stds[c]):
        likelihood *= gaussian_pdf(test_sample[i], mean, std)
    posterior = likelihood * priors[c]
    posteriors.append(posterior)

posteriors = np.array(posteriors)
posteriors /= posteriors.sum()
predicted_class = classes[np.argmax(posteriors)]
print(f"Predicted class for {test_sample} is: {predicted_class}")

#task 2
data = pd.read_csv('files/diabetes.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

def gaussian_pdf(x, mean, std):
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.priors = class_counts / len(y)
        self.data_by_class = {c: X[y == c] for c in self.classes}
        self.means_stds = {
            c: [(feature.mean(), feature.std()) for feature in self.data_by_class[c].T]
            for c in self.classes
        }

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                likelihood = 1
                for i, (mean, std) in enumerate(self.means_stds[c]):
                    likelihood *= gaussian_pdf(x[i], mean, std)
                posterior = likelihood * self.priors[c]
                posteriors.append(posterior)
            posteriors = np.array(posteriors)
            posteriors /= posteriors.sum()
            predictions.append(self.classes[np.argmax(posteriors)])
        return predictions

nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")