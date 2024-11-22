import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from math import sqrt, pi, exp

def fit(X_train, y_train):
    train_positive = X_train[y_train == 1]
    train_negative = X_train[y_train == 0]
 
    prior_positive = len(train_positive) / len(X_train) 
    prior_negative = len(train_negative) / len(X_train) 
 
    mean_positive = np.array(train_positive.mean()) 
    mean_negative = np.array(train_negative.mean()) 
 
    covariance_positive = np.cov(train_positive, rowvar=False) 
    covariance_negative = np.cov(train_negative, rowvar=False) 

    values= {'prior_p':prior_positive,'prior_n':prior_negative,
            'cov_p':covariance_positive, 'cov_n':covariance_negative,
            'mean_p':mean_positive,'mean_n':mean_negative} 
    return values   
   
def predict(values, X_test):
    probabilities_positive = multivariate_normal.pdf(X_test, mean=values['mean_p'], cov=values['cov_p'])
    probabilities_negative = multivariate_normal.pdf(X_test, mean=values['mean_n'], cov=values['cov_n'])
 
    posterior_positive = probabilities_positive * values['prior_p']
    posterior_negative = probabilities_negative * values['prior_n']
 
    evidence= posterior_positive+posterior_negative
 
    posterior_positive = posterior_positive/evidence
    posterior_negative = posterior_negative/evidence
 
    predictions = (posterior_positive > posterior_negative).astype(int)
    return predictions

# Task 1 (a)
dataset = pd.read_csv("files/diabetes.csv")
 
np_array = dataset.to_numpy()
dataset = pd.DataFrame(np_array, columns=dataset.columns)
X = dataset.drop('Outcome', axis=1)  
y = dataset['Outcome']  
 
split_index = int(0.5 * len(dataset))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
values=fit(X_train,y_train)

predictions= predict(values,X_test)
actual_outcome=y_test.values.astype(int)
print(f'when first 50% of the data is used for training and the rest for testing')
print("Accuracy ",np.sum(predictions == actual_outcome)/ len(actual_outcome))

# Task 1 (b)

X_train, X_test = X.iloc[split_index:], X.iloc[:split_index]
y_train, y_test = y.iloc[split_index:], y.iloc[:split_index]
values=fit(X_train,y_train)

predictions= predict(values,X_test)
actual_outcome=y_test.values.astype(int)
print(f'when last 50% of the data is used for training and the rest for testing')
print("Accuracy ",np.sum(predictions == actual_outcome)/ len(actual_outcome))

# Task 2

data = [
    (2, 0), (1, 0), (3, 0), (1.5, 0), (4.1, 0), 
    (6, 1), (5, 1), (7, 1), (8, 1), (7.6, 1)
]
test_sample = 5  

class_0 = [x[0] for x in data if x[1] == 0]
class_1 = [x[0] for x in data if x[1] == 1]

prior_0 = len(class_0) / len(data)
prior_1 = len(class_1) / len(data)

mean_0, var_0 = np.mean(class_0), np.var(class_0)
mean_1, var_1 = np.mean(class_1), np.var(class_1)

def gaussian_likelihood(x, mu, var):
    return (1 / sqrt(2 * pi * var)) * exp(-(x - mu)**2 / (2 * var))

likelihood_0 = gaussian_likelihood(test_sample, mean_0, var_0)
likelihood_1 = gaussian_likelihood(test_sample, mean_1, var_1)

posterior_0 = likelihood_0 * prior_0
posterior_1 = likelihood_1 * prior_1

evidence = posterior_0 + posterior_1
posterior_0 /= evidence
posterior_1 /= evidence

predicted_class = 0 if posterior_0 >= posterior_1 else 1

print(f"Predicted Class: {predicted_class}")
