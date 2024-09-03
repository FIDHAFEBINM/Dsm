import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset
# Let's create a dataset with 2 features and 3 classes (similar to the Iris dataset)

# Number of samples per class
n_samples = 50

# Class 0: centered at (0, 0)
class_0 = np.random.randn(n_samples, 2) + np.array([0, 0])

# Class 1: centered at (3, 3)
class_1 = np.random.randn(n_samples, 2) + np.array([3, 3])

# Class 2: centered at (0, 5)
class_2 = np.random.randn(n_samples, 2) + np.array([0, 5])

# Combine the classes to create the dataset
X = np.vstack((class_0, class_1, class_2))
y = np.array([0]*n_samples + [1]*n_samples + [2]*n_samples)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the k-NN algorithm classifier with a specified value of k
k = 5  # You can change this value as needed
knn = KNeighborsClassifier(n_neighbors=k)

# Fit the model on the training data
knn.fit(X_train, y_train)

# Make the predictions on the test data
y_pred = knn.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of k-NN with k={k}: {accuracy*100:.2f}%')
