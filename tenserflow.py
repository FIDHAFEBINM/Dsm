import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sn 

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() 
x_train = x_train / 255.0
x_test = x_test / 255.0 

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="sigmoid")  # Fixed the activation function
]) 

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # Fixed the optimizer

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test) 

y_predicted = model.predict(x_test)
print((np.argmax(y_predicted[1])))

y_predicted_labels = [np.argmax(i) for i in y_predicted]
print(y_predicted_labels[:5])

cn = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
plt.figure(figsize=(10, 7))
sn.heatmap(cn, annot=True, fmt="d")  # Fixed the heatmap parameters
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()  # Added parentheses to show the plot
