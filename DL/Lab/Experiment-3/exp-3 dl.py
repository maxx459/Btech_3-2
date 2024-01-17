
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Setting Seaborn Style and Displaying TensorFlow Version
sns.set()
tf.__version__

# Load the dataset
data = pd.read_csv("/content/heart.csv")

# Separate features (X) and target variable (y)
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model with dropout
model = Sequential([
    Dense(300, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(150, activation='relu'),
    Dense(75, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert labels to 0 or 1 for binary classification
y_train_binary = y_train.astype('float32')

# Train the model
history = model.fit(
    X_train_scaled, y_train_binary, epochs=10,
    verbose=2
)

# Displaying the summary of the model
model.summary()

# Print layer information
print("List of layers:", model.layers)
print("\nName of the second layer:", model.layers[1].name)

# Accessing the second hidden layer and printing weights and bias
hidden2 = model.layers[2]
weights, bias = hidden2.get_weights()
print("Weights:", weights)
print("Bias:", bias)

# Plotting the training history
pd.DataFrame(history.history).plot()
plt.xlabel("Epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

# Evaluate the model on the test set
loss, acc = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", acc)
