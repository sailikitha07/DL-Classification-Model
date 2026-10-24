# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## DESIGN STEPS
### STEP 1: Import Libraries – Import PyTorch, NumPy, and Scikit-learn libraries.



### STEP 2: Load Dataset – Load the Iris dataset containing four features and three flower species.



### STEP 3: Preprocess Data – Split the dataset into training and testing sets and normalize the features.



### STEP 4: Build Model – Create a neural network with two hidden layers using ReLU activation and three output neurons.



### STEP 5: Train Model – Train the model using CrossEntropyLoss and the Adam optimizer for 100 epochs.



### STEP 6: 1. Evaluate Model – Predict the flower species and calculate the model's test accuracy.


## PROGRAM

### Name: Cholimgapuram Sai Likitha

### Register Number: 212224230046

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

# 1. Load Dataset
iris = load_iris()
X, y = iris.data, iris.target
print("Dataset Information")
print("Shape:", X.shape)
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)
print("Samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=16, shuffle=True
)

# 2. Define Model
class IrisClassifier(nn.Module):
    def __init__(self, input_size):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 3. Initialize Model
model = IrisClassifier(X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Train Model
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

train_model(model, train_loader, criterion, optimizer, 100)

# 5. Evaluate Model
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predicted = outputs.argmax(dim=1)
    accuracy = (predicted == y_test).float().mean()

print("\nTest Accuracy:", round(accuracy.item() * 100, 2), "%")

# 6. Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test.numpy(), predicted.numpy()))

# 7. Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test.numpy(), predicted.numpy(),
    target_names=iris.target_names,
    zero_division=0
))

# 8. New Sample Prediction
# Features: sepal length, sepal width, petal length, petal width
new_sample = torch.tensor([[5.1, 3.5, 1.4, 0.2]], dtype=torch.float32)

with torch.no_grad():
    output = model(new_sample)
    prediction = output.argmax(dim=1).item()

print("\nNew Sample:", new_sample.tolist()[0])
print("Predicted Iris Species:", iris.target_names[prediction])


```

### Dataset Information
<img width="782" height="427" alt="image" src="https://github.com/user-attachments/assets/c9a8050a-3382-4ade-8fe6-0840e5df25ba" />

### OUTPUT
<img width="215" height="147" alt="image" src="https://github.com/user-attachments/assets/5f8c295d-a9da-4849-89e3-44b552d3dd40" />

## Confusion Matrix
<img width="472" height="392" alt="image" src="https://github.com/user-attachments/assets/492f5626-81f0-4afc-bcf1-90b7b284a8d0" />


<img width="112" height="57" alt="image" src="https://github.com/user-attachments/assets/1c757ff1-9ca1-4ec2-808b-ddeea6445049" />

## Classification Report
<img width="410" height="150" alt="image" src="https://github.com/user-attachments/assets/6fe47e57-6589-4f55-9e3b-6e96c3770fc8" />


### New Sample Data Prediction
<img width="520" height="262" alt="image" src="https://github.com/user-attachments/assets/4bf1c0d4-ad72-4f92-b724-75785a778e34" />


## RESULT
Thus, a neural network classification model was successfully developed and trained using PyTorch
