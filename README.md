 # Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load the Iris dataset, split it into train-test sets, standardize features, and convert them to tensors.

### STEP 2: 
Use TensorDataset and DataLoader for efficient data handling.

### STEP 3: 
Define IrisClassifier with input, hidden (ReLU), and output layers.

### STEP 4: 
Train using CrossEntropyLoss and Adam optimizer for 100 epochs.

### STEP 5: 
Evaluate with accuracy, confusion matrix, and classification report, and visualize results.

### STEP 6: 
Predict a sample input and compare with the actual class.

## PROGRAM

**Name :** AMALJOSH MAADHAV J
**Register Number :** 212223230012

```python
class IrisClassifier(nn.Module):
    def __init__(self, input_size):
        super(IrisClassifier, self).__init__()
        #Include your code here

    def forward(self, x):
        #Include your code here

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

iris=load_iris()
x=iris.data
y=iris.target

df=pd.DataFrame(x,columns=iris.feature_names)
df['target']=y

print("\nFirst 5 rows of the dataset:\n",df.head())
print("\nLast 5 rows of the dataset:\n",df.tail())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

x_train=torch.tensor(x_train,dtype=torch.float32)
x_test=torch.tensor(x_test,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.long)
y_test=torch.tensor(y_test,dtype=torch.long)

train_dataset=TensorDataset(x_train, y_train)
test_dataset=TensorDataset(x_test, y_test)
train_loader=DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=16)

class IrisClassifier(nn.Module):
  def __init__(self,input_size):
    super(IrisClassifier,self).__init__()
    self.fc1=nn.Linear(input_size, 16)
    self.fc2=nn.Linear(16, 8)
    self.fc3=nn.Linear(8, 3)

  def forward(self, x):
    x=F.relu(self.fc1(x))
    x=F.relu(self.fc2(x))
    return self.fc3(x)

def train_model(model, train_loader, criterion, optimizer, epochs):
  for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(x_batch)
      loss=criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
    if (epoch + 1) % 10 == 0:
      print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

model = IrisClassifier(input_size=x_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, epochs=100)

model.eval()
predictions, actuals = [], []
with torch.no_grad():
  for x_batch, y_batch in test_loader:
    outputs = model(x_batch)
    _, predicted = torch.max(outputs, 1)
    predictions.extend(predicted.numpy())
    actuals.extend(y_batch.numpy())

accuracy=accuracy_score(actuals, predictions)
confusion_matrix=confusion_matrix(actuals, predictions)
classification_report=classification_report(actuals, predictions)

print("Name: AMALJOSH MAADHAV J")
print("Register Number: 212223230012")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", confusion_matrix)
print("Classification Report:\n", classification_report)

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix, annot=True,cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

sample_input = x_test[5].unsqueeze(0)
with torch.no_grad():
  output = model(sample_input)
  predicted_class_index = torch.argmax(output[0]).item()
  predicted_class_label = iris.target_names[predicted_class_index]

print("\nName: AMALJOSH MAADHAV J")
print("\nRegister Name: 212223230012")
print(f"Predicted class for sample input: {predicted_class_label}")
print(f"Actual class for sample input: {iris.target_names[y_test[5].item()]}")

```

### Dataset Information
![Screenshot 2025-03-27 142635](https://github.com/user-attachments/assets/2727f8d3-92b7-4b40-b2c6-ee8e35893ea8)
![Screenshot 2025-03-27 142646](https://github.com/user-attachments/assets/a86b0788-2456-4f70-8577-81a8337a7b8f)

### OUTPUT

## Confusion Matrix

![Screenshot 2025-03-27 142753](https://github.com/user-attachments/assets/653f10c4-2355-409b-aa92-b7c5e5f60b51)

## Classification Report
![Screenshot 2025-03-27 142818](https://github.com/user-attachments/assets/9a96bd0f-bbb4-4630-98e3-9b9f270fc4c1)

### New Sample Data Prediction
![Screenshot 2025-03-29 103609](https://github.com/user-attachments/assets/786da58e-0c99-49b4-a085-8f572681c524)

### RESULT
Thus, a neural network classification model has been successfully built.
