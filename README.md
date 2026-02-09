# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Develop a Convolutional Neural Network (CNN) to classify images into predefined categories. The dataset consists of labeled images split into training and testing sets, with preprocessing steps like resizing, normalization, and augmentation to enhance model performance. The trained model will be tested on new images to verify accuracy.


## Neural Network Model

<img width="1183" height="467" alt="425547172-cb131631-9bba-4dc8-a3c8-dd7a9b3c98ba" src="https://github.com/user-attachments/assets/4bb8cd96-bf53-4665-9f77-25e96fec1121" />


## DESIGN STEPS

### STEP 1:
Import the necessary libraries such as NumPy, Matplotlib, and PyTorch.

### STEP 2:
Load and preprocess the dataset:

Resize images to a fixed size (128×128).
Normalize pixel values to a range between 0 and 1.
Convert labels into numerical format if necessary.
### STEP 3:
Define the CNN Architecture, which includes:

Input Layer: Shape (8,128,128)
Convolutional Layer 1: 8 filters, kernel size (16×16), ReLU activation
Max-Pooling Layer 1: Pool size (2×2)
Convolutional Layer 2: 24 filters, kernel size (8×8), ReLU activation
Max-Pooling Layer 2: Pool size (2×2)
Fully Connected (Dense) Layer:
First Dense Layer with 256 neurons
Second Dense Layer with 128 neurons
Output Layer for classification
### STEP 4:
Define the loss function (e.g., Cross-Entropy Loss for classification) and optimizer (e.g., Adam or SGD).

### STEP 5:
Train the model by passing training data through the network, calculating the loss, and updating the weights using backpropagation.

### STEP 6:
Evaluate the trained model on the test dataset using accuracy, confusion matrix, and other performance metrics.

### STEP 7:
Make predictions on new images and analyze the results.


## PROGRAM

### Name:MEENAKSHI.R
### Register Number:212224220062
```
class CNNClassifier(nn.Module):
    def __init__(self): # Changed _init_ to __init__
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0), -1)
        x =torch.relu(self.fc1(x))
        x =torch.relu(self.fc2(x))
        return x
# Initialize the Model, Loss Function, and Optimizer

model = CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.01)

# Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images,labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


        print('Name:MEENAKSHI.R')
        print('Register Number:212224220062')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
train_model(model, train_loader)
```

## OUTPUT
### Training Loss per Epoch



### Confusion Matrix
```
Name:MEENAKSHI.R
Register Number:212224220062
Test Accuracy: 0.8415
Name:MEENAKSHI.R        
Register Number:212224220062
```
<img width="709" height="608" alt="download" src="https://github.com/user-attachments/assets/65508424-1c8c-48c3-8554-4f3c77d1eb2f" />

### Classification Report




### New Sample Data Prediction
```
Name: MEENAKSHI.R       
Register Number: 212224220062
```
<img width="389" height="432" alt="download" src="https://github.com/user-attachments/assets/c8d91b72-cc71-4415-b190-c28a09fe5646" />
```
Actual: Trouser, Predicted: Trouser
```

## RESULT
The Convolutional Neural Network (CNN) was successfully implemented for image classification. The model was trained on the dataset, and its performance was evaluated using accuracy metrics, confusion matrix, and classification report. Predictions on new sample images were verified, confirming the model's effectiveness in classifying images.
