import cv2
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt
from tqdm import tqdm


# IMG_SIZE = 128
IMG_SIZE = 64
BATCH_SIZE = 110

TRANSFORMATIONS = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BOXES = [
    [502, 344, 61, 86],
    [627, 427, 627 + 193, 427 + 93],
    [522, 654, 522 + 90, 654 + 85],
    [704, 594, 704 + 90, 594 + 80],
    [545, 577, 545 + 81, 577 + 75],
    [460, 700, 52, 80]
]

BOXIMAGES = [
    cv2.cvtColor(cv2.imread('BASE/BOX1.jpg'), cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(cv2.imread('BASE/BOX2.jpg'), cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(cv2.imread('BASE/BOX3.jpg'), cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(cv2.imread('BASE/BOX4.jpg'), cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(cv2.imread('BASE/BOX5.jpg'), cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(cv2.imread('BASE/BOX6_1.jpg'), cv2.COLOR_BGR2GRAY),
]

THRESHOLDS = [50, 60, 40, 30, 40, 40]

class Net(nn.Module):
    
    def __init__(self, in_channels=3, n_classes=2):
        super(Net, self).__init__()
        self.train_losses = []
        self.train_acc = []
        
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        self.network = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, stride=2),
            
            nn.Flatten(),
            
            nn.Linear(2048, 4096), nn.ReLU(), nn.Dropout(0.5),
            
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            
            nn.Linear(4096, self.n_classes)   
        )


    def forward(self, x):
        return self.network(x)

    
    def learn(self, train_dl, epochs=20, optimizer=None, loss_fcn=None):
        # Omptimizer and Loss Function are required
        if optimizer is None or loss_fcn is None:
            print('Need to specify an optimizer and loss function')
            return
        
        for epoch in tqdm(range(epochs)):
            # Iterate through batches
            total_loss = 0.
            correct = 0
            total = 0

            for x, t in train_dl:
                x, t = x.to(device), t.to(device)
                y = self(x)
                _, y_pred = torch.max(y, 1)
                correct += torch.sum(y_pred == t)
                total += len(t)
                
                # Backpropagation
                loss = loss_fcn(y, t)
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient Descent
                optimizer.step()
                
                # Bookkeeping
                total_loss += loss.item() * len(t)

            self.train_losses.append(total_loss / len(train_dl.dataset))
            self.train_acc.append((correct / total).item())

        plt.figure(figsize=(6,4))
        plt.plot(self.train_losses); plt.yscale('log');


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((np.asarray(imageA) - np.asarray(imageB)) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def video(path):
    # Counters
    roi1_count, roi6_count = 0, 0

    # Load Models
    model_roi6 = torch.load('models/roi6/vgg11_trained.pt')
    model_roi6.eval()

    model_roi1 = torch.load('models/roi1/vgg11_trained.pt')
    model_roi1.eval()

    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        _, image = cap.read()
        image = cv2.resize(image, (1024, 1024))

        # ROI 1
        x, y, w, h = BOXES[0]
        cv2.imwrite("temp.jpg", image[y:y + h, x:x + w])
        crop = TRANSFORMATIONS(Image.open("temp.jpg"))
        crop = crop.view(1, 3, IMG_SIZE, IMG_SIZE)
        
        outputs = model_roi1(crop)
        _, preds = torch.max(abs(outputs), 1)

        color = (0,255,0)
        if preds[0].item() == 0:
            roi1_count += 1
            color = (255, 0, 0)
            
        cv2.putText(image, f"Count- {roi1_count}", (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # ROI 6
        x, y, w, h = BOXES[-1]
        cv2.imwrite("temp.jpg", image[y:y + h, x:x + w])
        crop = TRANSFORMATIONS(Image.open("temp.jpg"))
        crop = crop.view(1, 3, IMG_SIZE, IMG_SIZE)
        
        outputs = model_roi6(crop)
        _, preds = torch.max(abs(outputs), 1)

        color = (0,255,0)
        if preds[0].item() == 0:
            roi6_count += 1
            color = (255, 0, 0)
            
        cv2.putText(image, f"Count- {roi6_count}", (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        cv2.imshow('Video', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    os.remove("temp.jpg")

    return roi1_count, roi6_count

if __name__ == '__main__':
    roi1, roi6 = video('video.avi')
    print(f'ROI1 Count - {roi1}\nROI6 Count - {roi6}')