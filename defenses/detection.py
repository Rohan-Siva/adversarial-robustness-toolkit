
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import config


def detect_by_prediction_inconsistency(model, images, transformations, threshold=0.3, device=None):
    
    if device is None:
        device = config.DEVICE
    
    model.eval()
    images = images.to(device)
    
    predictions = []
    
                                                         
    with torch.no_grad():
                             
        outputs = model(images)
        orig_pred = outputs.argmax(dim=1)
        predictions.append(orig_pred)
        
                                            
        for transform in transformations:
            transformed = transform(images)
            outputs = model(transformed)
            pred = outputs.argmax(dim=1)
            predictions.append(pred)
    
                             
    predictions = torch.stack(predictions, dim=1)                                  
    
                                                     
    inconsistency = (predictions != orig_pred.unsqueeze(1)).float().mean(dim=1)
    
    is_adversarial = inconsistency > threshold
    
    return is_adversarial


def detect_by_confidence(model, images, threshold=0.9, device=None):
    
    if device is None:
        device = config.DEVICE
    
    model.eval()
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        max_confidence = probabilities.max(dim=1)[0]
    
    is_adversarial = max_confidence < threshold
    
    return is_adversarial


def detect_by_kernel_density(model, clean_features, images, threshold_percentile=5, device=None):
    
    if device is None:
        device = config.DEVICE
    
                                                                                           
                                                         
    
    model.eval()
    images = images.to(device)
    
                                                                           
                                                     
    features = []
    
    def hook_fn(module, input, output):
        features.append(output)
    
                                                                 
                                                           
                                       
    
    with torch.no_grad():
        outputs = model(images)
                                                      
        test_features = outputs
    
                                                       
    clean_mean = clean_features.mean(dim=0)
    clean_std = clean_features.std(dim=0) + 1e-8
    
               
    test_normalized = (test_features - clean_mean) / clean_std
    clean_normalized = (clean_features - clean_mean) / clean_std
    
                                         
    distances = torch.norm(test_normalized, dim=1)
    
                                                
    clean_distances = torch.norm(clean_normalized, dim=1)
    threshold = torch.quantile(clean_distances, 1 - threshold_percentile / 100)
    
    is_adversarial = distances > threshold
    
    return is_adversarial


class AdversarialDetector(nn.Module):
    
    
    def __init__(self, input_shape):
        super(AdversarialDetector, self).__init__()
        
                             
        self.conv1 = nn.Conv2d(input_shape[0], 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
                                  
        self.flat_size = 64 * (input_shape[1] // 8) * (input_shape[2] // 8)
        
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.fc2 = nn.Linear(128, 2)                                
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_adversarial_detector(detector, clean_loader, adversarial_loader, epochs=10, 
                               lr=0.001, device=None):
    
    if device is None:
        device = config.DEVICE
    
    detector = detector.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
    
    for epoch in range(epochs):
        detector.train()
        train_loss = 0
        correct = 0
        total = 0
        
                                                
        clean_iter = iter(clean_loader)
        adv_iter = iter(adversarial_loader)
        
        for _ in range(min(len(clean_loader), len(adversarial_loader))):
                             
            clean_images, _ = next(clean_iter)
            clean_images = clean_images.to(device)
            clean_labels = torch.zeros(clean_images.size(0), dtype=torch.long).to(device)
            
                                   
            adv_images, _ = next(adv_iter)
            adv_images = adv_images.to(device)
            adv_labels = torch.ones(adv_images.size(0), dtype=torch.long).to(device)
            
                     
            images = torch.cat([clean_images, adv_images], dim=0)
            labels = torch.cat([clean_labels, adv_labels], dim=0)
            
                     
            perm = torch.randperm(images.size(0))
            images = images[perm]
            labels = labels[perm]
            
                   
            optimizer.zero_grad()
            outputs = detector(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Epoch {epoch+1}: Loss: {train_loss:.4f}, Acc: {100.*correct/total:.2f}%')
    
    return detector
