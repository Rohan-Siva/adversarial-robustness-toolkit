
import torch
import torch.nn as nn
import torchvision.models as models
import config


class SimpleCNN(nn.Module):
    
    def __init__(self, num_classes=10, in_channels=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
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


def load_model(model_name='resnet18', num_classes=10, pretrained=True, device=None):

    if device is None:
        device = config.DEVICE
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
                                                             
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        if num_classes != 1000:
            model.classifier[6] = nn.Linear(4096, num_classes)
    
    elif model_name == 'simple_cnn':
        in_channels = 1 if num_classes == 10 else 3                              
        model = SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = model.to(device)
    model.eval()
    
    return model


def save_model(model, path):

    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"Model saved to {path}")


def load_checkpoint(model, path, device=None):

    if device is None:
        device = config.DEVICE
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {path}")
    return model


def evaluate_model(model, data_loader, device=None):

    if device is None:
        device = config.DEVICE
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy
