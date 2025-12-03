
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def adversarial_training(model, train_loader, test_loader, epochs=None, lr=None, 
                        attack_fn=None, attack_params=None, adv_ratio=None, device=None):

    if epochs is None:
        epochs = config.EPOCHS
    if lr is None:
        lr = config.LEARNING_RATE
    if attack_fn is None:
        attack_fn = pgd_attack                                       
    if attack_params is None:
        attack_params = {}
    if adv_ratio is None:
        adv_ratio = config.ADV_TRAINING_RATIO
    if device is None:
        device = config.DEVICE
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config.MOMENTUM, 
                         weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_adv_acc': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
                                                                 
            batch_size = images.size(0)
            adv_size = int(batch_size * adv_ratio)
            clean_size = batch_size - adv_size
            
                                                                 
            if adv_size > 0:
                adv_images_batch = images[:adv_size]
                adv_labels_batch = labels[:adv_size]
                
                                               
                with torch.enable_grad():
                    adversarial_images = attack_fn(
                        model, adv_images_batch, adv_labels_batch, 
                        device=device, **attack_params
                    )
                
                                                        
                if clean_size > 0:
                    combined_images = torch.cat([images[adv_size:], adversarial_images], dim=0)
                    combined_labels = torch.cat([labels[adv_size:], adv_labels_batch], dim=0)
                else:
                    combined_images = adversarial_images
                    combined_labels = adv_labels_batch
            else:
                combined_images = images
                combined_labels = labels
            
                          
            optimizer.zero_grad()
            outputs = model(combined_images)
            loss = criterion(outputs, combined_labels)
            
                           
            loss.backward()
            optimizer.step()
            
                        
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += combined_labels.size(0)
            train_correct += (predicted == combined_labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        scheduler.step()
        
                              
        test_acc = evaluate_clean_accuracy(model, test_loader, device)
        test_adv_acc = evaluate_adversarial_accuracy(
            model, test_loader, attack_fn, attack_params, device
        )
        
                        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['test_acc'].append(test_acc)
        history['test_adv_acc'].append(test_adv_acc)
        
        print(f'Epoch {epoch+1}: Test Acc: {test_acc:.4f}, Test Adv Acc: {test_adv_acc:.4f}')
    
    return model, history


def evaluate_clean_accuracy(model, data_loader, device=None):
    
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
    
    return correct / total


def evaluate_adversarial_accuracy(model, data_loader, attack_fn, attack_params=None, device=None):
    """
    Evaluate model accuracy on adversarial images.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        attack_fn: Attack function
        attack_params: Attack parameters
        device: Device to run on
        
    Returns:
        accuracy: Adversarial accuracy
    """
    if attack_params is None:
        attack_params = {}
    if device is None:
        device = config.DEVICE
    
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
                                       
        adversarial_images = attack_fn(model, images, labels, device=device, **attack_params)
        
                  
        with torch.no_grad():
            outputs = model(adversarial_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total


def train_standard(model, train_loader, test_loader, epochs=None, lr=None, device=None):
    
    if epochs is None:
        epochs = config.EPOCHS
    if lr is None:
        lr = config.LEARNING_RATE
    if device is None:
        device = config.DEVICE
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config.MOMENTUM,
                         weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        scheduler.step()
        
        test_acc = evaluate_clean_accuracy(model, test_loader, device)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1}: Test Acc: {test_acc:.4f}')
    
    return model, history
