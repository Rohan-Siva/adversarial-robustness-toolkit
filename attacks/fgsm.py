
import torch
import torch.nn as nn
import config


def fgsm_attack(model, images, labels, epsilon=None, device=None):

    if epsilon is None:
        epsilon = config.ATTACK_EPSILON
    if device is None:
        device = config.DEVICE
    
                                  
    model.eval()
    
                    
    images = images.to(device)
    labels = labels.to(device)
    
                                            
    images.requires_grad = True
    
                  
    outputs = model(images)
    
                    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    
                                    
    model.zero_grad()
    loss.backward()
    
                                  
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    
                                 
    adversarial_images = images + epsilon * sign_data_grad
    
                                                                           
                                                                       
    adversarial_images = torch.clamp(adversarial_images, images.min(), images.max())
    
    return adversarial_images.detach()


def fgsm_targeted_attack(model, images, target_labels, epsilon=None, device=None):

    if epsilon is None:
        epsilon = config.ATTACK_EPSILON
    if device is None:
        device = config.DEVICE
    
    model.eval()
    
    images = images.to(device)
    target_labels = target_labels.to(device)
    
    images.requires_grad = True
    
    outputs = model(images)
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, target_labels)
    
    model.zero_grad()
    loss.backward()
    
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    
                                                         
    adversarial_images = images - epsilon * sign_data_grad
    adversarial_images = torch.clamp(adversarial_images, images.min(), images.max())
    
    return adversarial_images.detach()


def evaluate_fgsm(model, data_loader, epsilon=None, device=None):

    if epsilon is None:
        epsilon = config.ATTACK_EPSILON
    if device is None:
        device = config.DEVICE
    
    model.eval()
    
    clean_correct = 0
    adversarial_correct = 0
    total = 0
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
                                  
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            clean_correct += (predicted == labels).sum().item()
        
                                       
        adversarial_images = fgsm_attack(model, images, labels, epsilon, device)
        
                                        
        with torch.no_grad():
            outputs = model(adversarial_images)
            _, predicted = torch.max(outputs.data, 1)
            adversarial_correct += (predicted == labels).sum().item()
        
        total += labels.size(0)
    
    clean_accuracy = clean_correct / total
    adversarial_accuracy = adversarial_correct / total
    attack_success_rate = 1 - adversarial_accuracy
    
    return clean_accuracy, adversarial_accuracy, attack_success_rate
