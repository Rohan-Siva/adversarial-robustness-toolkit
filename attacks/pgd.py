
import torch
import torch.nn as nn
import config


def pgd_attack(model, images, labels, epsilon=None, alpha=None, num_iter=None, device=None, random_start=True):
    

    if epsilon is None:
        epsilon = config.ATTACK_EPSILON
    if alpha is None:
        alpha = config.ATTACK_ALPHA
    if num_iter is None:
        num_iter = config.PGD_ITERATIONS
    if device is None:
        device = config.DEVICE
    
    model.eval()
    
    images = images.to(device)
    labels = labels.to(device)
    
                                                      
    if random_start:
                                                 
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        adversarial_images = images + delta
        adversarial_images = torch.clamp(adversarial_images, images.min(), images.max())
    else:
        adversarial_images = images.clone()
    
    criterion = nn.CrossEntropyLoss()
    
    for i in range(num_iter):
        adversarial_images.requires_grad = True
        
        outputs = model(adversarial_images)
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
                                    
        data_grad = adversarial_images.grad.data
        adversarial_images = adversarial_images.detach() + alpha * data_grad.sign()
        
                                                            
        perturbation = torch.clamp(adversarial_images - images, -epsilon, epsilon)
        adversarial_images = images + perturbation
        
                                    
        adversarial_images = torch.clamp(adversarial_images, images.min(), images.max())
    
    return adversarial_images.detach()


def pgd_targeted_attack(model, images, target_labels, epsilon=None, alpha=None, num_iter=None, device=None):
 
    if epsilon is None:
        epsilon = config.ATTACK_EPSILON
    if alpha is None:
        alpha = config.ATTACK_ALPHA
    if num_iter is None:
        num_iter = config.PGD_ITERATIONS
    if device is None:
        device = config.DEVICE
    
    model.eval()
    
    images = images.to(device)
    target_labels = target_labels.to(device)
    
                  
    delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    adversarial_images = images + delta
    adversarial_images = torch.clamp(adversarial_images, images.min(), images.max())
    
    criterion = nn.CrossEntropyLoss()
    
    for i in range(num_iter):
        adversarial_images.requires_grad = True
        
        outputs = model(adversarial_images)
        loss = criterion(outputs, target_labels)
        
        model.zero_grad()
        loss.backward()
        
        data_grad = adversarial_images.grad.data
                                                             
        adversarial_images = adversarial_images.detach() - alpha * data_grad.sign()
        
        perturbation = torch.clamp(adversarial_images - images, -epsilon, epsilon)
        adversarial_images = images + perturbation
        adversarial_images = torch.clamp(adversarial_images, images.min(), images.max())
    
    return adversarial_images.detach()


def evaluate_pgd(model, data_loader, epsilon=None, alpha=None, num_iter=None, device=None):
    
    if epsilon is None:
        epsilon = config.ATTACK_EPSILON
    if alpha is None:
        alpha = config.ATTACK_ALPHA
    if num_iter is None:
        num_iter = config.PGD_ITERATIONS
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
        
                                       
        adversarial_images = pgd_attack(model, images, labels, epsilon, alpha, num_iter, device)
        
                                        
        with torch.no_grad():
            outputs = model(adversarial_images)
            _, predicted = torch.max(outputs.data, 1)
            adversarial_correct += (predicted == labels).sum().item()
        
        total += labels.size(0)
    
    clean_accuracy = clean_correct / total
    adversarial_accuracy = adversarial_correct / total
    attack_success_rate = 1 - adversarial_accuracy
    
    return clean_accuracy, adversarial_accuracy, attack_success_rate
