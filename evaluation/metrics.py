import torch
import numpy as np


def robust_accuracy(model, data_loader, attack_fn, attack_params=None, device=None):
    
    if attack_params is None:
        attack_params = {}
    if device is None:
        import config
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


def attack_success_rate(model, data_loader, attack_fn, attack_params=None, device=None):
    
    return 1 - robust_accuracy(model, data_loader, attack_fn, attack_params, device)


def average_perturbation_norm(original_images, adversarial_images, p=2):
    
    perturbation = adversarial_images - original_images
    batch_size = original_images.shape[0]
    
    if p == 'inf' or p == float('inf'):
        norms = torch.max(torch.abs(perturbation.view(batch_size, -1)), dim=1)[0]
    else:
        norms = torch.norm(perturbation.view(batch_size, -1), p=p, dim=1)
    
    return norms.mean().item()


def confidence_on_adversarial(model, adversarial_images, device=None):
    
    if device is None:
        import config
        device = config.DEVICE
    
    model.eval()
    adversarial_images = adversarial_images.to(device)
    
    with torch.no_grad():
        outputs = model(adversarial_images)
        probabilities = torch.softmax(outputs, dim=1)
        max_confidence = probabilities.max(dim=1)[0]
    
    return max_confidence.mean().item()


def robustness_gap(clean_accuracy, robust_accuracy):
    
    return clean_accuracy - robust_accuracy


def calculate_all_metrics(model, data_loader, attack_fn, attack_params=None, device=None):
    
    if attack_params is None:
        attack_params = {}
    if device is None:
        import config
        device = config.DEVICE
    
    model.eval()
    
    clean_correct = 0
    robust_correct = 0
    total = 0
    total_perturbation = 0
    total_confidence = 0
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
                        
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            clean_correct += (predicted == labels).sum().item()
        
                                       
        adversarial_images = attack_fn(model, images, labels, device=device, **attack_params)
        
                         
        with torch.no_grad():
            outputs = model(adversarial_images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            robust_correct += (predicted == labels).sum().item()
            
                        
            max_confidence = probabilities.max(dim=1)[0]
            total_confidence += max_confidence.sum().item()
        
                           
        perturbation_norm = average_perturbation_norm(images, adversarial_images, p=2)
        total_perturbation += perturbation_norm * images.size(0)
        
        total += labels.size(0)
    
    clean_acc = clean_correct / total
    robust_acc = robust_correct / total
    
    metrics = {
        : clean_acc,
        : robust_acc,
        : 1 - robust_acc,
        : clean_acc - robust_acc,
        : total_perturbation / total,
        : total_confidence / total
    }
    
    return metrics
