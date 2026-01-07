import torch
import config
from .fgsm import fgsm_attack
from .pgd import pgd_attack


def test_transferability(source_model, target_models, data_loader, attack_fn, attack_params=None, device=None):

    if attack_params is None:
        attack_params = {}
    if device is None:
        device = config.DEVICE
    
    source_model.eval()
    for model in target_models:
        model.eval()
    
                         
    total = 0
    source_success = 0
    target_success = {i: 0 for i in range(len(target_models))}
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
                                                       
        adversarial_images = attack_fn(source_model, images, labels, device=device, **attack_params)
        
                              
        with torch.no_grad():
            outputs = source_model(adversarial_images)
            _, predicted = torch.max(outputs.data, 1)
            source_success += (predicted != labels).sum().item()
        
                               
        for i, target_model in enumerate(target_models):
            with torch.no_grad():
                outputs = target_model(adversarial_images)
                _, predicted = torch.max(outputs.data, 1)
                target_success[i] += (predicted != labels).sum().item()
        
        total += labels.size(0)
    
                             
    results = {
        : source_success / total,
        : {i: target_success[i] / total for i in range(len(target_models))},
        : total
    }
    
    return results


def cross_model_transferability_matrix(models, model_names, data_loader, attack_fn, attack_params=None, device=None):
   
    if attack_params is None:
        attack_params = {}
    if device is None:
        device = config.DEVICE
    
    n_models = len(models)
    transfer_matrix = torch.zeros(n_models, n_models)
    
    for i, source_model in enumerate(models):
        print(f"Generating adversarial examples from {model_names[i]}...")
        
        source_model.eval()
        total = 0
        success_counts = [0] * n_models
        
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
                                                           
            adversarial_images = attack_fn(source_model, images, labels, device=device, **attack_params)
            
                                
            for j, target_model in enumerate(models):
                target_model.eval()
                with torch.no_grad():
                    outputs = target_model(adversarial_images)
                    _, predicted = torch.max(outputs.data, 1)
                    success_counts[j] += (predicted != labels).sum().item()
            
            total += labels.size(0)
        
                                          
        for j in range(n_models):
            transfer_matrix[i, j] = success_counts[j] / total
    
    return transfer_matrix


def ensemble_attack(models, images, labels, attack_fn, attack_params=None, device=None):
    
    if attack_params is None:
        attack_params = {}
    if device is None:
        device = config.DEVICE
    
                                                                   
                                                               
    
    for model in models:
        model.eval()
    
    images = images.to(device)
    labels = labels.to(device)
    
                           
    epsilon = attack_params.get('epsilon', config.ATTACK_EPSILON)
    
    images.requires_grad = True
    
                           
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    for model in models:
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss
    
                  
    total_loss = total_loss / len(models)
    
                   
    total_loss.backward()
    
                  
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    
                                 
    adversarial_images = images + epsilon * sign_data_grad
    adversarial_images = torch.clamp(adversarial_images, images.min(), images.max())
    
    return adversarial_images.detach()
