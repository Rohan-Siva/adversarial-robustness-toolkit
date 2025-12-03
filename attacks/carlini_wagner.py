import torch
import torch.nn as nn
import torch.optim as optim
import config


def cw_l2_attack(model, images, labels, targeted=False, c=1.0, kappa=0, max_iter=None, 
                 learning_rate=None, device=None):

    if max_iter is None:
        max_iter = config.CW_MAX_ITERATIONS
    if learning_rate is None:
        learning_rate = config.CW_LEARNING_RATE
    if device is None:
        device = config.DEVICE
    
    model.eval()
    
    images = images.to(device)
    labels = labels.to(device)
    
    batch_size = images.shape[0]
    
                                      
                                              
    w = torch.zeros_like(images, requires_grad=True)
    
    optimizer = optim.Adam([w], lr=learning_rate)
    
    best_adversarial = images.clone()
    best_l2 = torch.full((batch_size,), float('inf')).to(device)
    
    for iteration in range(max_iter):
                                                
        adversarial_images = 0.5 * (torch.tanh(w) + 1)
        
                               
        l2_dist = torch.norm((adversarial_images - images).view(batch_size, -1), p=2, dim=1)
        
                      
        outputs = model(adversarial_images)
        
                                                     
        real = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = outputs.clone()
        other.scatter_(1, labels.unsqueeze(1), -float('inf'))
        other_max = other.max(1)[0]
        
                  
        if targeted:
                                                                      
            loss_attack = torch.clamp(other_max - real + kappa, min=0)
        else:
                                                         
            loss_attack = torch.clamp(real - other_max + kappa, min=0)
        
                                                   
        loss = l2_dist + c * loss_attack
        loss = loss.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
                                          
        with torch.no_grad():
                                                 
            pred = outputs.argmax(1)
            if targeted:
                successful = (pred == labels)
            else:
                successful = (pred != labels)
            
                                  
            mask = successful & (l2_dist < best_l2)
            best_l2 = torch.where(mask, l2_dist, best_l2)
            best_adversarial = torch.where(
                mask.view(-1, 1, 1, 1),
                adversarial_images,
                best_adversarial
            )
    
    return best_adversarial.detach()


def cw_l2_attack_binary_search(model, images, labels, targeted=False, binary_search_steps=None,
                                max_iter=None, learning_rate=None, device=None):
    
    if binary_search_steps is None:
        binary_search_steps = config.CW_BINARY_SEARCH_STEPS
    if device is None:
        device = config.DEVICE
    
    batch_size = images.shape[0]
    
                                
    c_lower = torch.zeros(batch_size).to(device)
    c_upper = torch.full((batch_size,), 1e10).to(device)
    c = torch.full((batch_size,), 0.01).to(device)
    
    best_adversarial = images.clone()
    
    for search_step in range(binary_search_steps):
                                                                       
        c_mean = c.mean().item()
        adversarial = cw_l2_attack(
            model, images, labels, targeted, c_mean, 
            max_iter=max_iter, learning_rate=learning_rate, device=device
        )
        
                       
        with torch.no_grad():
            outputs = model(adversarial)
            pred = outputs.argmax(1)
            if targeted:
                successful = (pred == labels)
            else:
                successful = (pred != labels)
        
                                     
        c_upper = torch.where(successful, c, c_upper)
        c_lower = torch.where(~successful, c, c_lower)
        c = (c_lower + c_upper) / 2
        
                                          
        best_adversarial = torch.where(
            successful.view(-1, 1, 1, 1),
            adversarial,
            best_adversarial
        )
    
    return best_adversarial.detach()


def evaluate_cw(model, data_loader, c=1.0, max_iter=100, device=None):
    
    if device is None:
        device = config.DEVICE
    
    model.eval()
    
    clean_correct = 0
    adversarial_correct = 0
    total = 0
    total_l2 = 0.0
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
                                  
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            clean_correct += (predicted == labels).sum().item()
        
                                       
        adversarial_images = cw_l2_attack(
            model, images, labels, c=c, max_iter=max_iter, device=device
        )
        
                               
        l2_dist = torch.norm(
            (adversarial_images - images).view(images.shape[0], -1), 
            p=2, dim=1
        ).mean().item()
        total_l2 += l2_dist * images.shape[0]
        
                                        
        with torch.no_grad():
            outputs = model(adversarial_images)
            _, predicted = torch.max(outputs.data, 1)
            adversarial_correct += (predicted == labels).sum().item()
        
        total += labels.size(0)
    
    clean_accuracy = clean_correct / total
    adversarial_accuracy = adversarial_correct / total
    attack_success_rate = 1 - adversarial_accuracy
    avg_l2_distance = total_l2 / total
    
    return clean_accuracy, adversarial_accuracy, attack_success_rate, avg_l2_distance
