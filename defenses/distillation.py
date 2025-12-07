import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import config


def train_teacher_model(model, train_loader, test_loader, temperature=None, epochs=None, 
                       lr=None, device=None):
    
    if temperature is None:
        temperature = config.DISTILLATION_TEMP
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
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Training Teacher - Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images) / temperature
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})
    
    print("Teacher model training complete")
    return model


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha=0.5):
    
                                      
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (temperature ** 2)
    
                     
    hard_loss = F.cross_entropy(student_logits, labels)
    
                   
    loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    return loss


def defensive_distillation(teacher_model, student_model, train_loader, test_loader,
                          temperature=None, epochs=None, lr=None, alpha=0.5, device=None):
    
    if temperature is None:
        temperature = config.DISTILLATION_TEMP
    if epochs is None:
        epochs = config.EPOCHS
    if lr is None:
        lr = config.LEARNING_RATE
    if device is None:
        device = config.DEVICE
    
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    teacher_model.eval()                     
    
    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=config.MOMENTUM,
                         weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        : [],
        : []
    }
    
    for epoch in range(epochs):
        student_model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Distillation - Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
                                                    
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            
                                     
            optimizer.zero_grad()
            student_logits = student_model(images)
            
                                         
            loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})
        
        scheduler.step()
        
                  
        student_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = correct / total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1}: Test Acc: {test_acc:.4f}')
    
    return student_model, history


def create_distilled_model(model_fn, train_loader, test_loader, temperature=None,
                          teacher_epochs=None, student_epochs=None, device=None):
    
    if temperature is None:
        temperature = config.DISTILLATION_TEMP
    if teacher_epochs is None:
        teacher_epochs = config.EPOCHS
    if student_epochs is None:
        student_epochs = config.EPOCHS
    if device is None:
        device = config.DEVICE
    
                   
    print("Step 1: Training teacher model...")
    teacher_model = model_fn()
    teacher_model = train_teacher_model(
        teacher_model, train_loader, test_loader,
        temperature, teacher_epochs, device=device
    )
    
                                    
    print("\nStep 2: Training student model via distillation...")
    student_model = model_fn()
    student_model, history = defensive_distillation(
        teacher_model, student_model, train_loader, test_loader,
        temperature, student_epochs, device=device
    )
    
    return student_model
