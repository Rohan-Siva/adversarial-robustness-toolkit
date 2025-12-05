
import torch
import numpy as np
import matplotlib.pyplot as plt
import config


def clip_perturbation(perturbation, epsilon):
    
    return torch.clamp(perturbation, -epsilon, epsilon)


def calculate_success_rate(model, adversarial_images, labels, device=None):
    
    if device is None:
        device = config.DEVICE
    
    model.eval()
    adversarial_images = adversarial_images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(adversarial_images)
        _, predicted = torch.max(outputs.data, 1)
        success = (predicted != labels).sum().item()
    
    success_rate = success / labels.size(0)
    return success_rate


def calculate_perturbation_norm(original_images, adversarial_images, p=2):
    
    perturbation = adversarial_images - original_images
    batch_size = original_images.shape[0]
    
    if p == float('inf'):
        norm = torch.max(torch.abs(perturbation.view(batch_size, -1)), dim=1)[0]
    else:
        norm = torch.norm(perturbation.view(batch_size, -1), p=p, dim=1)
    
    return norm.mean().item()


def visualize_adversarial_examples(original_images, adversarial_images, labels, predictions, 
                                   num_examples=5, denormalize_fn=None, save_path=None):
    
    num_examples = min(num_examples, original_images.shape[0])
    
    fig, axes = plt.subplots(3, num_examples, figsize=(3*num_examples, 9))
    
    for i in range(num_examples):
                        
        orig_img = original_images[i].cpu()
        if denormalize_fn:
            orig_img = denormalize_fn(orig_img)
        orig_img = orig_img.permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        
                           
        adv_img = adversarial_images[i].cpu()
        if denormalize_fn:
            adv_img = denormalize_fn(adv_img)
        adv_img = adv_img.permute(1, 2, 0).numpy()
        adv_img = np.clip(adv_img, 0, 1)
        
                                                 
        perturbation = (adversarial_images[i] - original_images[i]).cpu()
        if denormalize_fn:
            perturbation = denormalize_fn(perturbation)
        perturbation = perturbation.permute(1, 2, 0).numpy()
        perturbation = np.clip(perturbation * 10 + 0.5, 0, 1)                      
        
              
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'Original\nLabel: {labels[i].item()}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(adv_img)
        axes[1, i].set_title(f'Adversarial\nPred: {predictions[i].item()}')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(perturbation)
        axes[2, i].set_title('Perturbation\n(10x amplified)')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.close()


def generate_random_targets(labels, num_classes):
    
    target_labels = torch.randint(0, num_classes, labels.shape).to(labels.device)
    
                                                   
    same_mask = (target_labels == labels)
    while same_mask.any():
        target_labels[same_mask] = torch.randint(0, num_classes, (same_mask.sum(),)).to(labels.device)
        same_mask = (target_labels == labels)
    
    return target_labels
