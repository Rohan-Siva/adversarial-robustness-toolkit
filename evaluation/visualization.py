

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config


def plot_adversarial_examples(original_images, adversarial_images, labels, predictions,
                              num_examples=5, denormalize_fn=None, save_path=None):
    
    num_examples = min(num_examples, original_images.shape[0])
    
    fig, axes = plt.subplots(3, num_examples, figsize=(3*num_examples, 9))
    if num_examples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_examples):
                  
        orig = original_images[i].cpu().detach()
        if denormalize_fn:
            orig = denormalize_fn(orig)
        orig = orig.permute(1, 2, 0).numpy()
        orig = np.clip(orig, 0, 1)
        
                     
        adv = adversarial_images[i].cpu().detach()
        if denormalize_fn:
            adv = denormalize_fn(adv)
        adv = adv.permute(1, 2, 0).numpy()
        adv = np.clip(adv, 0, 1)
        
                      
        pert = (adversarial_images[i] - original_images[i]).cpu().detach()
        if denormalize_fn:
            pert = denormalize_fn(pert)
        pert = pert.permute(1, 2, 0).numpy()
        pert = np.clip(pert * 10 + 0.5, 0, 1)
        
        axes[0, i].imshow(orig if orig.shape[2] == 3 else orig.squeeze(), cmap='gray' if orig.shape[2] == 1 else None)
        axes[0, i].set_title(f'Original\nTrue: {labels[i].item()}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(adv if adv.shape[2] == 3 else adv.squeeze(), cmap='gray' if adv.shape[2] == 1 else None)
        axes[1, i].set_title(f'Adversarial\nPred: {predictions[i].item()}')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(pert if pert.shape[2] == 3 else pert.squeeze(), cmap='gray' if pert.shape[2] == 1 else None)
        axes[2, i].set_title('Perturbation (10x)')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()


def plot_attack_comparison(results_dict, save_path=None):
    
    attacks = list(results_dict.keys())
    success_rates = [results_dict[a]['attack_success_rate'] for a in attacks]
    perturbations = [results_dict[a]['avg_perturbation_l2'] for a in attacks]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
                   
    ax1.bar(attacks, success_rates, color='coral')
    ax1.set_ylabel('Attack Success Rate')
    ax1.set_title('Attack Success Rate Comparison')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
                        
    ax2.bar(attacks, perturbations, color='skyblue')
    ax2.set_ylabel('Average L2 Perturbation')
    ax2.set_title('Perturbation Size Comparison')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()


def plot_defense_comparison(results_dict, save_path=None):
    
    defenses = list(results_dict.keys())
    clean_acc = [results_dict[d]['clean_accuracy'] for d in defenses]
    robust_acc = [results_dict[d]['robust_accuracy'] for d in defenses]
    
    x = np.arange(len(defenses))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, clean_acc, width, label='Clean Accuracy', color='lightgreen')
    bars2 = ax.bar(x + width/2, robust_acc, width, label='Robust Accuracy', color='salmon')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Defense Mechanism Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(defenses, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()


def plot_robustness_curve(epsilons, accuracies, save_path=None):
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Epsilon (Perturbation Budget)')
    plt.ylabel('Accuracy')
    plt.title('Robustness Curve: Accuracy vs Perturbation Budget')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()


def plot_transferability_matrix(transfer_matrix, model_names, save_path=None):
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        transfer_matrix.cpu().numpy() if torch.is_tensor(transfer_matrix) else transfer_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=model_names,
        yticklabels=model_names,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Attack Success Rate'}
    )
    
    plt.xlabel('Target Model')
    plt.ylabel('Source Model')
    plt.title('Adversarial Example Transferability Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()


def plot_training_history(history, save_path=None):
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
          
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
              
    axes[1].plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(epochs, history['test_acc'], label='Test Acc', marker='s')
    if 'test_adv_acc' in history:
        axes[1].plot(epochs, history['test_adv_acc'], label='Test Adv Acc', marker='^')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()


def plot_confidence_distribution(clean_confidences, adv_confidences, save_path=None):
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(clean_confidences, bins=50, alpha=0.6, label='Clean', color='green')
    plt.hist(adv_confidences, bins=50, alpha=0.6, label='Adversarial', color='red')
    
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution: Clean vs Adversarial Examples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()
