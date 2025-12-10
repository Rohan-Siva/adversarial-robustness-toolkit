

import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models import load_model
from utils import get_data_loaders
from attacks import fgsm_attack, pgd_attack, cross_model_transferability_matrix
from evaluation import (
    RobustnessEvaluator,
    plot_attack_comparison,
    plot_robustness_curve,
    plot_transferability_matrix
)


def evaluate_epsilon_robustness():
    
    print("\n" + "=" * 80)
    print("EPSILON ROBUSTNESS ANALYSIS")
    print("=" * 80)
    
               
    train_loader, test_loader = get_data_loaders(config.DATASET, batch_size=config.EVAL_BATCH_SIZE)
    
                              
    limited_eval = []
    total_samples = 0
    for batch in test_loader:
        limited_eval.append(batch)
        total_samples += batch[0].size(0)
        if total_samples >= config.NUM_EVAL_SAMPLES:
            break
    
                
    model = load_model(config.MODEL_NAME, num_classes=config.NUM_CLASSES, 
                      pretrained=config.PRETRAINED, device=config.DEVICE)
    
    evaluator = RobustnessEvaluator(model, config.DEVICE)
    
                                   
    epsilons = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
    
    print("\nEvaluating FGSM across epsilon values...")
    eps_fgsm, acc_fgsm = evaluator.evaluate_epsilon_robustness(
        fgsm_attack, limited_eval, epsilons
    )
    
    print("\nEvaluating PGD across epsilon values...")
    eps_pgd, acc_pgd = evaluator.evaluate_epsilon_robustness(
        pgd_attack, limited_eval, epsilons, 
        attack_params={'alpha': 0.01, 'num_iter': 20}
    )
    
                            
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(eps_fgsm, acc_fgsm, marker='o', label='FGSM', linewidth=2)
    plt.plot(eps_pgd, acc_pgd, marker='s', label='PGD-20', linewidth=2)
    plt.xlabel('Epsilon (Perturbation Budget)')
    plt.ylabel('Accuracy')
    plt.title('Robustness Curve: Accuracy vs Perturbation Budget')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'epsilon_robustness_curve.png'), 
                dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\nRobustness curve saved to {config.RESULTS_DIR}/epsilon_robustness_curve.png")


def evaluate_transferability():
    
    print("\n" + "=" * 80)
    print("TRANSFERABILITY ANALYSIS")
    print("=" * 80)
    
               
    train_loader, test_loader = get_data_loaders(config.DATASET, batch_size=config.EVAL_BATCH_SIZE)
    
                              
    limited_eval = []
    total_samples = 0
    for batch in test_loader:
        limited_eval.append(batch)
        total_samples += batch[0].size(0)
        if total_samples >= 500:                                   
            break
    
                           
    print("\nLoading models...")
    models = [
        load_model('resnet18', num_classes=config.NUM_CLASSES, pretrained=True, device=config.DEVICE),
        load_model('resnet50', num_classes=config.NUM_CLASSES, pretrained=True, device=config.DEVICE),
        load_model('vgg16', num_classes=config.NUM_CLASSES, pretrained=True, device=config.DEVICE),
    ]
    model_names = ['ResNet-18', 'ResNet-50', 'VGG-16']
    
                                    
    print("\nComputing transferability matrix...")
    transfer_matrix = cross_model_transferability_matrix(
        models, model_names, limited_eval,
        fgsm_attack, {'epsilon': 0.03}, config.DEVICE
    )
    
                                 
    plot_transferability_matrix(
        transfer_matrix, model_names,
        save_path=os.path.join(config.RESULTS_DIR, 'transferability_matrix.png')
    )
    
    print(f"\nTransferability matrix saved to {config.RESULTS_DIR}/transferability_matrix.png")
    
                  
    print("\nTransferability Matrix:")
    print("(Rows: Source Model, Columns: Target Model)")
    print("\n" + " " * 15 + "  ".join(model_names))
    for i, source in enumerate(model_names):
        row_str = f"{source:15s}"
        for j in range(len(model_names)):
            row_str += f"  {transfer_matrix[i, j]:.2f}"
        print(row_str)


def main():
    print("=" * 80)
    print("COMPREHENSIVE ATTACK AND DEFENSE COMPARISON")
    print("=" * 80)
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
                                    
    evaluate_epsilon_robustness()
    
                                 
    evaluate_transferability()
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {config.RESULTS_DIR}")


if __name__ == '__main__':
    main()
