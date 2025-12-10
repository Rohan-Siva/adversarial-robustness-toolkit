

import torch
import os
import sys

                              
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models import load_model
from utils import get_data_loaders, denormalize_cifar10
from attacks import fgsm_attack, pgd_attack, cw_l2_attack
from evaluation import RobustnessEvaluator, plot_attack_comparison


def main():
    print("=" * 80)
    print("ADVERSARIAL ATTACK EVALUATION")
    print("=" * 80)
    
                              
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
               
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(
        config.DATASET,
        batch_size=config.EVAL_BATCH_SIZE
    )
    
                
    print(f"\nLoading {config.MODEL_NAME} model...")
    model = load_model(
        config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        device=config.DEVICE
    )
    
                      
    evaluator = RobustnessEvaluator(model, config.DEVICE)
    
                                
    attacks = {
        : (
            fgsm_attack,
            {'epsilon': 0.03}
        ),
        : (
            fgsm_attack,
            {'epsilon': 0.05}
        ),
        : (
            pgd_attack,
            {'epsilon': 0.03, 'alpha': 0.01, 'num_iter': 20}
        ),
        : (
            pgd_attack,
            {'epsilon': 0.03, 'alpha': 0.01, 'num_iter': 40}
        ),
        : (
            cw_l2_attack,
            {'c': 1.0, 'max_iter': 100}
        ),
    }
    
                          
    print("\n" + "=" * 80)
    print("EVALUATING ATTACKS")
    print("=" * 80)
    
                                                   
    subset_loader = torch.utils.data.DataLoader(
        test_loader.dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
                               
    limited_loader = []
    total_samples = 0
    for batch in subset_loader:
        limited_loader.append(batch)
        total_samples += batch[0].size(0)
        if total_samples >= config.NUM_EVAL_SAMPLES:
            break
    
                          
    for attack_name, (attack_fn, attack_params) in attacks.items():
        evaluator.evaluate_attack(attack_name, attack_fn, limited_loader, attack_params)
    
                     
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)
    
    report_path = evaluator.generate_report(config.RESULTS_DIR)
    
                                                 
    print("\nSaving adversarial examples...")
    evaluator.save_adversarial_examples(
        fgsm_attack,
        limited_loader[:1],                    
        num_batches=1,
        attack_params={'epsilon': 0.03},
        denormalize_fn=denormalize_cifar10,
        save_dir=config.RESULTS_DIR
    )
    
    evaluator.save_adversarial_examples(
        pgd_attack,
        limited_loader[:1],
        num_batches=1,
        attack_params={'epsilon': 0.03, 'alpha': 0.01, 'num_iter': 20},
        denormalize_fn=denormalize_cifar10,
        save_dir=config.RESULTS_DIR
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"Report: {report_path}")


if __name__ == '__main__':
    main()
