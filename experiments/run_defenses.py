

import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models import load_model, save_model
from utils import get_data_loaders
from attacks import pgd_attack
from defenses import adversarial_training, train_standard
from evaluation import RobustnessEvaluator, plot_defense_comparison, plot_training_history


def main():
    print("=" * 80)
    print("DEFENSE MECHANISM TRAINING AND EVALUATION")
    print("=" * 80)
    
                        
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
               
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(
        config.DATASET,
        batch_size=config.BATCH_SIZE
    )
    
                             
    eval_loader = torch.utils.data.DataLoader(
        test_loader.dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    limited_eval = []
    total_samples = 0
    for batch in eval_loader:
        limited_eval.append(batch)
        total_samples += batch[0].size(0)
        if total_samples >= config.NUM_EVAL_SAMPLES:
            break
    
    results = {}
    
                             
    print("\n" + "=" * 80)
    print("TRAINING STANDARD MODEL")
    print("=" * 80)
    
    standard_model = load_model(
        config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        device=config.DEVICE
    )
    
    standard_model, standard_history = train_standard(
        standard_model,
        train_loader,
        test_loader,
        epochs=5,                    
        lr=config.LEARNING_RATE,
        device=config.DEVICE
    )
    
                
    save_model(standard_model, os.path.join(config.CHECKPOINT_DIR, 'standard_model.pth'))
    
                           
    plot_training_history(
        standard_history,
        save_path=os.path.join(config.RESULTS_DIR, 'standard_training_history.png')
    )
    
              
    print("\nEvaluating standard model...")
    evaluator = RobustnessEvaluator(standard_model, config.DEVICE)
    standard_metrics = evaluator.evaluate_attack(
        ,
        pgd_attack,
        limited_eval,
        {'epsilon': 0.03, 'alpha': 0.01, 'num_iter': 20}
    )
    results['Standard'] = standard_metrics
    
                                          
    print("\n" + "=" * 80)
    print("TRAINING ADVERSARIALLY ROBUST MODEL")
    print("=" * 80)
    
    adv_model = load_model(
        config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        device=config.DEVICE
    )
    
    adv_model, adv_history = adversarial_training(
        adv_model,
        train_loader,
        test_loader,
        epochs=5,                    
        lr=config.LEARNING_RATE,
        attack_fn=pgd_attack,
        attack_params={'epsilon': 0.03, 'alpha': 0.01, 'num_iter': 7},                           
        adv_ratio=0.5,
        device=config.DEVICE
    )
    
                
    save_model(adv_model, os.path.join(config.CHECKPOINT_DIR, 'adversarial_trained_model.pth'))
    
                           
    plot_training_history(
        adv_history,
        save_path=os.path.join(config.RESULTS_DIR, 'adversarial_training_history.png')
    )
    
              
    print("\nEvaluating adversarially trained model...")
    evaluator = RobustnessEvaluator(adv_model, config.DEVICE)
    adv_metrics = evaluator.evaluate_attack(
        ,
        pgd_attack,
        limited_eval,
        {'epsilon': 0.03, 'alpha': 0.01, 'num_iter': 20}
    )
    results['Adversarial Training'] = adv_metrics
    
                         
    print("\n" + "=" * 80)
    print("COMPARING DEFENSES")
    print("=" * 80)
    
    plot_defense_comparison(
        results,
        save_path=os.path.join(config.RESULTS_DIR, 'defense_comparison.png')
    )
    
                     
    report_path = os.path.join(config.RESULTS_DIR, 'defense_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DEFENSE MECHANISM EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for defense_name, metrics in results.items():
            f.write(f"\n{defense_name}\n")
            f.write("-" * 80 + "\n")
            for key, value in metrics.items():
                f.write(f"{key:30s}: {value:.4f}\n")
            f.write("\n")
    
    print("\n" + "=" * 80)
    print("DEFENSE EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"Models saved to: {config.CHECKPOINT_DIR}")
    print(f"Report: {report_path}")


if __name__ == '__main__':
    main()
