

import torch
import os
from tqdm import tqdm
import config
from .metrics import calculate_all_metrics
from .visualization import (
    plot_attack_comparison,
    plot_defense_comparison,
    plot_robustness_curve,
    plot_adversarial_examples
)


class RobustnessEvaluator:
    
    
    def __init__(self, model, device=None):
        
        self.model = model
        self.device = device if device else config.DEVICE
        self.model.to(self.device)
        self.results = {}
    
    def evaluate_attack(self, attack_name, attack_fn, data_loader, attack_params=None):
        
        print(f"\nEvaluating {attack_name}...")
        
        if attack_params is None:
            attack_params = {}
        
        metrics = calculate_all_metrics(
            self.model, data_loader, attack_fn, attack_params, self.device
        )
        
        self.results[attack_name] = metrics
        
        print(f"{attack_name} Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
    
    def evaluate_multiple_attacks(self, attacks_dict, data_loader):
        
        for attack_name, (attack_fn, attack_params) in attacks_dict.items():
            self.evaluate_attack(attack_name, attack_fn, data_loader, attack_params)
        
        return self.results
    
    def evaluate_epsilon_robustness(self, attack_fn, data_loader, epsilons, attack_params=None):
        
        if attack_params is None:
            attack_params = {}
        
        accuracies = []
        
        for epsilon in tqdm(epsilons, desc="Evaluating epsilon robustness"):
            params = attack_params.copy()
            params['epsilon'] = epsilon
            
            metrics = calculate_all_metrics(
                self.model, data_loader, attack_fn, params, self.device
            )
            accuracies.append(metrics['robust_accuracy'])
        
        return epsilons, accuracies
    
    def generate_report(self, save_dir=None):
        
        if save_dir is None:
            save_dir = config.RESULTS_DIR
        
        os.makedirs(save_dir, exist_ok=True)
        
                              
        report_path = os.path.join(save_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ADVERSARIAL ROBUSTNESS EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for attack_name, metrics in self.results.items():
                f.write(f"\n{attack_name}\n")
                f.write("-" * 80 + "\n")
                for key, value in metrics.items():
                    f.write(f"{key:30s}: {value:.4f}\n")
                f.write("\n")
        
        print(f"\nReport saved to {report_path}")
        
                                  
        if len(self.results) > 1:
            plot_path = os.path.join(save_dir, 'attack_comparison.png')
            plot_attack_comparison(self.results, save_path=plot_path)
        
        return report_path
    
    def save_adversarial_examples(self, attack_fn, data_loader, num_batches=1, 
                                  attack_params=None, denormalize_fn=None, save_dir=None):
        
        if attack_params is None:
            attack_params = {}
        if save_dir is None:
            save_dir = config.RESULTS_DIR
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.eval()
        
        for batch_idx, (images, labels) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            images, labels = images.to(self.device), labels.to(self.device)
            
                                           
            adversarial_images = attack_fn(
                self.model, images, labels, device=self.device, **attack_params
            )
            
                             
            with torch.no_grad():
                outputs = self.model(adversarial_images)
                _, predictions = torch.max(outputs, 1)
            
                                
            save_path = os.path.join(save_dir, f'adversarial_examples_batch_{batch_idx}.png')
            plot_adversarial_examples(
                images, adversarial_images, labels, predictions,
                num_examples=min(5, images.size(0)),
                denormalize_fn=denormalize_fn,
                save_path=save_path
            )


def compare_models(models, model_names, attack_fn, data_loader, attack_params=None, device=None):
    
    if device is None:
        device = config.DEVICE
    
    results = {}
    
    for model, name in zip(models, model_names):
        print(f"\nEvaluating {name}...")
        evaluator = RobustnessEvaluator(model, device)
        metrics = evaluator.evaluate_attack(name, attack_fn, data_loader, attack_params)
        results[name] = metrics
    
    return results
