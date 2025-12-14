

import torch

                      
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                       
DATASET = 'CIFAR10'                                     
DATA_DIR = './data'
BATCH_SIZE = 128
NUM_WORKERS = 4

                     
MODEL_NAME = 'resnet18'                                                  
NUM_CLASSES = 10                                           
PRETRAINED = True

                        
EPOCHS = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

                      
ATTACK_EPSILON = 0.03                                  
ATTACK_ALPHA = 0.01                                     
PGD_ITERATIONS = 20                              
CW_CONFIDENCE = 0                                
CW_LEARNING_RATE = 0.01
CW_MAX_ITERATIONS = 1000
CW_BINARY_SEARCH_STEPS = 9

                       
ADV_TRAINING_RATIO = 0.5                                             
DISTILLATION_TEMP = 20                                            
JPEG_QUALITY = 75                                   
BIT_DEPTH = 5                                 

                          
EVAL_BATCH_SIZE = 100
NUM_EVAL_SAMPLES = 1000                                 
SAVE_EXAMPLES = True
RESULTS_DIR = './results'
CHECKPOINT_DIR = './checkpoints'

                             
PLOT_DPI = 150
FIGURE_SIZE = (12, 8)
NUM_EXAMPLES_TO_PLOT = 5
