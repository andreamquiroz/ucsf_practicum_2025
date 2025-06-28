import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
from datetime import datetime

# Import the custom mods :D
from bio_simplified_dataset import BiomedicalSimpleMedicalDataset
from bio_simplified_model import BiomedicalSeerClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("biomedical_training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_gpu_devices():
    """Setup GPU devices - same as the successful runs"""
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        print("CUDA_VISIBLE_DEVICES not set. Using all available GPUs.")
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Found {gpu_count} GPUs. Setting CUDA_VISIBLE_DEVICES to use first 4 or fewer.")
            visible_devices = ",".join(str(i) for i in range(min(4, gpu_count)))
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    
    print(f"Using GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    
    for i in range(torch.cuda.device_count()):
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        free_memory_gb = free_memory / 1e9
        print(f"GPU {i}: Free memory: {free_memory_gb:.2f} GB")

def save_metrics_to_file(output_dir, epoch, train_metrics, test_metrics):
    """
    Save metrics to both CSV and JSON files for easy tracking in tmux sessions
    TMUX DONT BE SAVING!! make function
    """
    # Create metrics directory
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save to CSV file
    csv_file = os.path.join(metrics_dir, "epoch_metrics.csv")
    
    # Check if file exists to write header
    write_header = not os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'epoch', 'timestamp', 
                'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
                'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc', 'test_loss'
            ])
        
        writer.writerow([
            epoch, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], train_metrics['f1'],
            test_metrics['accuracy'], test_metrics['precision'], test_metrics['recall'], 
            test_metrics['f1'], test_metrics['auc'], test_metrics['loss']
        ])
    
    # Save detailed metrics for this epoch to JSON
    # Convert confusion matrix from numpy array to list for JSON serialization
    test_metrics_copy = test_metrics.copy()
    if 'confusion_matrix' in test_metrics_copy:
        test_metrics_copy['confusion_matrix'] = test_metrics_copy['confusion_matrix'].tolist()
    
    epoch_data = {
        'epoch': epoch,
        'timestamp': datetime.now().isoformat(),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics_copy
    }
    
    json_file = os.path.join(metrics_dir, f"epoch_{epoch}_detailed.json")
    with open(json_file, 'w') as f:
        json.dump(epoch_data, f, indent=2)
    
    # Also save to a summary file that's easy to tail in tmux
    summary_file = os.path.join(metrics_dir, "training_summary.txt")
    with open(summary_file, 'a') as f:
        f.write(f"EPOCH {epoch} | {datetime.now().strftime('%H:%M:%S')} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Test Acc: {test_metrics['accuracy']:.4f} | "
                f"Test F1: {test_metrics['f1']:.4f} | "
                f"Test AUC: {test_metrics['auc']:.4f}\n")

def create_confusion_matrix_plot(cm, epoch, output_dir, accuracy, f1, auc):
    """Create and save confusion matrix plot"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Epoch {epoch}\n'
              f'Accuracy: {accuracy:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}')
    
    # Add text annotations with percentages
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"confusion_matrix_epoch_{epoch}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    return plot_file

def evaluate_model_safely(model, dataloader, device, criterion, clinical_features_used=True):
    """Evaluate model - same as the successful approach but with metric file saving cus last time
    that was an issue with tmux"""
    was_training = model.training
    model.train()  # Use training mode for compatibility
    
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_size = batch['input_ids'].size(0)
            batch_preds = []
            batch_probs = []
            
            for i in range(batch_size):
                input_ids = batch['input_ids'][i:i+1].to(device)
                attention_mask = batch['attention_mask'][i:i+1].to(device)
                
                overallsurvival = None
                if clinical_features_used and 'overallsurvival' in batch:
                    overallsurvival = batch['overallsurvival'][i:i+1].to(device).float()
                    
                stage_grade = None
                if clinical_features_used and 'stage_grade' in batch:
                    stage_grade = batch['stage_grade'][i:i+1].to(device).float()
                
                try:
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        overallsurvival=overallsurvival,
                        stage_grade=stage_grade
                    )
                    
                    probs = torch.softmax(logits, dim=1)
                    pred = torch.argmax(logits, dim=1).item()
                    
                    batch_preds.append(pred)
                    batch_probs.append(probs[0, 1].item())
                
                except Exception as e:
                    print(f"Error processing example {i}: {str(e)}")
                    batch_preds.append(0)
                    batch_probs.append(0.5)
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch['label'].cpu().numpy())
            all_probs.extend(batch_probs)
            
            # Calculate loss
            try:
                labels = batch['label'].to(device)
                pseudo_logits = torch.zeros((batch_size, 2), device=device)
                for j, prob in enumerate(batch_probs):
                    pseudo_logits[j, 0] = 1 - prob
                    pseudo_logits[j, 1] = prob
                
                loss = criterion(pseudo_logits, labels)
                test_loss += loss.item() * batch_size
            except Exception as e:
                print(f"Could not calculate loss: {str(e)}")
    
    # Reset model state
    if not was_training:
        model.eval()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        print(f"Could not calculate AUC: {str(e)}")
        auc = 0.5
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': test_loss / len(dataloader.dataset),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }

def train_biomedical_model(
    biomedical_model_path="ContactDoctor/Bio-Medical-Llama-3-8B-CoT-012025",
    seer_model_path="SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates",
    train_file="breast_train_30_0.pkl",
    test_file="breast_test_30_0.pkl",
    output_dir="./biomedical_output",
    max_length=8192,
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_epochs=3,
    use_clinical_features=True
):
    """
    Training function for biomedical model - same structure as the successful runs
    with added metric file tracking and visualizations
    """
    
    # Setup GPUs
    setup_gpu_devices()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load biomedical tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        biomedical_model_path, 
        padding_side="left", 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loaded biomedical tokenizer from {biomedical_model_path}")
    logger.info(f"Max length: {max_length}")
    
    # Create datasets with the successful chunking strategy
    train_dataset = BiomedicalSimpleMedicalDataset(
        train_file, tokenizer, max_length=max_length, use_clinical_features=use_clinical_features
    )
    test_dataset = BiomedicalSimpleMedicalDataset(
        test_file, tokenizer, max_length=max_length, use_clinical_features=use_clinical_features
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Create biomedical SeerAttention model
    model = BiomedicalSeerClassifier(
        biomedical_model_path=biomedical_model_path,
        seer_model_path=seer_model_path,
        num_classes=2,
        threshold=5e-4,
        max_length=max_length,
        use_clinical_features=use_clinical_features
    )
    
    # Move classifier to device
    model.classifier = model.classifier.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Create scheduler
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Track best accuracy
    best_accuracy = 0.0
    
    # Save initial configuration
    config_file = os.path.join(output_dir, "training_config.json")
    config_data = {
        'biomedical_model_path': biomedical_model_path,
        'seer_model_path': seer_model_path,
        'train_file': train_file,
        'test_file': test_file,
        'max_length': max_length,
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'use_clinical_features': use_clinical_features,
        'start_time': datetime.now().isoformat()
    }
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Starting Epoch {epoch}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} (Training)")
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            overallsurvival = batch.get('overallsurvival', None)
            if overallsurvival is not None:
                overallsurvival = overallsurvival.to(device).float()
            
            stage_grade = batch.get('stage_grade', None)
            if stage_grade is not None:
                stage_grade = stage_grade.to(device).float()
            
            # Forward pass
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                overallsurvival=overallsurvival,
                stage_grade=stage_grade
            )
            
            # Calculate loss
            loss = criterion(logits, labels) / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Step optimizer
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Track metrics
            train_loss += loss.item() * gradient_accumulation_steps
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
        
        # Calculate training metrics
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_labels, train_preds, average='binary', zero_division=0
        )
        
        train_metrics = {
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1,
            'loss': train_loss / len(train_loader)
        }
        
        # Evaluation
        model.eval()
        test_metrics = evaluate_model_safely(
            model=model,
            dataloader=test_loader,
            device=device,
            criterion=criterion, 
            clinical_features_used=use_clinical_features
        )
        
        # Log metrics
        logger.info(f"Epoch {epoch} - Training: Accuracy={train_accuracy:.4f}, F1={train_f1:.4f}")
        logger.info(f"Epoch {epoch} - Testing: Accuracy={test_metrics['accuracy']:.4f}, "
                   f"F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
        
        # Save metrics to files (this solves your tmux scrolling issue!)
        save_metrics_to_file(output_dir, epoch, train_metrics, test_metrics)
        
        # Create and save confusion matrix
        cm_file = create_confusion_matrix_plot(
            test_metrics['confusion_matrix'], epoch, output_dir,
            test_metrics['accuracy'], test_metrics['f1'], test_metrics['auc']
        )
        logger.info(f"Confusion matrix saved to: {cm_file}")
        
        # Save best model
        if test_metrics['accuracy'] > best_accuracy:
            best_accuracy = test_metrics['accuracy']
            
            # Check disk space before saving
            import shutil
            free_space = shutil.disk_usage(output_dir).free
            free_space_gb = free_space / (1024**3)
            logger.info(f"Available disk space: {free_space_gb:.2f} GB")
            
            if free_space_gb < 5.0:  # Less than 5GB free
                logger.warning(f"Low disk space: {free_space_gb:.2f} GB. Skipping model save.")
            else:
                try:
                    # Save only the classifier weights (much smaller than full model)
                    model_filename = f"best_biomedical_classifier.pt"
                    classifier_state = {
                        'classifier': model.classifier.state_dict(),
                        'epoch': epoch,
                        'accuracy': best_accuracy
                    }
                    
                    model_path = os.path.join(output_dir, model_filename)
                    torch.save(classifier_state, model_path)
                    logger.info(f"Classifier weights saved to: {model_path}")
                    
                    # Also save a checkpoint with minimal info
                    checkpoint = {
                        'epoch': epoch,
                        'accuracy': best_accuracy,
                        'model_config': {
                            'biomedical_model_path': biomedical_model_path,
                            'seer_model_path': seer_model_path,
                            'threshold': 5e-4,
                            'use_clinical_features': use_clinical_features
                        }
                    }
                    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Checkpoint saved to: {checkpoint_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to save model: {str(e)}")
                    logger.info("Continuing training without saving model...")
            
            # Save best model info (this is small and should always work hopefuly)
            try:
                best_info = {
                    'epoch': epoch,
                    'accuracy': best_accuracy,
                    'f1': test_metrics['f1'],
                    'auc': test_metrics['auc'],
                    'precision': test_metrics['precision'],
                    'recall': test_metrics['recall'],
                    'timestamp': datetime.now().isoformat(),
                    'disk_space_gb': free_space_gb
                }
                with open(os.path.join(output_dir, "best_model_info.json"), 'w') as f:
                    json.dump(best_info, f, indent=2)
                
                logger.info(f"New best model info saved with accuracy: {best_accuracy:.4f}")
            except Exception as e:
                logger.error(f"Failed to save model info: {str(e)}")
    
    logger.info("Training completed!")
    logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
    logger.info(f"All metrics saved to: {os.path.join(output_dir, 'metrics')}")
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train biomedical classifier with SeerAttention")
    parser.add_argument("--biomedical_model_path", default="ContactDoctor/Bio-Medical-Llama-3-8B-CoT-012025")
    parser.add_argument("--seer_model_path", default="SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates")
    parser.add_argument("--train_file", default="/data/datasets/outcome/breast/breast_train_30_0.pkl")
    parser.add_argument("--test_file", default="/data/datasets/outcome/breast/breast_test_30_0.pkl") 
    parser.add_argument("--output_dir", default="./biomedical_output")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--no_clinical_features", action="store_false", dest="use_clinical_features")
    
    args = parser.parse_args()
    
    train_biomedical_model(
        biomedical_model_path=args.biomedical_model_path,
        seer_model_path=args.seer_model_path,
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        use_clinical_features=args.use_clinical_features
    )