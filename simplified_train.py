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
from seer_attn import SeerAttnLlamaForCausalLM
# Import your custom modules
from simple_dataset import SimpleMedicalDataset
from simplified_model import SimpleMedicalClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_gpu_capacity(model_path, max_length):
    """Test if GPUs can handle the maximum context length"""
    try:
        logger.info(f"Testing GPU capacity for context length {max_length}")
        # Load configuration
        config = AutoConfig.from_pretrained(model_path)
        base_model = config.base_model
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, padding_side="left", trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with auto device mapping
        model = SeerAttnLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            seerattn_sparsity_method='threshold',
            seerattn_threshold=1e-3,  # Higher threshold for memory test
            device_map="auto"
        )
        
        # Create dummy input on CPU
        input_ids = torch.zeros((1, max_length), dtype=torch.long)
        attention_mask = torch.ones((1, max_length), dtype=torch.long)
        
        # Try a forward pass
        with torch.no_grad():
            # Only return the last_hidden_state, not all hidden states
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False  # Change to False to save memory
            )
        
        # If we get here, the model can handle the context length
        logger.info(f"GPU capacity test successful for context length {max_length}")
        return max_length
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            # If max_length is already small, return it
            if max_length <= 8192:
                return 8192
            
            # Try with half the context length
            reduced_length = max_length // 2
            logger.info(f"Reducing context length to {reduced_length} and retrying...")
            return test_gpu_capacity(model_path, reduced_length)
        elif "at least two devices" in str(e):
            # If there's a device mismatch in the capacity test, 
            # we'll still try with this length in the actual model
            logger.warning(f"Device mismatch in capacity test for length {max_length}.")
            logger.warning("Will still attempt to use this length with properly configured model.")
            return max_length
        else:
            # For other errors, print and return a conservative value
            logger.error(f"Error during capacity test: {str(e)}")
            return 8192

# Set visible devices based on environment variable or default to all
def setup_gpu_devices():
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        print("CUDA_VISIBLE_DEVICES not set. Using all available GPUs.")
        # Count available GPUs
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Found {gpu_count} GPUs. Setting CUDA_VISIBLE_DEVICES to use first 4 or fewer.")
            # Limit to first 4 or fewer GPUs
            visible_devices = ",".join(str(i) for i in range(min(4, gpu_count)))
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    
    # Print which GPUs are being used
    print(f"Using GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    
    # Check memory on each visible GPU
    for i in range(torch.cuda.device_count()):
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        free_memory_gb = free_memory / 1e9
        print(f"GPU {i}: Free memory: {free_memory_gb:.2f} GB")

# Call this function early in the script
setup_gpu_devices()

# Add this function above your train_model function
def evaluate_model_safely(model, dataloader, device, criterion, clinical_features_used=True):
    """Evaluate in training mode to avoid problematic attention mechanisms"""
    # Set to training mode but keep track of original state
    was_training = model.training
    model.train()  # Set to training mode (which uses a different forward path)
    
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():  # Still disable gradients
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Process one example at a time to avoid batch processing issues
            batch_size = batch['input_ids'].size(0)
            batch_preds = []
            batch_probs = []
            
            for i in range(batch_size):
                # Get a single example
                input_ids = batch['input_ids'][i:i+1].to(device)
                attention_mask = batch['attention_mask'][i:i+1].to(device)
                
                # Get clinical features if needed
                overallsurvival = None
                if clinical_features_used and 'overallsurvival' in batch:
                    overallsurvival = batch['overallsurvival'][i:i+1].to(device).float()
                    
                stage_grade = None
                if clinical_features_used and 'stage_grade' in batch:
                    stage_grade = batch['stage_grade'][i:i+1].to(device).float()
                
                try:
                    # Forward pass (should work since we're in training mode)
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        overallsurvival=overallsurvival,
                        stage_grade=stage_grade
                    )
                    
                    # Get predictions
                    probs = torch.softmax(logits, dim=1)
                    pred = torch.argmax(logits, dim=1).item()
                    
                    batch_preds.append(pred)
                    batch_probs.append(probs[0, 1].item())  # Probability of class 1
                
                except Exception as e:
                    print(f"Error processing example {i}: {str(e)}")
                    # Default to most common class if there's an error
                    batch_preds.append(0)
                    batch_probs.append(0.5)
            
            # Store predictions and labels
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
    
    # Reset model to its original training state
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
    
    # Create confusion matrix
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

def train_model(
    model_path="SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates",
    train_file="breast_train_60_0.pkl",
    test_file="breast_test_60_0.pkl",
    output_dir="./output",
    start_length=8192,  # Starting context length
    target_length=65536,  # Target context length
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_epochs=3,
    use_clinical_features=True
):
    """Training function with progressive context length scaling"""
    # Store results from each stage
    stage_results = []
    
    # Test available GPU memory to determine max possible context length
    max_possible_length = test_gpu_capacity(model_path, target_length)
    if max_possible_length < target_length:
        logger.warning(f"Not enough GPU memory for {target_length} tokens. "
                      f"Reducing to {max_possible_length}.")
        target_length = max_possible_length
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    config = AutoConfig.from_pretrained(model_path)
    base_model = config.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Progressive training stages
    context_lengths = [start_length]
    if target_length > start_length:
        # Add intermediate steps (e.g., 8192 -> 16384 -> 32768 -> 65536)
        while context_lengths[-1] * 2 < target_length:
            context_lengths.append(context_lengths[-1] * 2)
        context_lengths.append(target_length)
    
    logger.info(f"Progressive training with context lengths: {context_lengths}")
    
    # Train with progressively increasing context lengths
    best_model_state = None
    best_overall_accuracy = 0.0
    
    for stage, max_length in enumerate(context_lengths):
        try:
            logger.info(f"Starting training stage {stage+1}/{len(context_lengths)} with context length {max_length}")
            
            # Adjust gradient accumulation based on context length
            adjusted_grad_accum = gradient_accumulation_steps
            if max_length > 16384:
                adjusted_grad_accum = gradient_accumulation_steps * 2
            if max_length > 32768:
                adjusted_grad_accum = gradient_accumulation_steps * 4
            
            logger.info(f"Using gradient accumulation steps: {adjusted_grad_accum}")
            
            # Create datasets with current max_length
            train_dataset = SimpleMedicalDataset(
                train_file, tokenizer, max_length=max_length, use_clinical_features=use_clinical_features
            )
            test_dataset = SimpleMedicalDataset(
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
            
            # Create model with appropriate configuration for current context length
            model = SimpleMedicalClassifier(
                model_path=model_path,
                num_classes=2,
                threshold=5e-4 if max_length < 32768 else 1e-3,  # Adjust sparsity
                max_length=max_length,
                use_clinical_features=use_clinical_features
            )
            
            # If we have a previous best model, load its classifier weights
            if best_model_state is not None:
                model.classifier.load_state_dict(best_model_state)
                
            # move classifier to device
            model.classifier = model.classifier.to(device)
            
            # Move to device
            model = model.to(device)
            
            # Create optimizer
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                weight_decay=0.01
            )
            
            # Create scheduler
            total_steps = len(train_loader) * num_epochs // adjusted_grad_accum
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=100,
                num_training_steps=total_steps
            )
            
            # Loss function
            criterion = nn.CrossEntropyLoss()
            
            # Track best accuracy for this context length
            best_stage_accuracy = 0.0
            
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
                    # Move data to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Clinical features if needed
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
                    loss = criterion(logits, labels) / adjusted_grad_accum
                    
                    # Backward pass
                    loss.backward()
                    
                    # Step optimizer every gradient_accumulation_steps
                    if (step + 1) % adjusted_grad_accum == 0 or (step + 1 == len(train_loader)):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    # Track metrics
                    train_loss += loss.item() * adjusted_grad_accum
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    train_preds.extend(preds)
                    train_labels.extend(labels.cpu().numpy())
                    
                    # Update progress bar
                    progress_bar.set_postfix({"loss": loss.item() * adjusted_grad_accum})
                
                # Calculate training metrics
                train_accuracy = accuracy_score(train_labels, train_preds)
                train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
                    train_labels, train_preds, average='binary'
                )
                
                # Evaluation
                model.eval()
                test_metrics = evaluate_model_safely(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    criterion=criterion, 
                    clinical_features_used=use_clinical_features
                )
                
                # Log test metrics
                logger.info(f"Epoch {epoch} - Testing: Loss={test_metrics['loss']:.4f}, "
                        f"Accuracy={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
                
                # Save best model for this stage
                if test_metrics['accuracy'] > best_stage_accuracy:
                    best_stage_accuracy = test_metrics['accuracy']
                    best_stage_f1 = test_metrics['f1']
                    best_stage_auc = test_metrics['auc']
                    
                    # Save model for this specific context length
                    model_filename = f"best_model_length_{max_length}.pt"
                    torch.save(model.state_dict(), os.path.join(output_dir, model_filename))
                    logger.info(f"New best model for context length {max_length} saved with accuracy: {best_stage_accuracy:.4f}")
                    
                    # Update best overall model if this is better
                    if best_stage_accuracy > best_overall_accuracy:
                        best_overall_accuracy = best_stage_accuracy
                        best_model_state = model.classifier.state_dict()
                        
                        # Save as overall best model
                        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
                        logger.info(f"New overall best model saved with accuracy: {best_overall_accuracy:.4f}")
                    
                    # Plot confusion matrix
                    cm = test_metrics['confusion_matrix']
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title(f'Confusion Matrix - Context Length {max_length} - Epoch {epoch}')
                    plt.savefig(os.path.join(output_dir, f"confusion_matrix_length_{max_length}_epoch_{epoch}.png"))
                    plt.close()
            
            # Record results for this stage
            stage_results.append({
                'context_length': max_length,
                'best_accuracy': best_stage_accuracy,
                'best_f1': best_stage_f1,
                'best_auc': best_stage_auc
            })
            logger.info(f"Completed stage with context length {max_length}. Best accuracy: {best_stage_accuracy:.4f}")
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"Out of memory error with context length {max_length}. Stopping progression.")
                break
            else:
                # Re-raise other errors
                raise e
    
    # Print summary of performance across context lengths
    logger.info("=== Performance across context lengths ===")
    best_stage = max(stage_results, key=lambda x: x['best_accuracy']) if stage_results else None
    
    for result in stage_results:
        logger.info(f"Length {result['context_length']}: Accuracy={result['best_accuracy']:.4f}, F1={result['best_f1']:.4f}, AUC={result['best_auc']:.4f}")
    
    if best_stage:
        logger.info(f"Best context length: {best_stage['context_length']} with accuracy {best_stage['best_accuracy']:.4f}")
    
    logger.info("Training completed!")
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train medical classifier with SeerAttention")
    parser.add_argument("--model_path", default="SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates")
    parser.add_argument("--train_file", default="/data/datasets/outcome/breast/breast_train_365_0.pkl")
    parser.add_argument("--test_file", default="/data/datasets/outcome/breast/breast_test_365_0.pkl") 
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--start_length", type=int, default=8192)
    parser.add_argument("--target_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--no_clinical_features", action="store_false", dest="use_clinical_features")
    
    args = parser.parse_args()
    
    train_model(
        model_path=args.model_path,
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        start_length=args.start_length,
        target_length=args.target_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        use_clinical_features=args.use_clinical_features
    )