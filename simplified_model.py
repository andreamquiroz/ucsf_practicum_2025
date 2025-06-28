import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from seer_attn import SeerAttnLlamaForCausalLM
import logging

# Configure logging
logger = logging.getLogger(__name__)

class SimpleMedicalClassifier(nn.Module):
    def __init__(self, 
             model_path,
             num_classes=2,
             threshold=5e-4,
             max_length=8192,
             use_clinical_features=True):
        """Simplified medical classifier using SeerAttention"""
        super().__init__()
        
        # Load configuration
        config = AutoConfig.from_pretrained(model_path)
        
        # Load model with SeerAttention
        logger.info(f"Loading model with device_map='auto', threshold={threshold}")
        self.base_model = SeerAttnLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            seerattn_sparsity_method='threshold',
            seerattn_threshold=threshold,
            device_map="auto"  # Use auto device map
        )
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Get hidden dimension
        self.hidden_dim = config.hidden_size
        
        # Clinical features
        self.use_clinical_features = use_clinical_features
        clinical_dim = 0
        if use_clinical_features:
            # Survival (1) + stage_grade one-hot (3)
            clinical_dim = 4
        
        # Classification head
        combined_dim = self.hidden_dim + clinical_dim if use_clinical_features else self.hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def to(self, device):
        """Custom implementation of to() that only moves classifier"""
        # Don't try to move base_model since it's using device_map="auto"
        # Only move the classifier or the cross tensor issue will happen
        self.classifier = self.classifier.to(device)
        return self
    
    def forward(self, input_ids, attention_mask, overallsurvival=None, stage_grade=None):
        # Determine classifier device
        classifier_device = next(self.classifier.parameters()).device

        try:
            # Get model outputs - forward pass with distributed model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # Extract the final token's representation and move to classifier device
            sequence_output = outputs.hidden_states[-1][:, -1, :].to(classifier_device)

            # Combine with clinical features if available
            if self.use_clinical_features and overallsurvival is not None and stage_grade is not None:
                # Convert clinical features to the right device
                overallsurvival = overallsurvival.to(classifier_device)
                stage_grade = stage_grade.to(classifier_device)

                # Concatenate clinical features
                clinical_features = torch.cat([overallsurvival.unsqueeze(1), stage_grade], dim=1)
                combined_features = torch.cat([sequence_output, clinical_features], dim=1)
                logits = self.classifier(combined_features)
            else:
                # Text-only classification
                logits = self.classifier(sequence_output)

            return logits

        except RuntimeError as e:
            if "at least two devices" in str(e):
                # Simple fallback: use mean pooling of embeddings for a basic representation
                logger.warning("Device mismatch detected. Using simplified representation.")

                # Get classifier device for consistency
                device = classifier_device

                # Get embeddings layer directly (should be on a single device)
                embed_layer = self.base_model.model.embed_tokens
                embed_device = next(embed_layer.parameters()).device

                # Move inputs to embedding device
                local_input_ids = input_ids.to(embed_device)

                # Get embeddings
                with torch.no_grad():
                    embeddings = embed_layer(local_input_ids)
                    # Simple mean pooling (not ideal but works as fallback)
                    pooled = embeddings.mean(dim=1).to(device)

                # Process with classifier
                if self.use_clinical_features and overallsurvival is not None and stage_grade is not None:
                    clinical_features = torch.cat([
                        overallsurvival.to(device).unsqueeze(1),
                        stage_grade.to(device)
                    ], dim=1)
                    combined_features = torch.cat([pooled, clinical_features], dim=1)
                    logits = self.classifier(combined_features)
                else:
                    logits = self.classifier(pooled)

                return logits
            else:
                # Re-raise other errors
                raise e