import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from typing import Dict, Any

from src.models.base import BaseModelWrapper
# from training.balancing import DataBalancer
from src.training.objectives import ObjectivesHelper

class TrainingLoop:
    """
    Pure PyTorch training loop. Handles DataLoaders, Optimizers, and Loss functions.
    """
    def __init__(self, model: BaseModelWrapper, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        self.tuning_mode = self.config.get("tuning_mode", "sft")
        self.mix_real = self.config.get("mix_real_data", False)
        
        # Inherit device from model (which has robust 'auto' detection)
        self.device = getattr(self.model, 'device', self.config.get("device", "cpu"))
        
        # Inject the ObjectivesHelper to control loss functions
        self.objectives = ObjectivesHelper(device=self.device)

    def run(self, X_train: pd.DataFrame, y_train: pd.Series, X_real=None, y_real=None):
        """
        Executes the training process.
        """
        # if self.mix_real and X_real is not None and y_real is not None:
        #     augmenter = DataBalancer(strategy="concat")
        #     X_train, y_train = augmenter.mix(X_train, y_train, X_real, y_real)

        if self.tuning_mode == "sft":
            # Check which version of the model we are working with
            is_v2 = self.model.__class__.__name__ == 'TabPFNModelV2'
            
            if is_v2:
                print(">>> Triggering Native TabPFN V2 SFT...")
                epochs = self.config.get('ft_epochs', 10)
                # The value 1e-9 is too small for actual fine-tuning, setting it to 1e-5
                lr = self.config.get('ft_learning_rate', 1e-5) 
                # Delegate SFT to the wrapper's native method
                self.model.fine_tune(X_train, y_train, epochs=epochs, learning_rate=lr)
                
            else:
                print(">>> Triggering Pure PyTorch SFT for V1...")
                self._run_pytorch_loop(X_train, y_train)
                # Context loading is mandatory after updating weights
                self.model.fit_context(X_train, y_train)
                
        elif self.tuning_mode == "icl":
            print(">>> Triggering ICL (In-Context Learning) loop...")
            self.model.fit_context(X_train, y_train)
        else:
            raise ValueError(f"Unknown tuning mode: {self.tuning_mode}")
        
        return self.model

    def _run_pytorch_loop(self, X: pd.DataFrame, y: pd.Series):
        """
        The core PyTorch execution logic with support for both V1 and V2 TabPFN.
        """
        if not hasattr(self.model, 'get_pytorch_model'):
            raise NotImplementedError("Model does not expose get_pytorch_model() for pure PyTorch loop.")

        # 1. Setup Data
        epochs = self.config.get('ft_epochs', 10)
        # FIX: The default LR of 1e-9 was too small, changing to 1e-5
        lr = self.config.get('ft_learning_rate', 1e-5) 
        batch_size = self.config.get('ft_batch_size', 128)

        # CHECK: If this is V2, use its own preprocessor
        is_v2 = hasattr(self.model, 'prepare_v2_dataloader')
        
        if is_v2:
            print("      [V2 Mode] Using TabPFNv2 native preprocessor...")
            dataloader = self.model.prepare_v2_dataloader(X, y, max_data_size=batch_size)
        else:
            print("      [V1/Standard Mode] Using standard PyTorch DataLoader...")
            X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y.values, dtype=torch.long).to(self.device)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 2. Setup Model & Optimizer
        pytorch_model = self.model.get_pytorch_model()
        pytorch_model.train()
        
        # --- LAYER FREEZING LOGIC ---
        trainable_layers = self.config.get('ft_trainable_layers', 'all')
        
        if trainable_layers != 'all':
            n_layers = int(trainable_layers)
            print(f"      [Freezing] Unfreezing only the last {n_layers} layers.")
            for param in pytorch_model.parameters():
                param.requires_grad = False
                
            if hasattr(pytorch_model, 'encoder') and hasattr(pytorch_model.encoder, 'layers'):
                total_layers = len(pytorch_model.encoder.layers)
                for i in range(max(0, total_layers - n_layers), total_layers):
                    for param in pytorch_model.encoder.layers[i].parameters():
                        param.requires_grad = True
            else:
                modules = list(pytorch_model.children())
                for module in modules[-n_layers:]:
                    for param in module.parameters():
                        param.requires_grad = True
        else:
            print("      [Freezing] Full Fine-Tuning enabled.")
        # ----------------------------

        trainable_params = [p for p in pytorch_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

        # 3. Epoch Loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Attention: now iterating over a generic batch
            for batch in dataloader:
                optimizer.zero_grad()
                
                # BRANCHING LOGIC: V1 vs V2
                if is_v2:
                    # TabPFN v2 expects a dictionary (one argument)
                    logits, targets = self.model.forward_pass(batch)
                    targets = targets.to(self.device).long()
                elif hasattr(self.model, 'forward_pass'):
                    # TabPFN v1 expects X and y (two arguments)
                    batch_X, batch_y = batch
                    if batch_X.shape[0] < 2:
                        continue # Skip batches of 1 element
                    logits, targets = self.model.forward_pass(batch_X, batch_y)
                else:
                    # Universal fallback for other models
                    batch_X, batch_y = batch
                    logits = pytorch_model(batch_X)
                    targets = batch_y
                
                loss = self.objectives.compute_cross_entropy(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
            # Protection against division by zero if generator returns 0 batches
            num_batches = max(1, len(list(dataloader)) if is_v2 else len(dataloader))
            avg_loss = epoch_loss / num_batches
            print(f"      Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            
        print("Pure PyTorch weights update complete.")
        pytorch_model.eval()