import torch
import torch.nn as nn
import math
import gc
from typing import Dict, List

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer that can be applied to any linear layer.
    
    This implements the LoRA technique where we freeze the original weights W
    and add a low-rank decomposition: W + (B @ A) where A is (r, in_features) 
    and B is (out_features, r), with r << min(in_features, out_features).
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # Low-rank matrices
        device = original_layer.weight.device
        dtype = original_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, original_layer.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank, device=device, dtype=dtype))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Scaling factor
        self.scaling = self.alpha / self.rank

        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters of the LoRA layer."""
        # Kaiming uniform is standard for Matrix A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Matrix B is initialized to zero so the adapter starts as an Identity transform
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass
        result = self.original_layer(x)
        
        lora_result = self.dropout(x) @ self.lora_A.t()  # (batch, rank)
        lora_result = lora_result @ self.lora_B.t()  # (batch, out_features)
        
        # Combine with scaling
        return result + lora_result * self.scaling


class LoRAWrapper:
    """
    Utility class to wrap a module and add LoRA adapters to specified linear layers.
    """
    def __init__(self, module: nn.Module, target_modules: List[str] = None, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        """
        Args:
            module: The module to wrap with LoRA
            target_modules: List of module names to target (e.g., ['qkv', 'proj']). 
                          If None, targets common attention projections.
            rank: LoRA rank
            alpha: LoRA alpha scaling parameter
            dropout: LoRA dropout rate
        """
        self.module = module
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        if target_modules is None:
            # Default target modules for attention layers
            target_modules = ['qkv', 'proj', 'fc1', 'fc2', 'to_q', 'to_k', 'to_v', 'to_out']
        
        self.target_modules = target_modules
        self.lora_layers: Dict[str, LoRALayer] = {}
        
        self._add_lora_to_module(module, '')
    
    def _add_lora_to_module(self, module: nn.Module, prefix: str):
        """Recursively add LoRA to target linear layers."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(target in name for target in self.target_modules):
                # Replace with LoRA layer
                lora_layer = LoRALayer(child, rank=self.rank, alpha=self.alpha, dropout=self.dropout)
                setattr(module, name, lora_layer)
                self.lora_layers[full_name] = lora_layer
            else:
                # Recursively process children
                self._add_lora_to_module(child, full_name)
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for optimization."""
        params = []
        for lora_layer in self.lora_layers.values():
            params.extend([lora_layer.lora_A, lora_layer.lora_B])
        return params
    
    def enable_lora(self):
        """Enable gradients for LoRA parameters."""
        for param in self.get_lora_parameters():
            param.requires_grad_(True)
    
    def disable_lora(self):
        """Disable gradients for LoRA parameters."""
        for param in self.get_lora_parameters():
            param.requires_grad_(False)


class LoRAModel:
    """
    Mixin class that provides LoRA (Low-Rank Adaptation) functionality.
    
    This class can be inherited by any model to add LoRA adapter management.
    Subclasses should call `_init_lora_model()` in their `__init__` method
    and implement `get_lora_target_modules()` to specify which modules to apply LoRA to.
    """
    
    def _init_lora_model(self):
        """Initialize LoRA model attributes. Call this in subclass __init__."""
        self.lora_wrappers: List[LoRAWrapper] = []
        self._lora_setup_done = False
    
    def get_lora_target_modules(self) -> List[nn.Module]:
        """
        Return the list of modules to apply LoRA adapters to.
        
        Subclasses should override this method to specify which modules
        (e.g., transformer blocks) should have LoRA adapters applied.
        
        Returns:
            List of nn.Module objects to wrap with LoRA
        """
        raise NotImplementedError(
            "Subclasses must implement get_lora_target_modules() to specify "
            "which modules should have LoRA adapters applied."
        )
    
    def setup_lora_adapters(
        self, 
        lora_rank: int, 
        lora_alpha: float, 
        lora_target_modules: List[str] = None, 
        lora_dropout: float = 0.0
    ):
        """
        Setup fresh LoRA adapters for the target modules.
        
        Always creates brand new randomly initialized LoRA parameters.
        
        Args:
            lora_rank: Rank of the LoRA decomposition
            lora_alpha: LoRA alpha scaling parameter
            lora_target_modules: List of layer names to target (e.g., ['qkv', 'proj']).
                                If None, defaults to ['qkv', 'proj', 'fc1', 'fc2']
            lora_dropout: Dropout rate for LoRA layers
        """
        # Prepare configuration
        if lora_target_modules is None:
            lora_target_modules = ['qkv', 'proj', 'fc1', 'fc2']
        
        # Always clear existing wrappers and create fresh ones
        self.lora_wrappers.clear()
        
        # Get modules to apply LoRA to
        target_modules = self.get_lora_target_modules()
        
        # Add LoRA to each target module with fresh random initialization
        for module in target_modules:
            wrapper = LoRAWrapper(
                module, 
                target_modules=lora_target_modules,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
            self.lora_wrappers.append(wrapper)
            
        self._lora_setup_done = True

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for optimization."""
        if not self._lora_setup_done:
            return []  # No LoRA parameters if not set up yet
        params = []
        for wrapper in self.lora_wrappers:
            params.extend(wrapper.get_lora_parameters())
        return params
    
    def enable_lora_gradients(self):
        """Enable gradients for all LoRA parameters."""
        if not self._lora_setup_done:
            return  # No LoRA to enable if not set up yet
        for wrapper in self.lora_wrappers:
            wrapper.enable_lora()
    
    def disable_lora_gradients(self):
        """Disable gradients for all LoRA parameters."""
        if not self._lora_setup_done:
            return  # No LoRA to disable if not set up yet
        for wrapper in self.lora_wrappers:
            wrapper.disable_lora()
        
    def reset_lora_adapters(self, merge_weights: bool = False):
        """
        Reset LoRA adapters by restoring original linear layers.
        
        Args:
            merge_weights: If True, merge LoRA adaptations into original weights permanently.
                           If False, discard LoRA adaptations and restore original weights.
        """
        if not self._lora_setup_done:
            print("No LoRA setup to reset")
            return  # Nothing to reset
        
        # Collect all LoRALayer objects for explicit cleanup
        # This ensures complete memory cleanup including the class objects themselves
        lora_objects_to_delete = []
        
        for wrapper in self.lora_wrappers:
            for layer_name, lora_layer in wrapper.lora_layers.items():
                # Get the parent module and layer name
                parts = layer_name.split('.')
                parent_module = wrapper.module
                for part in parts[:-1]:
                    parent_module = getattr(parent_module, part)
                
                final_layer_name = parts[-1]
                
                # Get the original linear layer
                original_layer = lora_layer.original_layer
                
                if merge_weights:
                    # Merge LoRA weights into original weights
                    # W_new = W_original + (lora_B @ lora_A) * scaling
                    with torch.no_grad():
                        lora_weight = lora_layer.lora_B @ lora_layer.lora_A * lora_layer.scaling
                        original_layer.weight.data += lora_weight
                
                # Restore the original linear layer first
                setattr(parent_module, final_layer_name, original_layer)
                
                # Add to deletion list for later cleanup
                lora_objects_to_delete.append(lora_layer)
        
        # Explicitly cleanup all LoRALayer objects
        for lora_layer in lora_objects_to_delete:
            # Delete LoRA parameters
            del lora_layer.lora_A
            del lora_layer.lora_B
            
            # Clear all object references
            lora_layer.original_layer = None
            lora_layer.dropout = None
            
            # Delete the object itself
            del lora_layer

        # Clear all wrapper references and delete wrapper objects
        wrappers_to_delete = list(self.lora_wrappers)
        for wrapper in wrappers_to_delete:
            wrapper.lora_layers.clear()
            # Clear wrapper's object references
            wrapper.module = None
            wrapper.target_modules = None
            # Delete the wrapper object
            del wrapper
        
        self.lora_wrappers.clear()
        self._lora_setup_done = False
        
        # Force garbage collection to ensure cleanup
        gc.collect()
