"""
Transformers Compatibility Trainer for Phi-4

Fixes incompatibility between newer transformers library and Phi-4's forward() method.

Issue: Newer transformers (>4.46) passes `num_items_in_batch` parameter to model.forward(),
       but Phi-4MMForCausalLM doesn't accept this parameter.

Solution: Override compute_loss() to remove the parameter before calling model.
"""

from transformers import Trainer
import torch


class CompatTrainer(Trainer):
    """
    Trainer with Phi-4 compatibility fixes

    Removes `num_items_in_batch` parameter that Phi-4 doesn't accept.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to remove num_items_in_batch from model call.

        Args:
            model: The model to compute loss for
            inputs: Input dictionary for the model
            return_outputs: Whether to return outputs along with loss
            num_items_in_batch: Ignored (Phi-4 doesn't accept this)

        Returns:
            loss: The computed loss
            outputs: (optional) Model outputs if return_outputs=True
        """
        # Handle label smoothing if configured
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # Call model WITHOUT num_items_in_batch parameter
        # This is the key fix - Phi-4's forward() doesn't accept it
        outputs = model(**inputs)

        # Save past state if needed for generation
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # Compute loss
        if labels is not None:
            if self.label_smoother is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # Standard loss from model outputs
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        else:
            # Loss from model when no labels provided
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = outputs[0]

        return (loss, outputs) if return_outputs else loss
