Explanation: Loading LoRA and Base Models for Evaluation
I've created a comprehensive guide in claudedocs/model_loading_for_evaluation.md. Here's a quick summary:
Three Model Loading Scenarios
1. Base Model Only (No LoRA)

from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-multimodal-instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
Use for baseline comparison
No PEFT library needed
Direct loading
2. Fine-tuned LoRA Model (After Training) ⭐ Most Common

from peft import PeftModel

# Load base model first
base_model = AutoModelForCausalLM.from_pretrained(...)

# Then load LoRA adapters from checkpoint
model = PeftModel.from_pretrained(
    base_model,
    "src/output/paper_r64/checkpoint-120",  # Checkpoint path
    is_trainable=False  # Evaluation mode
)
Two-step process: base model + LoRA adapters
Use PeftModel.from_pretrained() to load trained adapters
Set is_trainable=False for evaluation
3. Pretrained LoRA Model (Before Training)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-multimodal-instruct",
    torch_dtype=torch.bfloat16  # Pretrained uses BF16
)
# Pretrained LoRA weights already included in model
LoRA weights built into the pretrained model
No separate adapter loading needed
Key Differences
Method	Import PEFT?	Steps	Use Case
Base	❌ No	1 (direct load)	Baseline
Fine-tuned	✅ Yes	2 (base + adapters)	Your trained model
Pretrained	❌ No	1 (direct load)	Before training
Complete Example

# Load your fine-tuned model
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# Step 1: Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-multimodal-instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Step 2: Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "src/output/paper_r64/checkpoint-120",
    is_trainable=False
)

# Step 3: Evaluate
model.eval()
evaluate_model(model, test_dataset)
The full guide includes error handling, best practices, and a complete evaluation script comparing all three scenarios!