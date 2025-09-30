import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig
import os
import time
from huggingface_hub import login

# It's good practice to explicitly set the cache directory
# inside the container to a user-writable location.
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface/models'
os.environ['HF_DATASETS_CACHE'] = '/tmp/huggingface/datasets'

import time
from transformers import TrainerCallback

class StepTimeCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        # Accumulator for the step times in the current 10-step batch
        self.step_times_accumulator = []

    def on_step_begin(self, args, state, control, **kwargs):
        # Record the start time of the individual step
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        # Calculate the duration of the completed step
        end_time = time.time()
        step_time = end_time - self.start_time
        
        # Add the step's time to our accumulator
        self.step_times_accumulator.append(step_time)

        # Check if the global step is a multiple of 10 and not 0
        if state.is_world_process_zero and state.global_step > 0 and state.global_step % 10 == 0:
            # Calculate the average time over the last 10 steps
            avg_time = sum(self.step_times_accumulator) / len(self.step_times_accumulator)
            
            # Determine the range of steps this average is for
            end_step = state.global_step
            start_step = end_step - len(self.step_times_accumulator) + 1
            
            print(f"Average step time for steps {start_step}-{end_step}: {avg_time:.2f} seconds")
            
            # Reset the accumulator for the next 10-step batch
            self.step_times_accumulator = []

def main():
    # --- Authenticate with Hugging Face ---
    print("Authenticating with Hugging Face...")
    try:
        hf_token = os.environ["HF_TOKEN"]
        login(token=hf_token)
        print("Hugging Face login successful.")
    except KeyError:
        print("ERROR: HF_TOKEN environment variable not set. Please ensure it's passed to the container.")
        raise
    # -----------------------------------------

    model_name = "openai/gpt-oss-20b"
    
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    
    print("Loading a small dataset for the demo.")
    dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

    training_args = SFTConfig(
        output_dir="/tmp/test_trainer",
        num_train_epochs=1,
        per_device_train_batch_size=1, 
        max_steps=100, 
        logging_steps=1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[StepTimeCallback()],
    )

    print("\nStarting training for 100 steps...")
    trainer.train()
    print("\nSUCCESS: Training step completed.")

    # Clean up the distributed process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
