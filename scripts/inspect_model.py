from transformers import AutoModelForCausalLM

model_name = "openai/gpt-oss-20b"
print(f"--- Inspecting model architecture for: {model_name} ---")
model = AutoModelForCausalLM.from_pretrained(model_name)
print(model)
print("\n--- End of model architecture ---")


