"""
Fine-tuning script for product recommendation LLM using PEFT/LoRA
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
from datasets import load_dataset
import json

def load_training_data(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def fine_tune_model(
    base_model: str = "mistralai/Mistral-7B-v0.1",
    output_dir: str = "./models/product-llm",
    epochs: int = 3
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch"
    )

    dataset = load_dataset("json", data_files={"train": "data/products.json"})

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seq_length=512
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    fine_tune_model()
