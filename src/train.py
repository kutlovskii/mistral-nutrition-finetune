import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from prepare_data import load_dataset
from auto_gptq import AutoGPTQForCausalLM

def train():
    # Путь к датасету (если в Colab — в корень загрузили вручную)
    data_path = "/content/test-00000-of-00001.parquet"

    # Загружаем и форматируем данные
    texts = load_dataset(data_path)
    dataset = Dataset.from_dict({"text": texts})

    # Имя модели
    model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Загружаем квантованную модель
    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        device_map="auto",
        use_safetensors=True,
        inject_fused_attention=False,
        trust_remote_code=True,
        revision="main"
    )

    # Подготовка модели для LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Параметры обучения
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    # Коллатор данных
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Тренер
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Обучение
    trainer.train()

if __name__ == "__main__":
    train()
