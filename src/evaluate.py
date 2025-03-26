import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
from src.prepare_data import load_dataset

def evaluate():
    # Путь к датасету и модели
    data_path = "test-00000-of-00001.parquet"
    model_path = "./outputs/final_model"

    # Загружаем тексты
    texts = load_dataset(data_path)

    # Используем только первые 100 примеров для оценки
    eval_texts = texts[:100]

    # Загружаем модель и токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)

    # Загружаем метрику
    perplexity = load("perplexity", module_type="metric")

    results = perplexity.compute(model=model, tokenizer=tokenizer, data=eval_texts)
    print(f"Perplexity: {results['perplexity']:.2f}")

if __name__ == "__main__":
    evaluate()
