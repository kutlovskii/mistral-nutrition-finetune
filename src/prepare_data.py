import pandas as pd

def load_dataset(parquet_path):
    """
    Загружает и обрабатывает NutritionQA датасет из .parquet.
    Пытается найти подходящие колонки с вопросами и ответами.
    Возвращает список строк в формате 'Вопрос: ...\nОтвет: ...'
    """
    df = pd.read_parquet(parquet_path)

    # Попытка угадать названия колонок
    question_col = None
    answer_col = None

    for col in df.columns:
        lowered = col.lower()
        if "question" in lowered or "prompt" in lowered:
            question_col = col
        if "answer" in lowered or "response" in lowered:
            answer_col = col

    # Если не получилось — взять первые 2 колонки
    if question_col is None or answer_col is None:
        print("⚠️ Автоопределение не удалось. Используем первые две колонки.")
        question_col, answer_col = df.columns[:2]

    # Форматируем строки
    samples = []
    for _, row in df.iterrows():
        prompt = str(row[question_col]).strip()
        response = str(row[answer_col]).strip()
        if prompt and response:
            samples.append(f"Вопрос: {prompt}\nОтвет: {response}")
    
    return samples
