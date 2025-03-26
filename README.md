# Fine-Tuning Mistral-7B-Instruct on NutritionQA

Этот проект содержит код для дообучения модели `Mistral-7B-Instruct` на датасете вопросов и ответов по питанию (`NutritionQA`), с использованием LoRA и 8-bit загрузки. Все скрипты совместимы с запуском в Google Colab.

## 🚀 Запуск в Google Colab

### 1. Клонируем репозиторий

```bash
!git clone https://github.com/kutlovskii/mistral-nutrition-finetune.git
%cd mistral-nutrition-finetune

### 2. Устанавливаем зависимости

!pip install -r requirements.txt


3. Загружаем датасет вручную

from google.colab import files
uploaded = files.upload()  # Загрузи test-00000-of-00001.parquet

4. Обучаем модель

!python src/train.py

5. Оцениваем результат

!python src/evaluate.py


Обоснование выбора

Модель: Mistral-7B-Instruct-v0.1 — открытая и эффективная модель, хорошо работает с LoRA и 8bit, подходит для fine-tuning в Colab.
Датасет: NutritionQA содержит заранее размеченные пары "вопрос-ответ", идеально подходящие для supervised fine-tuning. Второй предложенный датасет (Food Facts) не использовался, так как требует значительной дополнительной обработки.

Особенности

Используется 8-bit загрузка модели (через bitsandbytes) для экономии VRAM.
Fine-tuning выполняется через LoRA (peft) — быстро и экономично.
Работает на Google Colab с GPU (T4 и выше).
Поддерживается Perplexity-оценка модели через библиотеку evaluate.
