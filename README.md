# Resume Domain Classifier

### Автоматическое определение профессиональной области по тексту резюме

### Постановка задачи

Цель проекта: разработать систему, которая по входному тексту резюме автоматически определяет профессиональную категорию кандидата (например, HR, IT, Banking и т.д.). Такая система позволяет ускорить первичный скрининг резюме, снизить ручную нагрузку на HR и повысить качество маршрутизации кандидатов по направлениям.

### Формат входных и выходных данных

- **Вход:** текст резюме в виде строки. Предполагается, что текст может быть заранее извлечён из PDF/HTML и очищен от служебных символов, артефактов верстки и лишних пробелов.
- **Выход:**
  1. одна предсказанная категория из конечного множества классов;
  2. распределение вероятностей по всем классам (например, top-k вероятностей), чтобы можно было интерпретировать уверенность модели.

### Метрики

Задача является многоклассовой, при этом возможен дисбаланс категорий, поэтому используются две основные метрики:

- **Accuracy** — доля резюме, классифицированных правильно; удобна как итоговая “простая” метрика качества.
- **Macro F1-score** — среднее F1 по классам, которое одинаково учитывает редкие и частые категории и лучше отражает качество при дисбалансе.

Ожидаемые ориентиры качества:

- **Accuracy:** ~0.75–0.85
- **Macro F1:** ~0.7

### Валидация и тестирование

Датасет будет разделён на **обучающую / валидационную / тестовую** части в пропорции **70% / 15% / 15%**. Разбиение выполняется **стратифицированно по классу**, чтобы все категории были представлены в каждой части. Для воспроизводимости фиксируется random seed и сохраняются параметры разбиения в конфигурации.

### Данные

**Resume Dataset (Snehaan Bhawal)** — около 2400 резюме:
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Основные особенности данных, учтённые в реализации:

- тексты резюме могут содержать артефакты форматирования и “шум”, поэтому используется препроцессинг и нормализация текста;
- категории распределены неравномерно, поэтому для оценки используется Macro F1.

### Моделирование

#### Бейзлайн-модель

В качестве базовой модели реализован классический и интерпретируемый пайплайн:

- **TF-IDF векторизация текста**
- **линейная модель классификации**

Обучение выполнено с использованием **PyTorch Lightning**.

### Внедрение (формат использования)

В проекте реализован CLI-инструмент для:

- обучения модели (`train`)
- инференса по произвольному тексту резюме (`infer`)

Пользователь вводит текст резюме строкой, а система возвращает предсказанную категорию и вероятности.

## Setup

### Установка Poetry

```bash
pip install --upgrade pip
pip install poetry
```

```bash
poetry install --with dev
```

### Установка pre-commit

```bash
poetry run pre-commit install
poetry run pre-commit run -a
```

### Запуск MLflow сервера локально

Проект логирует метрики в MLflow. Если вы хотите смотреть эксперименты локально, поднимите MLflow сервер:

```bash
poetry run mlflow server --host 127.0.0.1 --port 8080
```

После этого UI будет доступен по адресу:

- http://127.0.0.1:8080

## Train

### Подготовка данных (DVC)

```bash
poetry run dvc pull
```

### Базовый запуск обучения

```bash
poetry run resume-domain-classifier train
```

### Запуск обучения с переопределением гиперпараметров

```bash
poetry run resume-domain-classifier train --model.optimizer.lr=0.05 --train.max_epochs=10
```

## Inference

```bash
poetry run resume-domain-classifier infer "Registered nurse. Patient care, vital signs monitoring, medication administration, clinical documentation, EMR systems, inpatient and outpatient experience."
```

```bash
(venv) (base) dara@MacBook-Air-2 resume-domain-classifier % poetry run resume-domain-classifier infer "Registered nurse. Patient care, vital signs monitoring, medication administration, clinical documentation, EMR systems, inpatient and outpatient experience."
HEALTHCARE
[{'label': 'HEALTHCARE', 'prob': 0.3406183123588562}, {'label': 'FITNESS', 'prob': 0.15670211613178253}, {'label': 'ADVOCATE', 'prob': 0.13434723019599915}, {'label': 'CONSULTANT', 'prob': 0.04569370672106743}, {'label': 'AVIATION', 'prob': 0.04298420622944832}]
```

```bash
(venv) (base) dara@MacBook-Air-2 resume-domain-classifier % poetry run resume-domain-classifier infer "Mechanical engineer. CAD (SolidWorks), design of assemblies, manufacturing drawings, tolerance analysis, BOM, failure analysis, ISO standards, product testing."
ENGINEERING
[{'label': 'ENGINEERING', 'prob': 0.5498255491256714}, {'label': 'AVIATION', 'prob': 0.1409938484430313}, {'label': 'DESIGNER', 'prob': 0.12028896808624268}, {'label': 'CONSULTANT', 'prob': 0.031953517347574234}, {'label': 'APPAREL', 'prob': 0.02880905009806156}]
```
