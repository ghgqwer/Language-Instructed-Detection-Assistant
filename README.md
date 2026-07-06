# Language-Instructed Detection Assistant

Оригинальные статьи:
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485)
- [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872)
- [LISA: Reasoning Segmentation via Large Language Model](https://arxiv.org/pdf/2308.00692)

| Статья | Что взял оттуда 
|--------|-----------------|
| [LLaVA: Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485) (Liu et al., 2023) | CLIP + LLM + обучаемый MLP-проектор, `<img>` токен, instruction tuning на caption-данных |
| [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872) (Carion et al., 2020) | L1 + GIoU loss для bbox, идея «предсказываем box напрямую из hidden state» |
| [LISA: Reasoning Segmentation via LLM](https://arxiv.org/pdf/2308.00692) (Lai et al., 2023) | Спецтокен (`<loc>` вместо `<SEG>`), embedding-as-output, LoRA на LLM | 

---

**Задача:** Собрать мультимодальную модель с нуля, которая умеет **описывать изображения** и **находить объекты по текстовому запросу** (grounding через bbox). Без готовых LLaVA-обёрток — свой датасет, forward pass, training loop и inference.

---

## **1. Архитектура**

```
Картинка → CLIP ViT → patch features (penultimate layer)
                           ↓
                      MLP Projector (+ pixel unshuffle)
                           ↓
Текст → Tokenizer → embeddings → замена <img> на visual tokens → Qwen2.5-0.5B
                           ↓
                 генерация текста / <loc> токен
                           ↓
                 DetectionHead → bbox [x1, y1, x2, y2]
```

**Заморожено:** CLIP ViT (`openai/clip-vit-base-patch16`)

**Учится:**
- Этап 1 — только MLP-проектор
- Этап 2 — MLP + DetectionHead + LoRA на Qwen

**Спецтокены:** `<img>` (вставка картинки), `<loc>` (маркер «здесь bbox»)

---

## **2. Описание решения**

Проект разбит на два этапа — сначала учим «склейку» vision + language, потом добавляем spatial grounding.

### **Этап 1: LLaVA adapter** (`LLaVA_adapter_train.ipynb`)

- Берём mini COCO 2014 captions
- Строим промпт: `<img>` + `Describe this image.` → ответ ассистента
- Лосс считается **только** по токенам ответа (`labels[:len(prompt)] = -100`)
- MLP с `pixel_unshuffle` — сжимаем число visual tokens в 4 раза перед проекцией

### **Этап 2: LIZA grounding** (`LIZA_grounding_train.ipynb`)

Название LIZA — моя версия LISA: та же идея «спецтокен → hidden state → spatial output», но вместо segmentation mask пока bbox.

Из COCO 2017 собираем 4 типа сэмплов:

| task_type | Что делает модель |
|-----------|-------------------|
| `caption` | Описывает картинку |
| `grounding` | «Locate the {object}» → `<loc>` + bbox |
| `joint` | Caption, потом grounding в одном диалоге |
| `locate_and_describe` | Описание + локализация в одном ответе |

- Модель генерирует `<loc>`, эмбеддинг этого токена идёт в `DetectionHead` → bbox
- Detection loss из DETR: **L1 + GIoU** (не просто MSE)
- LoRA (r=16) на attention/MLP слоях Qwen + обучение `embed_tokens` / `lm_head` под новые токены

```python
detection_loss = 2.0 * L1 + 5.0 * GIoU
total_loss = text_loss + detection_loss
```

### **Структура репозитория**

```
Mini-LLaVA-from-scratch/
├── LLaVA_adapter_train.ipynb    # Этап 1: MLP на COCO captions
├── LLaVA_inference.ipynb        # Captioning + сравнение с baseline
├── LIZA_grounding_train.ipynb   # Этап 2: grounding + LoRA + detection head
├── LIZA_inference.ipynb         # Describe / locate / joint
└── models/model0/checkpoints/   # Веса (adapter, head, LoRA, loss log)
```

---

## **3. Эксперименты**

### **Этап 1 — LLaVA adapter**

**Гиперпараметры**
- **Vision encoder:** CLIP ViT-B/16
- **LLM:** Qwen2.5-0.5B-Instruct (frozen)
- **Dataset:** mini COCO 2014 captions
- **Epochs:** 6
- **Batch size:** 16 × grad accum 2
- **Learning rate:** 1e-3 (AdamW, weight_decay=0.1)
- **Precision:** fp16

### **Этап 2 — LIZA grounding**

**Гиперпараметры**
- **Dataset:** COCO 2017 (captions + instances)
- **Epochs:** 1
- **Batch size:** 8 × grad accum 2
- **Learning rate:** 1e-3 (MLP + head), 1e-4 (LoRA)
- **LoRA:** r=16, alpha=32, dropout=0.05
- **Detection loss:** L1 (×2.0) + GIoU (×5.0)

---

## **4. Результаты**

**Качественные примеры**

- **Успехи:**
  - После обучения MLP captioning заметно лучше, чем с untrained projector
  - Модель генерирует `<loc>` и выдаёт bbox — pipeline работает end-to-end
  - Joint-задача (описать + найти объект) тоже отрабатывает

  - Captioning
    - <img alt="caption example" src="image.png" />

  - Grounding — Locate the train
    - <img alt="grounding example" src="image-7.png" />

  - Joint — Describe + locate
    - <img  alt="joint task" src="image-6.png" />

  - Другие примеры
    - <img alt="example 2" src="image-2.png" />
    - <img alt="example 3" src="image-3.png" />
    - <img alt="example 4" src="image-4.png" />

**Кривая обучения**

- <img alt="training loss" src="models/model0/checkpoints/training_loss.png" />

Средний loss за 1 эпоху: **6.47** (`models/model0/checkpoints/loss_log.txt`)

---

## **5. Выводы**

**Что получилось:**
✅ End-to-end пайплайн: датасет → обучение → инференс, всё на чистом PyTorch  
✅ Корректное маскирование лосса и вставка visual tokens в LLM  
✅ Captioning после adapter train — качественно лучше baseline без обучения  
✅ Grounding через `<loc>` + DetectionHead — идея из LISA/DETR работает  
✅ Multi-task обучение (caption + grounding + joint) в одном цикле  

**Что не получилось:**
❌ Качество bbox пока слабое — нужно больше эпох и, возможно, другие веса loss  
❌ Нет reasoning segmentation как в оригинальной LISA  
❌ Нет метрик на val и сравнения с SOTA grounding-моделями  

**Почему?**
- Мало эпох grounding-обучения (1 vs планируемых 6+)
- Маленькая LLM (0.5B) — для сложных запросов не хватает capacity
- Упрощённый detection head (MLP вместо полного DETR decoder)
- Не пробовал ReasonSeg и сложные implicit queries из LISA
<!-- 
**Что дальше:**
- Добавить метрики (IoU@mAP, CIDEr)
- Больше эпох + подбор весов L1/GIoU
- Gradio-демо для живого инференса
- Попробовать segmentation head вместо bbox (ближе к оригинальной LISA) -->
