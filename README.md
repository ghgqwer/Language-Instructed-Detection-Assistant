# Mini-LLaVA → LIZA

Я собрал в одном репозитории три идеи из разных статей и попробовал склеить их в одну модель, которая умеет **описывать картинки** и **находить объекты по текстовому запросу**. Без готовых LLaVA-обёрток от HuggingFace — всё на чистом PyTorch, в Jupyter-ноутбуках.

Название «LIZA» — моя версия LISA: та же логика «специальный токен → эмбеддинг → spatial output», но вместо сегментационных масок пока bbox.

---


## Используемые статьи

| Статья | Что взял оттуда 
|--------|-----------------|
| [LLaVA: Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485) (Liu et al., 2023) | CLIP + LLM + обучаемый MLP-проектор, `<img>` токен, instruction tuning на caption-данных |
| [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872) (Carion et al., 2020) | L1 + GIoU loss для bbox, идея «предсказываем box напрямую из hidden state» |
| [LISA: Reasoning Segmentation via LLM](https://arxiv.org/pdf/2308.00692) (Lai et al., 2023) | Спецтокен (`<loc>` вместо `<SEG>`), embedding-as-output, LoRA на LLM | 

---

## Зачем это вообще

Мне хотелось не просто вызвать `pipeline("image-to-text")`, а пройти весь путь руками:

- как собрать мультимодальный батч с `<img>` токеном;
- как правильно замаскировать лосс, чтобы модель училась только на ответе ассистента;
- как вставить визуальные эмбеддинги внутрь LLM вместо одного placeholder-токена;
- как добавить вторую «голову» — не текст, а координаты.

В итоге получился двухэтапный пайплайн: сначала учим «склейку» vision + language (LLaVA), потом докручиваем grounding (LIZA).

---

## Как это устроено

```
Картинка ──► CLIP ViT ──► patch features (penultimate layer)
                              │
                              ▼
                         MLP Projector (+ pixel unshuffle)
                              │
Текст ──► Tokenizer ──► embeddings ──► замена <img> на visual tokens ──► Qwen2.5-0.5B
                              │
                              ▼
                    генерация текста / <loc> токен
                              │
                              ▼
                    DetectionHead ──► bbox [x1, y1, x2, y2]
```

**Заморожено:** CLIP ViT (`openai/clip-vit-base-patch16`)

**Учится:**
- на этапе 1 — только MLP-проектор;
- на этапе 2 — MLP + DetectionHead + LoRA-адаптеры на Qwen.

**Спецтокены:** `<img>` (вставка картинки), `<loc>` (маркер «здесь должен быть bbox»)

---

## Структура репозитория

```
Mini-LLaVA-from-scratch/
├── LLaVA_adapter_train.ipynb    # Этап 1: обучение MLP на COCO captions
├── LLaVA_inference.ipynb        # Инференс captioning + сравнение с baseline
├── LIZA_grounding_train.ipynb   # Этап 2: grounding + LoRA + detection head
├── LIZA_inference.ipynb         # Инференс: describe / locate / joint
├── models/model0/checkpoints/    # Сохранённые веса
│   ├── adapter_epoch_*.pth
│   ├── head_module_epoch_*.pth
│   ├── llm_module_epoch_*/       # LoRA weights (PEFT)
│   ├── loss_log.txt
│   └── training_loss.png
└── samples/                      # сюда кладём скрины примеров (см. ниже)
```

---

## Этап 1: LLaVA adapter

**Ноутбук:** `LLaVA_adapter_train.ipynb`

Берём mini COCO 2014 (captions), строим простой датасет:

```
<|im_start|>user
<img>
Describe this image.
<|im_start|>assistant
{caption}
```

Лосс считается **только** по токенам ответа ассистента — всё до этого замаскировано `-100`.

MLP-проектор с `pixel_unshuffle`: сжимаем число визуальных токенов в 4 раза перед проекцией в text space. Это уменьшает длину последовательности и экономит память.

| Параметр | Значение |
|----------|----------|
| Vision | CLIP ViT-B/16 |
| LLM | Qwen2.5-0.5B-Instruct (frozen) |
| Dataset | mini COCO 2014 captions |
| Epochs | 6 |
| Batch size | 16 × grad accum 2 |
| LR | 1e-3 (AdamW) |
| Precision | fp16 |

Чекпоинты: `checkpoints/llava_mlp_epoch_{k}.pth`

---

## Этап 2: LIZA grounding

**Ноутбук:** `LIZA_grounding_train.ipynb`

Здесь датасет сложнее — из COCO 2017 собираются 4 типа сэмплов:

| task_type | Что делает модель |
|-----------|-------------------|
| `caption` | Описывает картинку |
| `grounding` | «Locate the {object}» → `<loc>` + bbox |
| `joint` | Два turn'а подряд: caption, потом grounding |
| `locate_and_describe` | Описание + локализация в одном ответе |

Bbox нормализуются в `[0, 1]`. Для сэмплов без bbox ставится `[-1, -1, -1, -1]` и detection loss не считается.

Detection loss — не просто MSE, а **L1 + GIoU** (из DETR):

```python
detection_loss = 2.0 * L1 + 5.0 * GIoU
total_loss = text_loss + detection_loss
```

LoRA (r=16, alpha=32) вешается на attention и MLP слои Qwen. Отдельно учатся `embed_tokens` и `lm_head`, потому что добавили новые токены.

| Параметр | Значение |
|----------|----------|
| Dataset | COCO 2017 (captions + instances) |
| Epochs | 1 (пока; планирую больше) |
| Batch size | 8 × grad accum 2 |
| LR | 1e-3 (MLP + head), 1e-4 (LoRA) |
| Detection loss | L1 + GIoU |

---

## Запуск

### Зависимости

```bash
pip install torch torchvision transformers peft pillow tqdm matplotlib
```

Нужна GPU. На CPU можно только инференс маленьких моделей, обучение — боль.

### Обучение

1. Положить датасет (пути в ноутбуках заточены под Kaggle, поменять на свои):
   - mini COCO 2014 → для adapter train
   - COCO 2017 → для grounding train
2. Запустить `LLaVA_adapter_train.ipynb` → получить MLP checkpoint
3. Запустить `LIZA_grounding_train.ipynb` → получить adapter + head + LoRA

### Инференс

```python
# LLaVA — captioning
# см. LLaVA_inference.ipynb

# LIZA — caption + grounding
model = LIZA(
    vision_model="openai/clip-vit-base-patch16",
    llm_name="Qwen/Qwen2.5-0.5B-Instruct",
    lora_checkpoint="models/model0/checkpoints/llm_module_epoch_1",
    mlp_checkpoint="models/model0/checkpoints/adapter_epoch_1.pth",
    detection_head_checkpoint="models/model0/checkpoints/head_module_epoch_1.pth",
)

res = model.generate(image, "Locate the train.")
# res["text"] — ответ модели
# res["bbox"]  — [x0, y0, x1, y1] в пикселях (или None)
```

---

## Примеры работы
![alt text](image.png)
![alt text](image-2.png)

![alt text](image-3.png)
![alt text](image-4.png)
### Captioning (LLaVA)

**Запрос:** `Describe this image.`

<!-- TODO: добавить скрин -->
![Caption example](samples/caption_example.png)
*Placeholder — сюда скрин из LLaVA_inference.ipynb*

| | До обучения MLP | После обучения | Qwen2-VL-2B (baseline) |
|---|---|---|---|
| Ответ | *(вставить)* | *(вставить)* | *(вставить)* |

---

### Grounding (LIZA)

**Запрос:** `Locate the train.`

<!-- TODO: добавить скрин -->
![Grounding example](image-7.png)

**Запрос:** `Describe this image and locate the train.`

![Joint task example](image-6.png)
*Placeholder — caption + bbox на одном изображении*

---

### Сравнение до / после grounding-обучения

![Before/after grounding](samples/before_after_grounding.png)
*Placeholder — один и тот же промпт, два результата*

---

### Кривая обучения

![Training loss](models/model0/checkpoints/training_loss.png)

---

## Что получилось
- end-to-end пайплайн от датасета до инференса;
- корректное маскирование лосса и вставка visual tokens;
- captioning после обучения adapter'а (качественно лучше untrained MLP);
- grounding pipeline: модель генерирует `<loc>`, head выдаёт bbox;
- LoRA + multi-task обучение (caption + grounding + joint).
---

## Стек

- Python 3.10+
- PyTorch
- HuggingFace Transformers + PEFT (LoRA)
- CLIP ViT, Qwen2.5-0.5B-Instruct
- COCO 2014 / 2017
- Jupyter Notebooks (основной формат проекта)

---

## Ссылки

- [LLaVA — Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485)
- [DETR — End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872)
- [LISA — Reasoning Segmentation via Large Language Model](https://arxiv.org/pdf/2308.00692)

---