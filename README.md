# Извлечение стен и точек доступа из PDF-плана

Проект извлекает из PDF-плана две группы данных:

- объединенные кривые стен;
- позиции точек доступа AP.

Основной скрипт: `extract_walls_and_aps.py`.
Параметры задаются в `config.yaml`.

## Быстрый запуск

```bash
python extract_walls_and_aps.py --config config.yaml
```

По умолчанию входной файл PDF должен лежать рядом с конфигом:

```yaml
input:
  pdf_path: "trajectory.pdf"
  page_index: 0
```

`page_index` нумеруется с нуля.

## Результаты

Финальные файлы записываются в папку `output/`.

Скрипт пишет только:

- `wall_curves.csv` — извлеченные кривые стен;
- `ap_positions.csv` — найденные точки доступа.

Промежуточные изображения, маски и диагностические CSV пишутся в `processing/`.

## Формат `wall_curves.csv`

Каждая строка описывает одну вершину полилинии стены.

Основные поля:

- `curve_id` — идентификатор кривой стены;
- `vertex_index` — номер вершины внутри кривой;
- `x_pdf_pt`, `y_pdf_pt` — координаты в PDF points;
- `x_px`, `y_px` — координаты в пикселях рендера;
- `curve_length_pdf_pt` — длина кривой в PDF points;
- `curve_length_px` — длина кривой в пикселях.

Координаты PDF считаются от левого верхнего угла страницы.

## Формат `ap_positions.csv`

Каждая строка описывает одну найденную точку доступа.

Основные поля:

- `id` — идентификатор AP;
- `type` — тип/цвет маркера (`orange` или `violet`);
- `x_pdf_pt`, `y_pdf_pt` — координаты центра в PDF points;
- `x_px`, `y_px` — координаты центра в пикселях;
- `bbox_pdf_pt` — bounding box найденного маркера;
- `cluster_items` — число векторных элементов, вошедших в кластер.

## Настройка конфига

Главные секции `config.yaml`:

- `input` — путь к PDF и номер страницы.
- `folders` — папки для итоговых файлов и промежуточной диагностики.
- `output` — имена итоговых CSV.
- `render` — DPI для рендера PDF.
- `plan_roi_pdf_pt` — область плана в координатах PDF points.
- `wall_extraction` — фильтры векторных кандидатов стен.
- `footnote_leader_filter` — удаление коротких выносок/подписей, похожих на стены.
- `wall_union` — объединение фрагментов стен, фильтрация компонент и трассировка скелета.
- `ap_extraction` — поиск AP по цветам и кластеризация.
- `debug` — сохранение отладочных изображений и CSV.

Если включены:

```yaml
clean_output_dir: true
clean_processing_dir: true
```

старые файлы внутри `output/` и `processing/` удаляются перед каждым запуском.

## Диагностика

При включенном `debug.save_processing_images` в `processing/` сохраняются:

- `render_page_1_300dpi.png` — растеризованная страница PDF;
- `raw_wall_candidates_mask.png` — сырая маска кандидатов стен;
- `united_wall_mask.png` — объединенная и очищенная маска стен;
- `wall_skeleton.png` — скелет стен;
- `wall_curves_preview_crop.png` — предпросмотр извлеченных кривых.

Дополнительные диагностические CSV:

- `removed_footnote_leaders.csv` — удаленные выноски;
- `wall_components.csv` — статистика компонент маски стен.

## Зависимости

Скрипт использует:

- PyMuPDF;
- OpenCV;
- NumPy;
- PyYAML;
- scikit-image;
- scikit-learn.

Пример установки:

```bash
pip install pymupdf opencv-python numpy pyyaml scikit-image scikit-learn
```

## Типовой рабочий процесс

1. Положить PDF рядом с `config.yaml` или указать абсолютный путь в `input.pdf_path`.
2. Настроить `plan_roi_pdf_pt`, чтобы исключить рамку листа, штамп и лишние подписи.
3. Запустить:

```bash
python extract_walls_and_aps.py --config config.yaml
```

4. Проверить `output/wall_curves.csv` и `output/ap_positions.csv`.
5. Если геометрия извлечена плохо, смотреть диагностические файлы в `processing/` и корректировать фильтры в `wall_extraction`, `footnote_leader_filter` и `wall_union`.
