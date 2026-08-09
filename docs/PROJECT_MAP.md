# Карта проекта

## Две физически разные зоны

```text
PUBLIC: GitHub/autoretush/              PRIVATE: <private-workspace>/
├── README.md                           ├── inventory/
├── configs/                            ├── manifests/
│   └── local.example.yaml              ├── reviews/
├── docs/                               ├── dataset/
├── src/autoretush/                     ├── weights/
├── tests/                              ├── runs/
└── .gitignore                          └── outputs/
```

Публичная зона содержит документацию, безопасные интерфейсы и общую механику. Приватная зона содержит всё, что связано с фотографиями, расположением архива, выбранными парами, параметрами собственного стиля, обучением и результатами.

## Текущая структура кода

```text
src/autoretush/
├── cli.py                 # единая командная строка
├── config.py              # проверка локального YAML
├── inventory.py           # read-only поиск папок pp и статистика
├── review.py              # publishable WebP-пакет без приватного manifest
├── identifiers.py         # строгие псевдонимные ID без traversal
├── paths.py               # границы приватной зоны, архивов и публичного repo
├── alignment.py           # full-resolution warp, valid mask и QC
├── dataset.py             # dry-run, split и атомарное copy-only создание набора
├── pairing/
│   ├── fingerprint.py     # отпечаток всего кадра
│   ├── geometry.py        # SIFT, RANSAC и edge correlation
│   ├── matcher.py         # one-to-one assignment
│   ├── snapshot.py        # content snapshot входов для безопасного resume
│   ├── run.py             # crash-safe shards и resumable orchestration
│   └── manifest.py        # локальный JSONL
└── __init__.py

local/core/                # существует только локально, Git его игнорирует
├── global_lut.py          # приватный image-adaptive 3D-LUT baseline
├── face_zones.py          # 478 landmarks и мягкие маски текущего кадра
├── training.py            # приватный рецепт Global 3D-LUT
├── inference.py           # full-resolution batch engine и QC
├── evaluation.py          # holdout-метрики и агрегаты без путей
└── zone_review.py         # локальный WebP-аудит масок зон
```

## Поток данных

| Артефакт | Содержит персональные данные | Где хранится | Git |
|---|---:|---|---:|
| Пример конфига | нет | публичный repo | да |
| Реальный конфиг с путями | да | локально | нет |
| Инвентаризация | да | `<private-workspace>/inventory` | нет |
| Пары/превью | да | `<private-workspace>/reviews` | нет |
| Датасет | да | `<private-workspace>/dataset` | нет |
| Веса/чекпойнты | собственность проекта | `<private-workspace>/weights` | нет |
| Исходный архив | да | существующие сезоны, read-only | нет |
| Документация архитектуры | нет | публичный repo | да |

## Компоненты будущего приложения

1. `Dataset Builder` — инвентаризация, подбор, выравнивание, ручная очередь.
2. `Trainer` — обучение global/local веток, возобновление, метрики.
3. `Retouch Engine` — пакетная обработка на GPU.
4. `Review Desk` — просмотр до/после, ползунок силы, принятие/повтор.
5. `Exporter` — формат, ICC, размер, структура выходных папок.
