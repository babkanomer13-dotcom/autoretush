# Карта проекта

## Две физически разные зоны

```text
PUBLIC: GitHub/autoretush/              PRIVATE: E:/autoretush_local/
├── README.md                           ├── inventory/
├── configs/                            ├── manifests/
│   └── local.example.yaml              ├── reviews/
├── docs/                               ├── dataset/
├── src/autoretush/                     ├── weights/
├── tests/                              ├── runs/
└── .gitignore                          └── outputs/
```

Публичная зона содержит документацию, безопасные интерфейсы и общую механику. Приватная зона содержит всё, что связано с фотографиями, расположением архива, выбранными парами, параметрами собственного стиля, обучением и результатами.

## Предлагаемая структура кода

```text
src/autoretush/
├── cli.py                 # единая командная строка
├── config.py              # проверка локального YAML
├── inventory.py           # read-only поиск папок pp и статистика
├── pairing/
│   ├── fingerprint.py     # отпечаток всего кадра
│   ├── geometry.py        # SIFT, RANSAC, ECC
│   ├── assignment.py      # one-to-one пары
│   └── manifest.py        # приватный JSONL
├── review/                # локальная очередь подтверждения
├── alignment/             # warp, валидная область, QC
├── regions/               # landmarks и мягкие маски зон
├── models/                # публичные интерфейсы; ядро приватно
├── training/              # публичный runner; рецепт приватно
├── inference/             # tiling, compositor, export
└── quality/               # метрики и стоп-сигналы
```

## Поток данных

| Артефакт | Содержит персональные данные | Где хранится | Git |
|---|---:|---|---:|
| Пример конфига | нет | публичный repo | да |
| Реальный конфиг с путями | да | локально | нет |
| Инвентаризация | да | `E:/autoretush_local/inventory` | нет |
| Пары/превью | да | `E:/autoretush_local/reviews` | нет |
| Датасет | да | `E:/autoretush_local/dataset` | нет |
| Веса/чекпойнты | собственность проекта | `E:/autoretush_local/weights` | нет |
| Исходный архив | да | существующие сезоны, read-only | нет |
| Документация архитектуры | нет | публичный repo | да |

## Компоненты будущего приложения

1. `Dataset Builder` — инвентаризация, подбор, выравнивание, ручная очередь.
2. `Trainer` — обучение global/local веток, возобновление, метрики.
3. `Retouch Engine` — пакетная обработка на GPU.
4. `Review Desk` — просмотр до/после, ползунок силы, принятие/повтор.
5. `Exporter` — формат, ICC, размер, структура выходных папок.
