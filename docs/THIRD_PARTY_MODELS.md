# Реестр сторонних моделей

Ни один сторонний вес не добавляется автоматически. Код и конкретный checkpoint могут иметь разные лицензии; проверяются оба.

| Компонент | Кандидат | Назначение | Статус |
|---|---|---|---|
| Face landmarks | [MediaPipe Face Landmarker](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarker) | геометрия текущего лица и маски зон | основной кандидат; проверить условия model asset |
| Vision runtime | [MediaPipe](https://github.com/google-ai-edge/mediapipe) | локальный inference | Apache-2.0 для кода |
| Mask refinement | [SAM 2](https://github.com/facebookresearch/sam2) | сложные границы по prompts | опционально; Apache-2.0 repo, проверить checkpoint |
| Global enhancement | [Image-Adaptive 3D LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) | архитектурный baseline | Apache-2.0; свой train с нуля |
| Bilateral transform | [HDRNet](https://groups.csail.mit.edu/graphics/hdrnet/) | архитектурный ориентир | paper/reference; условия реализации проверять |
| Local residual | [NAFNet](https://github.com/megvii-research/NAFNet) | локальная ретушь на face crops | MIT; архитектура, обучение с нуля |
| Interpretable filters | [DeepLPF](https://github.com/sjmoran/deeplpf-image-enhancement) | альтернативная локальная экспозиция/градиенты | BSD-3-Clause; чужие веса не нужны |

## Не использовать без отдельного аудита

- случайные Hugging Face checkpoints без явной лицензии;
- face parsing weights, обученные на датасете с research-only условиями;
- face recognition/identity embedding модели;
- модели, которые отправляют кадры в облако;
- generative checkpoints с неясным происхождением обучающих данных;
- веса, для которых лицензия кода ошибочно принимается за лицензию модели.

Отдельный известный риск: код [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) имеет MIT-лицензию, но распространённые веса обучены на [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), разрешённом только для некоммерческих исследований. Такие checkpoint не входят в продукт. Если точного mesh окажется недостаточно, парсер обучается с нуля на собственных исправленных масках после проверки прав на данные.


## Правило принятия

Для каждого компонента фиксируются: URL и commit/tag, лицензия кода, лицензия весов, лицензия исходного датасета, разрешённое применение, SHA-256 загруженного файла и дата проверки. Если хотя бы один пункт неизвестен, компонент остаётся экспериментальным и не входит в рабочую сборку.
