# Third-party notices — GuitarTrace

Инвентаризация 2026-11-12: `uv.lock` и лицензионные файлы установленного окружения Windows / CPython 3.12. Здесь перечислены зависимости, **не включённые в исходники приложения**: uv устанавливает их отдельно. Их лицензии, copyright и NOTICE сохраняют силу; Apache-2.0 приложения их не заменяет. Для других ОС и обновлённого lock состав бинарных компонентов нужно перепроверить.

## SAM 2 и BiRefNet

| Компонент | Происхождение / уведомление | Лицензия |
|---|---|---|
| SAM 2 / SAM 2.1 | [Meta official repository](https://github.com/facebookresearch/sam2); Copyright (c) Meta Platforms, Inc. and affiliates. | Apache-2.0; сохранять upstream LICENSE и применимые уведомления |
| `sam2==1.1.0` | [PyPI](https://pypi.org/project/sam2/1.1.0/), сторонняя перепаковка Jorge Insua / JinsuaFeito-dev, не официальная публикация Meta. [Подробное сравнение](docs/sam2-provenance.md). | Apache-2.0; перепаковка не заменяет правообладателей upstream |
| cc_torch / Connected_components_PyTorch | Опциональный CUDA connected-components код в SAM 2; Copyright (c) 2020, respective contributors, as shown by AUTHORS. [Upstream license](https://github.com/facebookresearch/sam2/blob/main/LICENSE_cctorch) | BSD-3-Clause; в установленном SAM: `sam2-*.dist-info/licenses/LICENSE_cctorch` |
| BiRefNet | [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet); Copyright (c) 2024 ZhengPeng | [MIT](https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE) |
| rembg (источник ONNX-экспорта, не Python-зависимость приложения) | [danielgatis/rembg](https://github.com/danielgatis/rembg); Copyright (c) 2020 Daniel Gatis | [MIT](https://github.com/danielgatis/rembg/blob/main/LICENSE.txt) |

Демо Meta, его фронтенд, шрифты Inter/Noto Color Emoji и SA-V датасет не копировались в приложение. Лицензии OFL и условия SA-V не являются лицензией нашего редактора или фотографий пользователя.

## OpenCV, FFmpeg и научные бинарные библиотеки

- **opencv-python-headless 5.0.0.93 / OpenCV 5**: Apache-2.0 в проверенном wheel. Авторитетные тексты конкретной сборки — `cv2/LICENSE.txt` и `cv2/LICENSE-3RD-PARTY.txt`; [upstream packaging](https://github.com/opencv/opencv-python).
- **FFmpeg** внутри OpenCV wheels: проверенный `LICENSE-3RD-PARTY.txt` прямо указывает распространение FFmpeg и **LGPL-2.1**. Headless означает отсутствие GUI, не отсутствие FFmpeg. То, что приложение работает с фото и не вызывает видео, не удаляет библиотеку и её условия из установленного wheel. [FFmpeg legal](https://ffmpeg.org/legal.html).
- Для macOS тот же файл перечисляет дополнительные библиотеки (libbluray, libgnutls, libnettle, libhogweed, libintl, libmp3lame, libp11, librtmp, libsoxr, libtasn1). Нельзя механически переносить состав Windows на другие платформы.
- **NumPy 2.5.4**: основной код BSD-3-Clause; метаданные проверенного wheel: `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`. Сохранять весь `numpy-*.dist-info/licenses/`, включая уведомления pocketfft, random generators, libdivide, Highway, SIMD и LAPACK при их наличии. [NumPy](https://github.com/numpy/numpy).
- **SciPy 1.18.1**: основной код BSD-3-Clause; в проверенном `scipy-*.dist-info/LICENSE.txt` отдельно перечислены **OpenBLAS (BSD-3-Clause), LAPACK (BSD), GCC runtime (GPL-3.0-or-later WITH GCC-exception-3.1)** и полные тексты. Есть также отдельные уведомления Qhull, uarray, DUCC FFT и интеграторов. GCC Runtime Library Exception существенен: это не обычная GPL на приложение и не основание объявлять весь бинарный стек исключительно BSD. [SciPy](https://github.com/scipy/scipy).
- **ONNX Runtime 1.31.1**: MIT, Copyright Microsoft Corporation; сохранять `onnxruntime/LICENSE` **и** `onnxruntime/ThirdPartyNotices.txt` — последний покрывает включённые сторонние компоненты. [ONNX Runtime](https://github.com/microsoft/onnxruntime).
- **PyTorch 2.15.0+cpu**: основной проект BSD-style; проверенный wheel декларирует `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib`. Сохранять всё дерево `torch-*.dist-info/licenses/`, а не только один BSD-текст. **TorchVision 0.29.2**: BSD-3-Clause и отдельные уведомления бинарной сборки. [PyTorch](https://github.com/pytorch/pytorch), [TorchVision](https://github.com/pytorch/vision).
- **Pillow 13.0.0**: MIT-CMU для основного кода, дополнительные условия включённых кодеков в `pillow-*.dist-info/licenses/LICENSE`; наличие JPEG/PNG/WebP/AVIF означает отдельные компоненты сборки. [Pillow](https://github.com/python-pillow/Pillow).

## tqdm — не только MIT

**tqdm 4.70.1**, [upstream](https://github.com/tqdm/tqdm), получен транзитивно через SAM 2: `MPL-2.0 AND MIT`, а не выбор «MIT или MPL» для всего кода. Установленный `tqdm-*.dist-info/licenses/LICENCE` указывает MPL-2.0 © 2015–2026 Casper da Costa-Luis, MIT © 2013 Noam Yorav-Raphael и вклад 2016 от Google Inc.

При распространении tqdm сохранять LICENCE и уведомления; исходная форма MPL-файлов должна оставаться доступной по MPL-2.0, а получатели исполняемой формы должны знать, где её получить ([MPL-2.0](https://www.mozilla.org/MPL/2.0/)). Отдельное использование библиотеки само по себе не требует лицензировать собственный код приложения под MPL. При изменении/встраивании MPL-файлов обязанности проверяются отдельно.

## Остальные пакеты текущего lock

Версии ниже — установленные пакеты текущего lock, включая dev-группу. Ссылки ведут к точному выпуску, его метаданным и исходным артефактам; полные copyright-имена и условия сохраняются в поставляемых самим пакетом LICENSE/LICENCE/COPYING/NOTICE, а не заменяются этой таблицей.

| Пакеты | Основная лицензия |
|---|---|
| [FastAPI 0.162.0](https://pypi.org/project/fastapi/0.162.0/), [Pydantic 2.14.1](https://pypi.org/project/pydantic/2.14.1/), [pydantic-core 2.50.1](https://pypi.org/project/pydantic-core/2.50.1/) | MIT |
| [Starlette 1.10.0](https://pypi.org/project/starlette/1.10.0/), [Uvicorn 0.54.0](https://pypi.org/project/uvicorn/0.54.0/) | BSD-3-Clause |
| [python-multipart 0.0.32](https://pypi.org/project/python-multipart/0.0.32/), [opentelemetry-api 1.45.1](https://pypi.org/project/opentelemetry-api/1.45.1/), [FlatBuffers 25.12.19](https://pypi.org/project/flatbuffers/25.12.19/) | Apache-2.0 |
| [Hydra Core 1.3.8](https://pypi.org/project/hydra-core/1.3.8/), [iopath 0.1.10](https://pypi.org/project/iopath/0.1.10/), [PyYAML 6.0.3](https://pypi.org/project/PyYAML/6.0.3/) | MIT |
| [OmegaConf 2.3.1](https://pypi.org/project/omegaconf/2.3.1/), [ANTLR runtime 4.9.3](https://pypi.org/project/antlr4-python3-runtime/4.9.3/) | BSD-3-Clause |
| [AnyIO 4.15.1](https://pypi.org/project/anyio/4.15.1/), [h11 0.16.0](https://pypi.org/project/h11/0.16.0/), [filelock 4.2.4](https://pypi.org/project/filelock/4.2.4/) | MIT |
| [Click 8.5.0](https://pypi.org/project/click/8.5.0/), [fsspec 2026.9.0](https://pypi.org/project/fsspec/2026.9.0/), [idna 3.20](https://pypi.org/project/idna/3.20/), [portalocker 4.4.2](https://pypi.org/project/portalocker/4.4.2/) | BSD-3-Clause |
| [Jinja2 3.1.6](https://pypi.org/project/Jinja2/3.1.6/), [MarkupSafe 3.0.4](https://pypi.org/project/MarkupSafe/3.0.4/), [NetworkX 3.7](https://pypi.org/project/networkx/3.7/), [mpmath 1.3.0](https://pypi.org/project/mpmath/1.3.0/), [SymPy 1.14.0](https://pypi.org/project/sympy/1.14.0/), [protobuf 7.36.2](https://pypi.org/project/protobuf/7.36.2/) | BSD-3-Clause; сохранять встроенные notices |
| [annotated-doc 0.0.5](https://pypi.org/project/annotated-doc/0.0.5/), [annotated-types 0.8.0](https://pypi.org/project/annotated-types/0.8.0/), [typing-inspection 0.4.4](https://pypi.org/project/typing-inspection/0.4.4/), [Colorama 0.4.6](https://pypi.org/project/colorama/0.4.6/) | MIT |
| [typing-extensions 4.16.0](https://pypi.org/project/typing-extensions/4.16.0/) | PSF-2.0 |
| [packaging 26.3](https://pypi.org/project/packaging/26.3/) | Apache-2.0 OR BSD-2-Clause |
| [pytest 9.2.0](https://pypi.org/project/pytest/9.2.0/), [pluggy 1.7.0](https://pypi.org/project/pluggy/1.7.0/), [iniconfig 2.3.1](https://pypi.org/project/iniconfig/2.3.1/), [setuptools 84.2.1](https://pypi.org/project/setuptools/84.2.1/) | MIT; setuptools содержит vendor-компоненты с отдельными уведомлениями |
| [Pygments 2.21.0](https://pypi.org/project/Pygments/2.21.0/) | BSD-2-Clause |

Build backend Hatchling (MIT) и его зависимости устанавливаются инструментом сборки отдельно и не фиксируются runtime-lock как часть приложения; проверять конкретную сборочную среду при распространении инструментов сборки. uv (MIT/Apache-2.0), Python (PSF и лицензии включённых компонентов), Node для тестов (MIT и сторонние notices) не вендорятся. GitHub Actions `actions/checkout`, `actions/setup-node` (MIT), `astral-sh/setup-uv` (MIT) — внешние CI-инструменты; сохранять их лицензии при копировании самих инструментов.

## Модели — отдельные загрузки

`scripts/fetch_models.py` скачивает только по запросу пользователя:

| Файл | Источник и условия | SHA-256 |
|---|---|---|
| `sam2.1_hiera_tiny.pt` | [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny); model card — Apache-2.0; официальные checkpoints SAM 2 лицензированы отдельно от приложения | `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69` |
| `birefnet-lite.onnx` | [rembg release asset](https://github.com/danielgatis/rembg/releases/tag/v0.0.0), имя upstream `BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx`; BiRefNet и rembg публикуют MIT. Отдельной цепочки разрешений именно на ONNX-артефакт этот аудит не устанавливает; MIT репозитория не является автоматическим доказательством прав на все веса и обучающие данные | `5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333` |

Для исходного релиза веса не перепубликуются; ссылки и хеши не заменяют условия правообладателей. Для будущего зеркала/упаковки весов отдельно проверить происхождение ONNX и исходных pretrained weights, сохранить лицензии и model cards. Обучающие датасеты не входят в релиз, права на них не выводятся из лицензии модели.

## Фото и результаты

Фотографии из `samples/`, модели из `models/`, обводки/наложения из `out/` и пользовательские проекты не распространяются вместе с кодом. `eval/samples.csv` — перечень источников, не передача прав. CC BY/CC BY-SA, public-domain утверждения и коммерческие каталожные фото проверяются по каждому источнику; для BY нужны автор и атрибуция, для BY-SA возможны условия к производным работам. Экспорт в SVG/PNG не гарантирует освобождения от прав на исходное фото, дизайн или товарные знаки.

## Если распространяете не только исходники

Этот список достаточен как инвентаризация исходного приложения без включённых библиотек; это **не готовый compliance-пакет для EXE, Docker-образа или набора wheels**. При включении библиотек необходимо сохранить полные лицензии/copyright/NOTICE конкретных артефактов, выполнить MPL source availability и LGPL (включая применимые требования к исходникам, модификациям и замене/перелинковке), сохранить GCC/LLVM exceptions и уведомления кодеков. Ссылки в Markdown сами по себе этих обязанностей не выполняют. Не удаляйте файлы лицензий из `.dist-info`/библиотек.
