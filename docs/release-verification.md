# GuitarTrace: проверка исходного релиза

Среда: Windows 11, CPython 3.12.16, текущий `uv.lock`. Проверка 2026-11-12. Новые зависимости и модели не устанавливались, ML не запускался.

- `uv run --no-sync pytest`: **175 passed, 2 skipped**. Пропущены только два теста сохранённых реальных SAM-полигонов из отсутствующего локального `out/e2e-lp-sg/`; это не падения и не результат проверки этих кэшей.
- `node --test tests/editor_project.test.mjs tests/editor_erase.test.mjs tests/editor_bend.test.mjs tests/editor_nodes.test.mjs`: **67 passed**.
- Chrome headless + `tests/editor_auto_fixture_server.py` в синтетическом режиме; `tests/editor_browser.mjs ... http://127.0.0.1:8770/ --auto-test --synthetic`: прошли семь наборов manual/export, auto-selection adapter, transforms, finish/eraser, Bend, project save/reopen, nodes/rotation/width. SAM и BiRefNet запрещены fixture-подменами; auto-selection проверена с синтетической маской, не с моделью. Сервер после проверки остановлен.
- До удаления старого fitter зафиксированы SVG синтетического Trace во всех трёх стилях; их хеши после удаления совпадают, regression-тесты добавлены в `tests/test_contour.py`.
- `scripts/check_release_geometry.py before`, затем `after`: SVG силуэта и растровые наложения трёх существующих прозрачных фото (`fender_1.png`, `fender_2.webp`, `lp_ref_1.webp`) побайтово совпадают. Это проверка геометрии экспортера на альфа-силуэтах, **не** повторный прогон полного CV. Визуально просмотрены наложения Fender/LP и синтетический редактор. Локальные материалы: `out/release-geometry/after.html`, SVG + PNG по подпапкам и `out/editor-*`; в git/релиз не включены.
- `uv lock --check --offline`: успешно, зависимости и lock не менялись.
- `uv build --offline --out-dir out/release-check`: wheel и sdist собираются; SPDX `License-Expression: Apache-2.0`, README и URL GuitarTrace корректны. LICENSE, THIRD_PARTY_NOTICES, licensing и sam2-provenance присутствуют в обоих архивах. Проверены отсутствие reference, фото, весов, результатов и старой реализации spline_samples.
- LICENSE побайтово совпадает с `https://www.apache.org/licenses/LICENSE-2.0.txt`; copyright-реквизиты не выдумывались.

Перед локальным коммитом проверяются `git diff --check`, состав индекса и характерные шаблоны секретов. Такой скан не является абсолютной гарантией отсутствия любых секретов. Проверка Linux оставлена существующему CI; полного ML/e2e на новых фото, безопасности публичного сервера и compliance бинарного dependency bundle этот этап не заявляет.

Область лицензии и сторонние ограничения — `licensing.md`, `../THIRD_PARTY_NOTICES.md`; дальнейший перенос отдельного дерева без старой истории — `github-release.md`.
