# GuitarTrace: перенос проверенного снимка в независимый репозиторий

Цель: `https://github.com/Tw1nk1e/GuitarTrace`. Следующие команды — **план**, они не выполняются при подготовке. Один локальный release-коммит остаётся в исходном GitLab-клоне; его родителей, refs, tags и `.git` нельзя переносить на GitHub.

## 1. Из исходного каталога: зафиксировать выбранный снимок

PowerShell; `$release` должен совпасть с полным хешем проверенного release-коммита из отчёта подготовки, не с произвольным будущим HEAD:

```powershell
$release = git rev-parse HEAD
git show --no-patch --format=fuller $release
git status --short
git ls-tree -r --name-only $release
git remote -v
```

Остановиться, если статус не чистый или состав не совпадает с проверенным. LICENSE, THIRD_PARTY_NOTICES и документы лицензирования должны быть закоммичены. Ни `reference/`, ни models/samples/out/dist/build/.venv, секретов и медиа быть не должно. Исходный `origin` остаётся GitLab.

## 2. Экспорт без Git-истории

```powershell
$target = Join-Path (Split-Path -Parent (Get-Location).Path) 'GuitarTrace'
if (Test-Path $target) { throw 'Каталог назначения уже существует; не перезаписывать' }
New-Item -ItemType Directory -Force out | Out-Null
$archive = Join-Path (Get-Location).Path 'out/GuitarTrace-source.zip'
if (Test-Path $archive) { throw 'Архив уже существует; выберите новое имя' }
git archive --format=zip --output="$archive" $release
if ($LASTEXITCODE -ne 0) { throw 'git archive failed' }
Expand-Archive -LiteralPath $archive -DestinationPath $target
Get-ChildItem -Force $target
if (Test-Path (Join-Path $target '.git')) { throw 'Старые Git-метаданные не должны попасть в снимок' }
```

`git archive` экспортирует только дерево выбранного коммита, включая dotfiles, без родителей/remote/reflog. Не заменять его копированием всей рабочей папки, `git clone`, `--mirror` или push старой ветки. Повторно просмотреть распакованный состав; при желании выполнить frozen-install и тесты из новой папки (веса для unit-тестов не нужны).

## 3. Только после отдельного разрешения: новый локальный Git

```powershell
git -C "$target" init -b main
git -C "$target" add --all
git -C "$target" diff --cached --check
git -C "$target" diff --cached --stat
git -C "$target" commit -m "Initial public source release of GuitarTrace"
git -C "$target" remote add origin https://github.com/Tw1nk1e/GuitarTrace.git
git -C "$target" rev-list --count HEAD
git -C "$target" status --short
git -C "$target" remote -v
```

Убедиться: ровно один коммит, чистый статус, новый origin указывает на GitHub только в новом каталоге. Эта операция не меняет origin исходного клона. Push запрещён до отдельного разрешения; перед ним повторно проверить, что GitHub-репозиторий действительно пустой. Не применять force push, не переносить старые теги и ветки. Чистая история не отменяет лицензии или атрибуцию текущего дерева.
