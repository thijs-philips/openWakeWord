---
name: setup-python-env
description: >
  Scaffold a complete Python project environment with virtual environment, CLI/lib separation,
  batch scripts, .gitignore, and data directories. Use this skill whenever the user asks to
  set up a new Python project, create a Python environment, scaffold a Python app, initialize
  a Python CLI tool, bootstrap a Python workspace, or wants a ready-to-go Python project structure —
  even if they don't say "scaffold" or "environment" explicitly. Also use it when the user says
  things like "start a new Python project", "set up folders for my Python code", or
  "create a basic Python app layout".
---

# Setup Python Environment

Scaffold a self-contained Python project with a clean separation between CLI front-end and
library back-end, a virtual environment managed via batch scripts, and sensible defaults for
`.gitignore`, `.editorconfig`, and folder structure.

## When to use

- User wants to start a new Python project from scratch
- User asks for a project skeleton, boilerplate, or scaffold
- User needs a CLI tool structure with argument parsing
- User wants a virtual environment with build/run scripts

## What gets created

```
<project-root>/
├── .venv/                    # Virtual environment (created by build.bat)
├── cli/
│   ├── __init__.py
│   └── <project-name>-cli.py  # CLI entry point (argparse stub)
├── lib/
│   └── __init__.py           # Library package for models, algorithms, logic
├── data/
│   ├── in/.gitkeep           # Input data
│   └── out/.gitignore        # Output data (contents git-ignored)
├── build.bat                 # Creates .venv & installs requirements (Windows)
├── build.sh                  # Creates .venv & installs requirements (macOS/Linux)
├── run.bat                   # Shortcut — delegates to the CLI .bat (Windows)
├── run.sh                    # Shortcut — delegates to the CLI .sh (macOS/Linux)
├── <project-name>-cli.bat    # Activates venv if needed, runs CLI (Windows)
├── <project-name>-cli.sh     # Activates venv if needed, runs CLI (macOS/Linux)
├── requirements.txt          # Python dependencies (starts near-empty)
├── .gitignore                # Python-oriented (GitHub default, adapted)
├── .editorconfig             # Consistent formatting across editors
└── README.md                 # Setup/run instructions and folder overview
```

## How to use this skill

### Step 1 — Gather project details

Ask the user for a **project name** (kebab-case, e.g. `image-sorter`). If they've already
mentioned one in conversation, use it. The project name drives file naming:

- CLI script: `cli/<project-name>-cli.py`
- CLI batch: `<project-name>-cli.bat`

Also confirm the **target directory**. Default is the current workspace root.

### Step 2 — Create all files

Use the templates in `assets/` as the starting point. For every template, replace the
placeholder `{{PROJECT_NAME}}` with the actual project name (kebab-case) and
`{{PROJECT_TITLE}}` with a human-friendly title (Title Case, derived from the project name).

Create files in this order:

1. `requirements.txt` — from `assets/requirements.txt.tmpl`
2. `lib/__init__.py` — empty file
3. `cli/__init__.py` — empty file
4. `cli/<project-name>-cli.py` — from `assets/cli.py.tmpl`
5. `<project-name>-cli.bat` — from `assets/cli.bat.tmpl`
6. `<project-name>-cli.sh` — from `assets/cli.sh.tmpl`
7. `build.bat` — from `assets/build.bat.tmpl`
8. `build.sh` — from `assets/build.sh.tmpl`
9. `run.bat` — from `assets/run.bat.tmpl`
10. `run.sh` — from `assets/run.sh.tmpl`
11. `data/in/.gitkeep` — empty file
12. `data/out/.gitignore` — from `assets/data-out-gitignore.tmpl`
13. `.gitignore` — from `assets/gitignore.tmpl`
14. `.editorconfig` — from `assets/editorconfig.tmpl`
15. `README.md` — from `assets/README.md.tmpl`

After creating the `.sh` files, mark them executable:
```bash
chmod +x build.sh run.sh <project-name>-cli.sh
```

### Step 3 — Build the environment

Run `build.bat` (Windows) or `bash build.sh` (macOS/Linux) to create the `.venv` and
install any requirements.

### Step 4 — Verify

Run `run.bat` (Windows) or `bash run.sh` (macOS/Linux) to confirm the CLI executes and
prints its stub message.

## Customization points

- **Extra dependencies**: If the user mentions specific libraries (pandas, requests, etc.),
  add them to `requirements.txt` before running `build.bat`.
- **Windows-only or Unix-only**: Both `.bat` and `.sh` scripts are created by default so
  the project works cross-platform. If the user explicitly only wants one OS, skip the other.
- **No data folders**: If the project doesn't need `data/in` and `data/out`, skip them.
- **Additional top-level folders**: The user might want `tests/`, `docs/`, `config/`, etc. —
  add them if requested.

## Key design decisions

The `lib/` folder is for importable library code — models, algorithms, utilities — while
`cli/` holds only the user-facing CLI layer (argument parsing, help text, console output).
This separation keeps the core logic testable and reusable independently of the CLI.

The `.gitignore` is based on GitHub's standard Python template but **excludes the `lib/`
entry** since we use `lib/` as a source folder, not a packaging artifact.

The batch files use `%~dp0` (directory of the script) for all paths so they work regardless
of the current working directory.
