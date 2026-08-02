# Quick Start Guide

## Environment Setup (First Time)

### 1. Create Virtual Environment
```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment
```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Your prompt should now show `(venv)` prefix. You're ready to go!

## Deactivate When Done
```bash
deactivate
```

## Verify Installation
```bash
python -c "import torch; print(torch.__version__)"
```

## Typical Workflow

```bash
# Start session
source venv/bin/activate

# Run training or notebooks
python applications/vae/path_a_scratch/train.py
# or
jupyter notebook

# When done
deactivate
```

## Adding New Dependencies

If you need a new package:

```bash
# Install it
pip install package_name

# Update requirements.txt with the version
pip freeze | grep package_name >> requirements.txt
```

Then commit the updated `requirements.txt` to git.

## Troubleshooting

### "venv not found"
Make sure you're in the project root directory:
```bash
ls CLAUDE.md  # Should work if in right directory
```

### "torch import fails"
The venv might not be activated. Check:
```bash
which python  # Should show path containing /venv/bin/python
```

### Fresh start needed
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

See `_specs/repo_overview.md` for detailed project structure and strategy.
