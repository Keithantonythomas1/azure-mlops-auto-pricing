
#!/usr/bin/env bash
set -e
python scripts/prep.py
python scripts/train.py
python scripts/tune.py
python scripts/register_model.py
