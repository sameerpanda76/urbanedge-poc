#!/bin/bash
pip install --upgrade pip
pip install cmdstanpy
python - << 'EOF'
import cmdstanpy
cmdstanpy.install_cmdstan()
EOF