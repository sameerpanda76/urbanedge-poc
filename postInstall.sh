#!/bin/bash
echo "🔧 Building CmdStan backend for Prophet..."
pip install --upgrade pip
pip install cmdstanpy
python - << 'EOF'
import cmdstanpy
print("CmdStan build started…")
cmdstanpy.install_cmdstan()
print("✅ CmdStan build finished.")
EOF