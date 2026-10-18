#!/bin/bash
set -e

# Install Julia, NeoGP.jl and its dependencies into the conda environment, so they
# are found at run time (the image is built as $MAMBA_USER, but the experiments run
# as root with a different ~/.julia). regressor.py sets the same depot.
# The trailing separator appends Julia's default depots (with the bundled stdlibs).
export JULIA_DEPOT_PATH="${CONDA_PREFIX}/julia_depot:"
# Precompile for a range of x86-64 CPUs (as the official Julia binaries), not only
# for the CPU of the machine that builds the image.
if [ "$(uname -m)" = "x86_64" ]; then
    export JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)"
fi

# The first import of pyneogp installs and precompiles Julia and NeoGP.jl; a short
# run checks the installation.
python - <<'PY'
import numpy as np
from pyneogp.sklearn import NeoGPRegressor

rng = np.random.default_rng(0)
X = rng.uniform(1, 5, size=(100, 2))
y = X[:, 0] * np.exp(-0.5 * X[:, 1])
model = NeoGPRegressor(generations=3, popsize=50, maxlen=20, random_state=1, threads=1)
model.fit(X, y)
print("NeoGP model:", model.get_best_sympy(), "R2:", model.score(X, y))
PY
