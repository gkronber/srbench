import os
import sys
import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

# Julia reads these when juliacall is first imported, so set them before importing
# pyneogp.  SRBench runs each job on `OMP_NUM_THREADS` cores (pyneogp defaults to all).
num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
os.environ.setdefault("PYTHON_JULIACALL_THREADS", str(num_threads))
# The Julia packages are installed into the conda environment by install.sh (the image
# is built as $MAMBA_USER, but the experiments run as root with a different ~/.julia).
# The trailing separator appends Julia's default depots (with the bundled stdlibs).
os.environ.setdefault(
    "JULIA_DEPOT_PATH", os.path.join(sys.prefix, "julia_depot") + os.pathsep
)

from pyneogp.sklearn import NeoGPRegressor  # noqa: E402


def tournament_size(popsize):
    """A larger selection pressure for larger populations."""
    for max_popsize, size in [(1000, 2), (3000, 3), (5000, 4), (10000, 5)]:
        if popsize <= max_popsize:
            return size
    return 6


class NeoGPRestarts(RegressorMixin, BaseEstimator):
    """
    Independent runs of NeoGPRegressor (NSGA2) with at most `generations` generations,
    restarted until the time limit `max_time` (seconds) or `max_restarts` runs.  The
    model is the expression with the lowest model-selection criterion (the description
    length) of all runs.

    A run is started only if the remaining time suffices for its initialization and
    one generation (estimated from the previous run); the last run stops at the time
    limit.  The runs are seeded from `random_state`.

    The tournament size grows with the population size if `tournamentsize` is None.
    """

    def __init__(
        self,
        max_time=3600,
        random_state=None,
        generations=100,
        popsize=1000,
        tournamentsize=None,
        maxlen=50,
        max_restarts=None,
        threads=0,
    ):
        self.max_time = max_time
        self.random_state = random_state
        self.generations = generations
        self.popsize = popsize
        self.tournamentsize = tournamentsize
        self.maxlen = maxlen
        self.max_restarts = max_restarts
        self.threads = threads

    def fit(self, X, y):
        start = time.monotonic()
        seeds = np.random.default_rng(self.random_state)
        tournamentsize = self.tournamentsize
        if tournamentsize is None:
            tournamentsize = tournament_size(self.popsize)
        self.n_restarts_ = 0
        self.criterion_ = float("inf")
        generation_time = 0.0  # of the previous run, initialization counts as one
        while self.max_restarts is None or self.n_restarts_ < self.max_restarts:
            remaining = None
            if self.max_time is not None:
                remaining = self.max_time - (time.monotonic() - start)
                if self.n_restarts_ > 0 and remaining < 2 * generation_time:
                    break
            run_start = time.monotonic()
            run = NeoGPRegressor(
                generations=self.generations,
                max_time=remaining,
                popsize=self.popsize,
                tournamentsize=tournamentsize,
                maxlen=self.maxlen,
                random_state=None
                if self.random_state is None
                else int(seeds.integers(2**31)),
                threads=self.threads,
            ).fit(X, y)
            generation_time = (time.monotonic() - run_start) / (
                run.nsga2_.generation + 1
            )
            self.n_restarts_ += 1

            criterion: float = min(e["criterion"] for e in run.get_pareto_front())
            if self.n_restarts_ == 1 or criterion < self.criterion_:
                self.criterion_ = criterion
                self.best_estimator_ = run
        return self

    def predict(self, X):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)


def model(est, X=None):
    # the variables are named by the columns of the training DataFrame
    return str(est.best_estimator_.get_best_sympy())


def pre_train(est, X, y):
    # a larger population for many features
    est.set_params(popsize=max(1000, 100 * X.shape[1]))


est = NeoGPRestarts(
    max_time=3600,
    generations=100,
    popsize=1000,
    maxlen=50,
    threads=num_threads,
)

eval_kwargs = {
    "pre_train": pre_train,
    "test_params": dict(generations=5, popsize=100, max_restarts=2),
}
