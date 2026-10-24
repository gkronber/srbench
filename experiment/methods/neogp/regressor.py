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

from juliacall import Main as jl  # noqa: E402
from pyneogp import neogp  # noqa: E402
from pyneogp.sklearn import NeoGPRegressor  # noqa: E402
from sklearn.utils.validation import validate_data  # noqa: E402

# Called after each generation, it keeps the individual with the lowest criterion of all
# Pareto fronts.  Individuals are not changed once they are in the population, so they
# are kept by reference and the criterion is evaluated once for each.
jl.seval("""
mutable struct BestOfRun{G,C}
    const gp::G
    const criterion::C
    const seen::IdDict{Any,Nothing}
    best::Any
    value::Float64
end
""")
best_of_run = jl.seval("""
begin
    function (t::BestOfRun)()
        for (indiv, _) in NeoGP.best_individuals(t.gp)
            haskey(t.seen, indiv) && continue
            t.seen[indiv] = nothing
            c = Float64(t.criterion(indiv))
            if t.best === nothing || c < t.value
                t.best, t.value = indiv, c
            end
        end
        nothing
    end
    (gp, criterion) -> BestOfRun(gp, criterion, IdDict{Any,Nothing}(), nothing, Inf)
end
""")


class NeoGPBestOfRun(NeoGPRegressor):
    """
    NeoGPRegressor (Gaussian likelihood) whose model is the expression with the lowest
    model-selection criterion of the Pareto fronts of all generations, not only of the
    final front.  Float32 data are kept (Float32 fits, criterion and predictions), any
    other data are converted to Float64.
    """

    def fit(self, X, y):
        start = time.monotonic()
        X, y = validate_data(
            self, X, y, y_numeric=True, dtype=(np.float64, np.float32)
        )
        self.dtype_ = X.dtype
        dataset = neogp.Dataset(X, y, None, None, self._varnames())
        lik = neogp.create_likelihood(self.likelihood, dataset)
        kwargs = dict(
            seed=-1 if self.random_state is None else int(self.random_state),
            popsize=self.popsize,
            maxlen=self.maxlen,
            tournamentsize=self.tournamentsize,
            mutation_prob=self.mutation_prob,
            crossover_prob=self.crossover_prob,
            maxeval=self.maxeval,
            threads=self.threads,
            indiv_loss_func=self._objectives(),
        )
        if self.functionset is not None:
            kwargs["functionset"] = [
                getattr(neogp, name.strip()) for name in self.functionset.split(",")
            ]
        gp = neogp.NSGA2(lik, X.shape[1], **kwargs)
        neogp.initialize_b(gp)
        tracker = best_of_run(gp, neogp.to_julia(self._criterion()))
        tracker()
        remaining = (
            float("inf")
            if self.max_time is None
            else float(self.max_time - (time.monotonic() - start))
        )
        if remaining > 0:
            neogp.evolve_b(
                gp, self.generations, iter_callback=tracker, max_time=remaining
            )
        self.nsga2_ = gp
        self.best_ = tracker.best
        self.criterion_ = float(tracker.value)
        return self

    def predict(self, X):
        check_is_fitted(self, "best_")
        X = validate_data(self, X, reset=False, dtype=self.dtype_)
        return np.asarray(neogp.predict(self.best_, X))

    def _best(self):
        return self.best_


def needs_float64(X, y, max_abs=1e6):
    """True if Float32 does not suffice for the data: a value of `X` or `y` is outside
    1/max_abs <= |v| <= max_abs (products and powers would leave the range of
    Float32), or converting a column to Float32 merges distinct values."""
    data = np.column_stack([np.asarray(X, dtype=np.float64), np.asarray(y, np.float64)])
    a = np.abs(data[data != 0])
    if a.size and (a.max() > max_abs or a.min() < 1 / max_abs):
        return True
    for c in data.T:
        if len(np.unique(c.astype(np.float32))) < len(np.unique(c)):
            return True
    return False


def tournament_size(popsize):
    """A larger selection pressure for larger populations."""
    for max_popsize, size in [(1000, 2), (3000, 3), (5000, 4), (10000, 5)]:
        if popsize <= max_popsize:
            return size
    return 6


class NeoGPRestarts(RegressorMixin, BaseEstimator):
    """
    Independent runs of NeoGPBestOfRun (NSGA2) with at most `generations` generations,
    restarted until the time limit `max_time` (seconds) or `max_restarts` runs.  The
    model is the expression with the lowest model-selection criterion (the description
    length, with the parameter complexity `param_complexity`: "rot", "diag",
    "rot_scaled" or "det") of all runs.  Each parameter fit (Levenberg-Marquardt with
    Broyden updates) is limited to `maxeval` evaluations.

    A run is started only if the remaining time suffices for its initialization and
    one generation (estimated from the previous run); the last run stops at the time
    limit.  The runs are seeded from `random_state`.

    The tournament size grows with the population size if `tournamentsize` is None.

    `objective` selects the two objectives of NSGA2 as in NeoGPRegressor: "nll"
    (negative log-likelihood and length) or "dl" (the rounded negative log-likelihood
    and the complexity penalty of the description length).  `precision` is "float32",
    "float64" or "auto": Float64 only if the magnitudes of the inputs or the target are
    large or small, or Float32 would merge distinct values (see `needs_float64`).
    """

    def __init__(
        self,
        max_time=3600,
        random_state=None,
        generations=100,
        popsize=1000,
        tournamentsize=None,
        maxlen=100,
        maxeval=200,
        max_restarts=None,
        threads=0,
        objective="nll",
        precision="auto",
        param_complexity="rot",
    ):
        self.max_time = max_time
        self.random_state = random_state
        self.generations = generations
        self.popsize = popsize
        self.tournamentsize = tournamentsize
        self.maxlen = maxlen
        self.maxeval = maxeval
        self.max_restarts = max_restarts
        self.threads = threads
        self.objective = objective
        self.precision = precision
        self.param_complexity = param_complexity

    def fit(self, X, y):
        start = time.monotonic()
        if self.precision == "auto":
            self.dtype_ = np.float64 if needs_float64(X, y) else np.float32
        elif self.precision in ("float32", "float64"):
            self.dtype_ = np.dtype(self.precision).type
        else:
            raise ValueError(
                f"precision must be auto, float32 or float64, got {self.precision!r}"
            )
        X, y = X.astype(self.dtype_), np.asarray(y).astype(self.dtype_)
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
            run = NeoGPBestOfRun(
                generations=self.generations,
                max_time=remaining,
                popsize=self.popsize,
                tournamentsize=tournamentsize,
                maxlen=self.maxlen,
                maxeval=self.maxeval,
                random_state=None
                if self.random_state is None
                else int(seeds.integers(2**31)),
                threads=self.threads,
                objective=self.objective,
                param_complexity=self.param_complexity,
            ).fit(X, y)
            generation_time = (time.monotonic() - run_start) / (
                run.nsga2_.generation + 1
            )
            self.n_restarts_ += 1

            if self.n_restarts_ == 1 or run.criterion_ < self.criterion_:
                self.criterion_ = run.criterion_
                self.best_estimator_ = run
        return self

    def predict(self, X):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X.astype(self.dtype_))


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
    maxlen=100,
    maxeval=200,
    threads=num_threads,
)

eval_kwargs = {
    "pre_train": pre_train,
    "test_params": dict(generations=5, popsize=100, max_restarts=2),
}
