#!/usr/bin/env python3


from collections.abc import Iterable

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class TargetScaler(BaseEstimator, TransformerMixin):
    '''
    Scale multi-output targets relative to the first column.

    Notes
    -----
    - NaNs are ignored when estimating mean and std, 
      but are preserved in `transform` and `inverse_transform`.
    - `y` argument exists only for API compatibility.
    '''
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        '''
        Estimate per-column nan-aware mean 
        and a scale derived from nan-aware standard deviations
        (expressed in units of column 0's std). 
        '''
        X = np.asarray(X, dtype=np.float64)
        std = np.nanstd(X, axis=0)
        std[std == 0] = 1.  # prevent division by zero for constant column(s)

        self.center_ = np.nanmean(X, axis=0)
        self.scale_ = std / std[0]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.center_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) * self.scale_ + self.center_


class MSVR(BaseEstimator, RegressorMixin):
    '''
    Multi-output support vector regression (MSVR) with a radial basis function (RBF) kernel,
    fitted via iteratively reweighted least squares (IRWLS) with jointly optimized intercept `b`
    (Sánchez-Fernández et al., 2004).
    '''
    def __init__(
        self, 
        C: float = 1.0, 
        epsilon: float = 0.1, 
        gamma: str | float = "scale", 
        tol: float = 1e-3, 
        max_iter: int = 100
    ):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    def fit(self, X, y):
        def _evaluate(beta: np.ndarray, b: np.ndarray, H_beta: np.ndarray):
            '''
            Residual norms, support mask, and primal objective at `(beta, b)`.
            '''
            u = np.linalg.norm(Y - H_beta - b, axis=1)
            sv = u > self.epsilon
            loss = 0.5 * np.sum(beta * H_beta) + self.C * np.sum((u[sv] - self.epsilon) ** 2)
            return u, sv, float(loss)

        X, y = validate_data(self, X, y, multi_output=True, y_numeric=True)
        self._y_ndim = y.ndim
        Y = np.asarray(y, dtype=np.float64).reshape(len(y), -1)

        X_var = float(X.var())
        self.gamma_ = (
            (1 / (X.shape[1] * X_var) if X_var > 0 else 1.) 
            if self.gamma == "scale" else 
            float(self.gamma)  
        )

        H = np.exp(-self.gamma_ * calc_quadrance(X, X))

        beta = np.zeros_like(Y)
        b = Y.mean(axis=0, keepdims=True)  # warm-start intercept at sample mean
        H_beta = np.zeros_like(Y)
        u, sv, loss = _evaluate(beta, b, H_beta)
        self.n_iter_ = 0

        while (self.n_iter_ < self.max_iter) and sv.any():
            self.n_iter_ += 1
            a = 2.0 * self.C * (u[sv] - self.epsilon) / u[sv]  # IRWLS weights

            # Solve the SPD system M [Z_Y, z_1] = [Y_sv, 1_sv] via Cholesky,
            # then enforce sum_i beta_i = 0 (Schur complement for joint intercept b).
            M = H[np.ix_(sv, sv)].copy()
            M[np.diag_indices_from(M)] += 1.0 / a
            ones = np.ones((sv.sum(), 1), dtype=np.float64)
            rhs = np.hstack([Y[sv], ones])
            sol = solve(M, rhs, assume_a="pos")
            Z_Y, z_1 = sol[:, :-1], sol[:, -1:]

            step_b = Z_Y.sum(axis=0, keepdims=True) / z_1.sum()
            step_beta = np.zeros_like(Y)
            step_beta[sv] = Z_Y - z_1 @ step_b

            # Precompute H @ step_beta once (using sparsity of step_beta)
            H_step = H[:, sv] @ step_beta[sv]

            eta = 1.0
            while True:  # backtrack towards (beta, b) until objective does not increase
                cand_beta = eta * step_beta + (1.0 - eta) * beta
                cand_b = eta * step_b + (1.0 - eta) * b
                cand_H_beta = eta * H_step + (1.0 - eta) * H_beta
                u_c, sv_c, loss_c = _evaluate(cand_beta, cand_b, cand_H_beta)
                if (loss_c <= loss) or (eta < 1e-16):
                    break
                eta /= 10.0

            if loss_c > loss:
                break

            converged = (loss - loss_c) <= self.tol * max(loss, 1e-12)
            beta, b, H_beta, u, sv, loss = cand_beta, cand_b, cand_H_beta, u_c, sv_c, loss_c
            if converged:
                break

        self.support_ = np.flatnonzero(np.any(beta != 0, axis=1))
        self.support_vectors_ = X[self.support_].astype(np.float32, copy=False)  # halves what a saved model holds
        self.dual_coef_ = beta[self.support_]
        self.intercept_ = b.ravel()
        self.objective_ = loss

        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        if self.support_.size == 0:
            Y = np.broadcast_to(self.intercept_, (len(X), len(self.intercept_))).copy()
        else:
            Y = np.exp(-self.gamma_ * calc_quadrance(X, self.support_vectors_)) @ self.dual_coef_ + self.intercept_
        return Y.ravel() if self._y_ndim == 1 else Y


class MOGP(BaseEstimator, RegressorMixin):
    '''
    Multi-output Gaussian process (MOGP) regression under the intrinsic coregionalization model (ICM), 
    a form of co-kriging, fitted with GPyTorch.
    '''
    _icm_cls = None

    def __init__(
        self,
        rank: int | None = None,
        n_iter: int = 200,
        lr: float = 0.1,
        tol: float = 0.0,
        random_state: int | None = None,
    ):
        self.rank = rank
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_eval_model_", None)  # avoid pickling GPyTorch cache; rebuilt lazily on predict
        return state

    @classmethod
    def _get_icm_cls(cls):
        if cls._icm_cls is None:
            import gpytorch
            import torch

            class _ICM(gpytorch.models.ExactGP):
                def __init__(
                    self,
                    train_x: torch.Tensor | Iterable[torch.Tensor] | None,
                    train_y: torch.Tensor | None,
                    n_tasks: int,
                    rank: int,
                ):
                    super().__init__(
                        train_x,
                        train_y,
                        gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, has_global_noise=False),
                    )
                    self.mean_module = gpytorch.means.MultitaskMean(
                        gpytorch.means.ConstantMean(), num_tasks=n_tasks
                    )
                    log_scale = gpytorch.constraints.Positive(
                        transform=torch.exp, inv_transform=torch.log
                    )  # the default softplus moves by ~lr per step, too slow for a length scale in the hundreds
                    self.covar_module = gpytorch.kernels.MultitaskKernel(
                        gpytorch.kernels.RBFKernel(lengthscale_constraint=log_scale),
                        num_tasks=n_tasks,
                        rank=rank,
                    )

                def forward(self, x):
                    return gpytorch.distributions.MultitaskMultivariateNormal(
                        self.mean_module(x), self.covar_module(x)
                    )
        
            cls._icm_cls = _ICM

        return cls._icm_cls

    def _build_icm(self):
        '''
        Create an ICM model on the stored training data, with freshly initialized hyperparameters.
        '''
        import torch

        n_tasks = self.train_y_.shape[1]
        x = torch.tensor(self.train_x_, dtype=torch.float64)
        y = torch.tensor(self.train_y_, dtype=torch.float64)
        icm_cls = self._get_icm_cls()

        with torch.random.fork_rng():  # seeds the initial W and kappa without touching the global generator
            if self.random_state is not None:
                torch.manual_seed(self.random_state)                
            return icm_cls(x, y, n_tasks, self.rank or n_tasks).double()

    def fit(self, X, y):
        import gpytorch
        import torch

        X, y = validate_data(self, X, y, multi_output=True, y_numeric=True)
        self._y_ndim = y.ndim
        self.train_x_ = X.astype(np.float32, copy=False)
        self.train_y_ = np.asarray(y, dtype=np.float64).reshape(len(y), -1)

        model = self._build_icm()
        d2 = calc_quadrance(self.train_x_, self.train_x_)[np.triu_indices(len(self.train_x_), k=1)]
        med_d2 = float(np.median(d2)) if d2.size > 0 else 0.0
        model.covar_module.data_covar_module.lengthscale = np.sqrt(med_d2) if med_d2 > 0 else 1.0

        model.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        x, y_t = model.train_inputs[0], model.train_targets

        prev_loss = np.inf
        with gpytorch.settings.max_cholesky_size(10**9):  # exact Cholesky rather than conjugate gradients
            for step in range(self.n_iter):
                optimizer.zero_grad()
                loss = -mll(model(x), y_t)
                loss.backward()
                optimizer.step()
                cur_loss = float(loss.item())
                if self.tol > 0 and step > 0 and abs(prev_loss - cur_loss) <= self.tol * max(abs(prev_loss), 1.0):
                    break
                prev_loss = cur_loss

            with torch.no_grad():
                self.log_marginal_likelihood_value_ = float(mll(model(x), y_t)) * y_t.numel()  # mll is per data point

        self.state_ = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        with torch.no_grad():
            self.length_scale_ = float(model.covar_module.data_covar_module.lengthscale.item())
            self.coregionalization_ = model.covar_module.task_covar_module.covar_matrix.to_dense().numpy()
            self.noise_variance_ = model.likelihood.task_noises.numpy().ravel()

        model.eval()
        self._eval_model_ = model

        return self

    def predict(self, X) -> np.ndarray:
        import gpytorch
        import torch

        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        model = getattr(self, "_eval_model_", None)
        if model is None:
            model = self._build_icm()
            model.load_state_dict({k: torch.tensor(v) for k, v in self.state_.items()})
            model.eval()
            self._eval_model_ = model

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(10**9):
            Y = model(torch.tensor(X, dtype=torch.float64)).mean.numpy()

        return Y.ravel() if self._y_ndim == 1 else Y


def calc_quadrance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    Squared Euclidean distances between the rows of `A` and those of `B`, computed in float64.
    '''
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    # d2 = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2 * A @ B.T

    # return np.maximum(d2, 0.)
    return cdist(A, B, metric="sqeuclidean")
