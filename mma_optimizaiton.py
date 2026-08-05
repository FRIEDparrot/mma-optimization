import warnings
import numpy as np
from typing import Callable, Any
from dataclasses import dataclass, field
from mmapy import subsolv

def finite_difference(f: Callable, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Finite Difference Method (FDM) utilities for solving partial differential equations (PDEs).

    This function computes the numerical gradient of a scalar function `f` at a given point `x` using central differences.

    .. math::
        \\nabla f(x) \\approx \\frac{f(x + \\epsilon) - f(x - \\epsilon)}{2\\epsilon}
    """
    grad = np.zeros_like(x, dtype=float)
    for j in range(len(x)):
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[j] += eps
        x_bwd[j] -= eps
        grad[j] = (f(x_fwd) - f(x_bwd)) / (2 * eps)
    return grad

@dataclass
class MMAConfig:
    asyinit: float  = field(
        default=0.5,
        metadata={"help": "Initial asymptote scaling factor (s0 in essay)."},
    )  
    asydecr: float = field(
        default=0.7,
        metadata={"help": "Asymptote update scaling factor. (s in essay)"},
    )
    max_subproblem_iter: int = field(
        default=50,
        metadata={"help": "Maximum iterations for solving the MMA subproblem."},
    )
    pdip_tol: float = field(
        default=1e-3,
        metadata={"help": "Tolerance for dual optimization convergence."},
    )
    move: float = field(
        default=0.1,
        metadata={"help": "Move limit factor for per-iteration bound updates."},
    )
    auxiliary_penalty: float = field(
        default=1e6,
        metadata={"help": "Penalty factor for auxiliary variables in the subproblem."},
    )
    auxiliary_penalty_sq: float = field(
        default=1.0,
        metadata={"help": "Quadratic penalty factor for auxiliary variables in the subproblem."},
    )
    log_constraint: bool = field(
        default=False,
        metadata={"help": "Whether to log constraint difference during optimization."},
    )

    @classmethod
    def from_problem_config(cls, cfg: Any) -> "MMAConfig":
        """
        Build MMA config from a generic problem config object.

        The mapper accepts both generic names (e.g. ``move``) and
        MMA-prefixed names (e.g. ``mma_move``), with prefixed names taking
        precedence when both are present.
        """
        return cls(
            asyinit=getattr(cfg, "mma_asyinit", getattr(cfg, "asyinit", 0.5)),
            asydecr=getattr(cfg, "mma_asydecr", getattr(cfg, "asydecr", 0.7)),
            max_subproblem_iter=getattr(
                cfg, "mma_max_subproblem_iter", getattr(cfg, "max_subproblem_iter", 50)
            ),
            pdip_tol=getattr(cfg, "mma_pdip_tol", getattr(cfg, "pdip_tol", 1e-3)),
            move=getattr(cfg, "mma_move", getattr(cfg, "move", 0.1)),
            auxiliary_penalty=getattr(
                cfg, "mma_auxiliary_penalty", getattr(cfg, "auxiliary_penalty", 1e6)
            ),
            auxiliary_penalty_sq=getattr(
                cfg, "mma_auxiliary_penalty_sq", getattr(cfg, "auxiliary_penalty_sq", 1.0)
            ),
            log_constraint=getattr(
                cfg, "mma_log_constraint", getattr(cfg, "log_constraint", False)
            ),
        )


class MMASolver:
    r"""
    An MMA solver in python for general gradient optimization problem

    .. math ::
        \begin{aligned}
        \text{minimize}  &  \quad f_{0} (x)  \\
        \text{subject to} &  \quad f_{i} (x) \leq  \hat{f}_{i} \qquad \text{for } i = 1, \dots m \\
        &  \quad  \underline{x}_{j} \leq  x_{j} \leq \overline{x}_{j}  \quad  \text{for } j = 1, \dots n \\
        \end{aligned}

    The function x and constraints are needed. For every `step`, the objective and
        its gradient is needed, use `add_constraint` to add a constraint function

    Args:
        low_bnd (np.ndarray | None): Lower bounds for design variables.
        up_bnd (np.ndarray | None): Upper bounds for design variables.

    .. seealso::
        The official MMA theory in original paper: https://doi.org/10.1002/nme.1620240207.

    This solver automatically record the history of design variables to update the asymptotes,
        use different solver for different problem, or call `clear_history` to reset the history.
    """
    def __init__(
        self,
        low_bnd: np.ndarray,
        up_bnd: np.ndarray,
        config: MMAConfig | None = None,
    ):
        self.m = 0
        self.config = config if config is not None else MMAConfig()
        self.constraints: list[Callable] = []
        self.constraint_derivatives: list[Callable|bool] = []
        self.cons_targets: list[float | None] = []
        self.low_bnd = np.asarray(low_bnd, dtype=float).copy()
        self.up_bnd = np.asarray(up_bnd, dtype=float).copy()
        self.x_min = self.low_bnd.copy()  # prevent getting the inference
        self.x_max = self.up_bnd.copy()
        self.n = len(self.x_min)
        self.L: np.ndarray | None = None
        self.U: np.ndarray | None = None
        self.x_km1 = None
        self.x_km2 = None

    def clear_history(self):
        self.x_km1 = None
        self.x_km2 = None

    def add_constraint(
        self,
        f: Callable,
        df_dx: Callable| bool |None = None,
    ):
        r"""
        Add a constraint function (define target in function) :

        .. math::
            f_i(x) \leq 0.

        This should be a linear function that can easily compute the gradient by finite difference.

        :param f: the constraint function, should be a callable that takes x as input and return a scalar value.
        :param df_dx: the derivative of the constraint function,
            If True, we suppose (f, df_dx) is returned by f

        .. note::
            set `df_dx` = True if `f` returns `(f, df_dx)`, to save the cost of computing
        """
        self.constraints.append(f)
        if df_dx is not True and not callable(df_dx):
            warnings.warn("The derivative of the constraint is not provided, this will make solver less efficient.")
        self.constraint_derivatives.append(df_dx)
        self.cons_targets.append(0.0)
        self.m = self.m + 1

    def _set_bounds(self, x0: np.ndarray, x_min: np.ndarray | None, x_max: np.ndarray | None):
        """
        Update bounds x_min and x_max

        :param x0:
        :param x_min:
        :param x_max:
        :return:
        """
        if x_min is not None:
            assert len(x0) == len(x_min), "x0, x_min must have the same length"
            self.x_min = np.asarray(np.maximum(x_min, self.low_bnd), dtype=float)
        else:
            self.x_min = self.low_bnd.copy()
        if x_max is not None:
            assert len(x0) == len(x_max), "x0, x_max must have the same length"
            self.x_max = np.asarray(np.minimum(x_max, self.up_bnd), dtype=float)
        else:
            self.x_max = self.up_bnd.copy()

    def _initialize_asymptotes(self, x: np.ndarray):
        span = self.up_bnd - self.low_bnd
        self.L = np.maximum(x - span * self.config.asyinit, self.low_bnd)
        self.U = np.minimum(x + span * self.config.asyinit, self.up_bnd)

    def _update_asymptotes(
        self,
        x_k: np.ndarray,
        x_km1: np.ndarray | None,
        x_km2: np.ndarray | None = None,
    ):
        r"""
        Update the moving asymptotes L and U.

        For k <= 1 (not enough history), simply initialize.
        For k >= 2, detect oscillation: if (x_k - x_{k-1}) and (x_{k-1} - x_{k-2})

        When the points oscillate, shrink the asymptote :

        .. math::
            L_{j}^{(k)} = x_{j}^{(k)} - s(x_{j}^{(k-1)} - L_{j}^{(k-1)})  \\
            U_{j}^{(k)} = x_{j}^{(k)} + s (U_{j}^{(k-1)} - x_{j}^{(k-1)})

        And enlarge the asymptote when not oscillating:

        .. math::
            L_{j}^{(k)} =x_{j}^{(k)} - \frac{x_{j}^{(k-1)} - L_{j}^{(k-1)}}{s} \\
            U_{j}^{(k)}= x_{j}^{(k)} + \frac{U_{j}^{(k-1)} - x_{j}^{(k-1)}}{s}

        When 2 historical values have opposite signs, the variable is oscillating, so we shrink the asymptote
            distance. Otherwise, we enlarge it for a faster convergence.
        """
        if self.L is None or self.U is None:
            self._initialize_asymptotes(x_k)
            return

        if x_km1 is None or x_km2 is None:
            self._initialize_asymptotes(x_k)
            return

        diff1 = x_k - x_km1
        diff2 = x_km1 - x_km2
        osc = diff1 * diff2 < 0.0

        # `osc` is a boolean array, e.g., [True, False, True]
        gamma = np.where(osc, self.config.asydecr, 1.0 / np.sqrt(self.config.asydecr))
        eps = 1e-6 * (self.x_max - self.x_min)
        self.L = np.minimum(x_k - gamma * (x_km1 - self.L), x_k - eps)
        self.U = np.maximum(x_k + gamma * (self.U - x_km1), x_k + eps)

    def _compute_move_limits(self, x: np.ndarray):
        r"""
        Compute next asymptotes (move limits) alpha and beta based on current x and the asymptotes L and U.

        .. math::
            \alpha_{j}^{k}  = L_{j}^{(k)}+ 0.1 x_{j}^{(k)} \qquad  \beta_{j}^{k} = 0.9 U_{j}^{(k)} + 0.1 x_{j}^{(k)}
        
        so that : 
        
        .. math::
            L_{j}^{(k) } < \alpha_{j}^{(k)} < x_{j}^{(k)} < \beta_{j}^{(k)} < U_{j}^k

        :param x: the current design variable, used to compute the move limits alpha and beta based on the asymptotes L and U.
        :return:
        """
        if self.L is None or self.U is None:
            raise ValueError("Asymptotes must be initialized before computing move limits.")

        span = self.up_bnd - self.low_bnd
        alpha_ = np.maximum(self.L + 0.1 * (x - self.L), x - self.config.move * span)
        beta_ = np.minimum(self.U - 0.1 * (self.U - x), x + self.config.move * span)
        alpha_ = np.clip(alpha_, self.x_min, x - 1e-9)
        beta_ = np.clip(beta_, x + 1e-9, self.x_max)

        return alpha_, beta_

    def _precompute_pqr(self,
                        x_0: np.ndarray,
                        f0: float,
                        df_dx: np.ndarray) :
        r"""
        pre-compute p0, q0, r0 and p, q for all constraints to speed up the dual optimization.

        For subfunction :
        compute p_k, q_k, r_k for the k-th function (objective or constraint) at the current point x_k.
        .. math::
            p_{ij}^{(k)} = \begin{cases}
            (U_{j}^{(k)} - x_{j}^{(k)})^{2} \frac{ \partial f_{i} }{ \partial x_{j} }  &   \frac{ \partial f_{i}}{ \partial x_{j} } \geq  0  \\
            0  &  \text{others}
            \end{cases}

        .. math::
            q_{ij}^{(k)} = \begin{cases}
            - (x_{j}^{(k)} - L_{j}^{(k)})^{2} \frac{ \partial f_{i}}{ \partial x_{j} }  \qquad  \frac{ \partial f_{i} }{ \partial x_{j} }  < 0 \\
            0 \qquad \text{others}
            \end{cases}

        .. math::
            r_{i}^{(k)} = f_{i} (x^{(k)}) -  \sum_{j =1}^{n} \left( \frac{p_{ij}^{(k)}}{U_{j}^{(k)} - x_{j}^{(k)}} +  \frac{q_{ij}^{(k)}}{x_{j}^{(k)} - L_{j}^{(k)}}\right)

        :return: p0, q0, r0, p, q, b for the subproblem, where p0, q0, (r0) are for objective function, and p, q, b are for constraints.
        """
        constraints = np.zeros(self.m)
        grad_constraint = np.zeros((self.m, self.n))

        for i in range(self.m):
            cons_drv = self.constraint_derivatives[i]
            if cons_drv is True:
                constraints[i], grad_constraint[i, :] = self.constraints[i](x_0)
            elif isinstance(cons_drv, Callable):
                constraints[i] = self.constraints[i](x_0)
                grad_constraint[i, :] = cons_drv(x_0)
            else:
                constraints[i] = self.constraints[i](x_0)
                grad_constraint[i, :] = finite_difference(self.constraints[i], x_0)

        # The small positive regularization is part of the standard MMA
        # approximation.  It keeps both reciprocal terms active and avoids
        # singular dual systems when a gradient component is zero.
        xmami = np.maximum(self.x_max - self.x_min, 1e-5)
        raa0 = 1e-5

        def _compute_pqr(k: int):
            if k == 0:
                f_val = f0
                grad = df_dx
            else:
                f_val = constraints[k - 1]
                grad = grad_constraint[k - 1]

            denom_u = np.maximum(self.U - x_0, 1e-12)
            denom_l = np.maximum(x_0 - self.L, 1e-12)
            grad = np.asarray(grad, dtype=float)
            p_raw = np.maximum(grad, 0.0)
            q_raw = np.maximum(-grad, 0.0)
            regularization = 0.001 * (p_raw + q_raw) + raa0 / xmami
            p_k = (p_raw + regularization) * denom_u ** 2
            q_k = (q_raw + regularization) * denom_l ** 2
            r = f_val - np.sum(p_k / denom_u + q_k / denom_l)
            return p_k, q_k, r

        p0, q0, r0 = _compute_pqr(0)
        p = np.zeros((self.m, self.n))
        q = np.zeros((self.m, self.n))
        b = np.zeros(self.m)

        # hint : for m = 0, we have p, q, r = 0
        for i in range(self.m):
            p[i, :], q[i, :], r_i = _compute_pqr(i + 1)
            target = 0.0 if self.cons_targets[i] is None else self.cons_targets[i]
            b[i] = target - r_i

        return p0, q0, r0, p, q, b

    def _solve_subproblem(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        p0, q0, p, q, b,
        c:np.ndarray,
        d:np.ndarray,
        a0: float=1.0, a=0.0,
    ):
        r"""Solve the subproblem of the MMA using Dual method for satisfying the constraints.

        Original Subproblem
        -------------------
        After adding the slack variable `s` and the Auxiliary variables `y` and `z`, the lagrangian can be written as :

        .. math::
            \begin{aligned}
            & \text{minimize}  &&  \psi (\boldsymbol{x}) =  \sum_{j = 1}^{n} \left(\frac{p_{0j}}{U_{j} - x_{j}} + \frac{q_{0j}}{x_{j} - L_{j}}\right) + a_{0} z + \sum_{i = 1}^{m} (c_{i} y_{i} +\frac{1}{2} d_{i} y_{i}^{2})  \\
            & \text{subject to}  &&   \sum_{j = 1}^{n} \left( \frac{p_{ij}}{ U_{j} - x_{j} } + \frac{q_{ij}}{x_{j} - L_{j}} \right) - a_{i} z_{i} - y_{i} \leq  b_{i} + s \qquad  \text{for } i = 1\dots  m   \\
            &  && \alpha_{j} \leq  x_{j} \leq  \beta_{j} \qquad   y_{i} \geq  0  \qquad z \geq  0 \qquad  s \geq  0
            \end{aligned}

        The Lagrangian of this problem is defined as :

        .. math::
            \begin{aligned} \mathcal{L}(x, y, z, \lambda, \xi, \eta, \mu, \zeta)
            &= \underbrace{\sum_{j=1}^n \left( \frac{p_{0j}}{U_j - x_j} + \frac{q_{0j}}{x_j - L_j} \right) + a_0 z +
            \sum_{i=1}^m \left( c_i y_i + \frac{1}{2} d_i y_i^2 \right)}_{\text{Objective } \psi}  \\
             &+ \sum_{i=1}^m \lambda_i \left[ \sum_{j=1}^n \left( \frac{p_{ij}}{U_j - x_j} + \frac{q_{ij}}{x_j - L_j} \right) - a_i z - y_i - b_{i} +s_{i}\right] \\
             &+ \sum_{j=1}^n \xi_j ( \alpha_{j}-x_{j}) + \sum_{j=1}^n \eta_j (x_{j} - \beta_{j})  \\
              &- \sum_{i=1}^m \mu_i y_i - \underbrace{ \zeta z }_{ L_{\zeta} } \end{aligned}

        Since original dual solution of the problem in essay is unstable,
            we use PDIP (Primal-Dual Interior Point) method to solve this problem, directly
            call MMA subproblem (Method of Moving Asymptotes) implemented by GCMMA-python (mmapy)

        .. seealso::
            for original source code of GCMMA, see https://github.com/arjendeetman/GCMMA-MMA-Python

        :param alpha:
        :param beta:
        :param c: m size array for linear penalty of auxiliary variable y in the objective
        :param d: m size array for quadratic penalty of auxiliary variable y in the objective
        :return:
        """
        cfg = self.config
        x_mma, y_mma, z_mma, lmbda_mma, xi_mma, eta_mma, mu_mma, zeta_mma, s_mma = subsolv(
            m=self.m, n=self.n, epsimin=cfg.pdip_tol, low=self.L.reshape(-1, 1), upp=self.U.reshape(-1, 1),
            alfa=alpha.reshape(-1, 1), beta=beta.reshape(-1, 1), p0=p0.reshape(-1, 1), q0=q0.reshape(-1, 1),
            P=p, Q=q, a0=a0, a=np.full((self.m, 1), a), b=b.reshape(-1, 1),
            c=c.reshape(-1, 1), d=d.reshape(-1, 1),
        )
        x_new = x_mma.ravel()  # flatten back to 1-D vector
        return x_new

    def step(
        self,
        x0: np.ndarray,
        f0: float,
        df0_dx: np.ndarray,
        x_min: np.ndarray | None = None,
        x_max: np.ndarray | None = None,
    ) -> np.ndarray:
        """Perform one step of MMA optimization, return the updated design variable x_new.
        :param x0:
        :param f0: objective function at x0, i.e., f0 = f(x0)
        :param df0_dx: derivative of objective function, if None, it will be computed by finite difference,
                which is less efficient.
        :param x_min:
        :param x_max:
        :return:
            updated design variable x_new after one step of MMA optimization.
        """
        cfg = self.config
        x_k = np.asarray(x0, dtype=float)
        self._set_bounds(x_k, x_min, x_max)
        self._update_asymptotes(x_k, self.x_km1, self.x_km2)
        alpha_k, beta_k = self._compute_move_limits(x_k)
        # MMA is invariant to multiplying the objective by a positive
        # constant, but the primal-dual solve is not.  Homogenization and
        # elasticity objectives can be many orders of magnitude larger than
        # the normalized volume constraint, so normalize objective data before
        # building the reciprocal approximation.
        objective_gradient = np.asarray(df0_dx, dtype=float)
        objective_scale = max(
            1.0,
            float(np.max(np.abs(objective_gradient), initial=0.0)),
        )
        f0_scaled = float(f0) / objective_scale
        df0_dx_scaled = objective_gradient / objective_scale

        # pre-compute p, q, r for all constraints to speed up the dual optimization.
        p0, q0, r0, p, q, b = self._precompute_pqr(x_k, f0_scaled, df0_dx_scaled)
        x_new = self._solve_subproblem(
            alpha_k, beta_k, p0, q0, p, q, b,
            c=np.full(self.m, cfg.auxiliary_penalty),
            d=np.full(self.m, cfg.auxiliary_penalty_sq),
        )
        self.x_km2 = self.x_km1
        self.x_km1 = x_k
        if self.config.log_constraint:
            cons_values = [cons(x_k)[0] if self.constraint_derivatives[i] is True else cons(x_k) for i, cons in enumerate(self.constraints)]
            print(f"Constraint values at current point: {cons_values}")
        return x_new
