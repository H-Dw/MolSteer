# Flow-matching control: model assumptions and gradient conflicts

These are mathematical reference notes, not a universal controller or evidence of molecular efficacy.

## Endpoint objectives and live coordinates

For a differentiable endpoint prediction h_t(x_t), the exact local chain rule is
g_i = J_h_t(x_t)^T grad f_i(h_t(x_t)). A saved endpoint coordinate gradient does
not include that Jacobian. State and endpoint terms require the same live variable,
coordinate units and editable mask before a cosine is meaningful. A graph change
invalidates branch-specific functions and references.

Flow matching learns time-dependent vector fields along chosen probability paths.
An endpoint-prediction implementation must declare how its integrator turns that
prediction into a velocity. The time label alone does not fix a guidance sign.
Source: Lipman et al., *Flow Matching for Generative Modeling*,
https://arxiv.org/abs/2210.02747 (2023 revision).

## Guidance is a separate approximation

General flow guidance depends on the probability path and coupling. Localized or
Gaussian approximations require their respective assumptions; an energy gradient
with an arbitrary coefficient is not automatically exact tilted-distribution
sampling. Preserve the approximation and parameter origins alongside any adapted
control law. In MolSteer, dt times a minimizing direction is the declared FLOWR
intervention convention, not a proof of exact posterior sampling.
Source: Feng et al., *On the Guidance of Flow Matching*, sections 3.1–3.4,
https://arxiv.org/html/2502.02150v3 (2025).

## Common descent in a declared editable space

Let g_i denote gradients of normalized local deficits in the same editable space.
Solve alpha* = argmin_{alpha >= 0, sum(alpha)=1} ||sum_i alpha_i g_i||^2 and
d = -sum_i alpha*_i g_i. Check convergence and each g_i^T d. A negative pairwise
cosine is a warning, not proof that common descent is impossible. A zero derivative
has undefined cosine. Satisfied zero-gradient objectives can be inactive; an
unsatisfied zero-gradient objective is an unresolved control limitation.

These coefficients follow a local optimization problem, not biological importance.
This masked Euclidean solve does not implement a general constrained QP. Check
hard predicates after the actual nonlinear proposal. Nonuniform clipping can change
directional signs and requires another check. Rejection retains the native state,
conditioning and random stream. Local descent makes no terminal efficacy guarantee.
Method reference: Sener and Koltun, *Multi-Task Learning as Multi-Objective Optimization*,
https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html .
