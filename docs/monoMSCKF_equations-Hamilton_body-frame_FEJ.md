# Mono MSCKF equations — Hamilton, body-frame error, FEJ

Oct 19, 2026 · @Yonatan

## 1. Conventions and state

All equations here use Hamilton quaternions, body-to-world rotation storage, a body-frame (right) attitude error, and IMU-pose clones. They are the MSCKF and MSCKF 2.0 equations rederived under those choices, with FEJ applied to every state that appears in the unobservable subspace.

| Item | Definition |
| --- | --- |
| World frame G | Local, gravity-aligned, z up; fixed at initialization; Earth rotation ignored |
| Gravity | g = \[0, 0, −9.81\]ᵀ m/s² in G |
| Attitude | R = R\_GI (Hamilton q\_GI), maps body vectors to world: ᴳx = R ⁱx |
| Attitude error | R = R̂ Exp(δθ), δθ ∈ ℝ³ in the body frame I |
| Velocity, position | v, p of I, expressed in G; additive errors δv = v − v̂, δp = p − p̂ |
| Biases | b\_g, b\_a in I; additive errors |
| Extrinsic | R\_IC, p\_IC: camera pose in the IMU frame; R\_IC = R̂\_IC Exp(δφ), p\_IC = p̂\_IC + δp\_IC |
| Time offset | t\_imu = t\_cam + t\_d; additive error δt\_d |
| Clone i | IMU pose (R\_i, p\_i) at image time i, same error form as the IMU attitude and position |

The error state, in the MSCKF order, is

```latex
\tilde x = \big[\,\underbrace{\delta\theta^\top\ \delta b_g^\top\ \delta v^\top\ \delta b_a^\top\ \delta p^\top}_{\tilde x_I\ (15)}\ \ \underbrace{\delta\phi^\top\ \delta p_{IC}^\top\ \delta t_d}_{\tilde x_W\ (7)}\ \ \underbrace{\delta\theta_1^\top\ \delta p_1^\top \cdots \delta\theta_N^\top\ \delta p_N^\top}_{\tilde x_C\ (6N)}\,\big]^\top
```

Notation: ⌊a×⌋ is the skew-symmetric matrix of a, so ⌊a×⌋ b = a × b. Exp and Log are the SO(3) exponential and logarithm, and J\_r is the right Jacobian of SO(3). A hat means the current estimate. A star means the first estimate, the value FEJ freezes (Section 7).

Translation from the JPL papers: wherever they write C(q̂) use R̂ᵀ, and wherever they write C(q̂)ᵀ use R̂. The 2007 MSCKF and OpenVINS use the same body-frame δθ. MSCKF 2.0 uses a global-frame error, related by δθ\_G = R̂ δθ.

## 2. IMU model and nominal propagation

The IMU measures body-frame angular rate and specific force, corrupted by bias and white noise. Biases are random walks.

```latex
\omega_m = \omega + b_g + n_g, \qquad a_m = R^\top(a - g) + b_a + n_a, \qquad \dot b_g = n_{wg}, \qquad \dot b_a = n_{wa}
```

With the current bias estimates, the corrected readings are ω̂ = ω\_m − b̂\_g and â = a\_m − b̂\_a. Their errors are ω − ω̂ = −δb\_g − n\_g and the body-frame specific-force error is −δb\_a − n\_a.

Nominal kinematics, with noise set to zero:

```latex
\dot{\hat R} = \hat R\lfloor\hat\omega\times\rfloor, \qquad \dot{\hat v} = \hat R\,\hat a + g, \qquad \dot{\hat p} = \hat v, \qquad \dot{\hat b}_g = \dot{\hat b}_a = 0
```

**Preintegrated form.** Between times k and k+1, integrate the IMU samples m in the body frame of time k, starting from ΔR = I and Δv = Δp = 0:

```latex
\Delta p \leftarrow \Delta p + \Delta v\,\delta t + \tfrac12\Delta R\,\hat a_m\,\delta t^2, \qquad \Delta v \leftarrow \Delta v + \Delta R\,\hat a_m\,\delta t, \qquad \Delta R \leftarrow \Delta R\,\mathrm{Exp}(\hat\omega_m\,\delta t)
```

Update in that order, so Δp uses the old Δv and ΔR. Then the nominal state at k+1 is

```latex
\hat R_{k+1} = \hat R_k\,\Delta R, \qquad \hat v_{k+1} = \hat v_k + g\,\Delta t + \hat R_k\,\Delta v, \qquad \hat p_{k+1} = \hat p_k + \hat v_k\,\Delta t + \tfrac12 g\,\Delta t^2 + \hat R_k\,\Delta p
```

Here δt is one IMU sample period and Δt = t\_{k+1} − t\_k. In practice k and k+1 are consecutive image times, propagation stops at t\_cam + t̂\_d, and the last sample is interpolated to land exactly there. This is the same integration as per-sample Euler, midpoint or RK4, rewritten so that Φ can be formed in closed form (Section 4).

## 3. Continuous-time error dynamics

The IMU error obeys a linear system driven by the IMU noise. The calibration and clone errors are constant between updates.

```latex
\dot{\tilde x}_I = F\,\tilde x_I + G\,n, \qquad n = \begin{bmatrix} n_g^\top & n_{wg}^\top & n_a^\top & n_{wa}^\top \end{bmatrix}^\top
```

**Attitude.** Differentiate R = R̂(I + ⌊δθ×⌋), substitute Ṙ = R⌊ω×⌋ and the nominal equation, cancel R̂, and drop second-order terms. Using ⌊a×⌋⌊b×⌋ − ⌊b×⌋⌊a×⌋ = ⌊(a×b)×⌋ gives

```latex
\dot{\delta\theta} = -\lfloor\hat\omega\times\rfloor\,\delta\theta - \delta b_g - n_g
```

**Velocity.** Write v̇ = R(â − δb\_a − n\_a) + g with R ≈ R̂(I + ⌊δθ×⌋), subtract the nominal equation, and use ⌊δθ×⌋â = −⌊â×⌋δθ. Gravity cancels because it is known.

```latex
\dot{\delta v} = -\hat R\lfloor\hat a\times\rfloor\,\delta\theta - \hat R\,\delta b_a - \hat R\,n_a
```

**Position and biases.**

```latex
\dot{\delta p} = \delta v, \qquad \dot{\delta b}_g = n_{wg}, \qquad \dot{\delta b}_a = n_{wa}
```

Stacked in the order \[δθ, δb\_g, δv, δb\_a, δp\], with 3×3 blocks:

```latex
F = \begin{bmatrix}
-\lfloor\hat\omega\times\rfloor & -I & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
-\hat R\lfloor\hat a\times\rfloor & 0 & 0 & -\hat R & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & I & 0 & 0
\end{bmatrix},
\qquad
G = \begin{bmatrix}
-I & 0 & 0 & 0 \\
0 & I & 0 & 0 \\
0 & 0 & -\hat R & 0 \\
0 & 0 & 0 & I \\
0 & 0 & 0 & 0
\end{bmatrix}
```

```latex
Q_c = \mathrm{diag}\big(\sigma_g^2 I,\ \sigma_{wg}^2 I,\ \sigma_a^2 I,\ \sigma_{wa}^2 I\big)
```

The σ values are continuous-time densities from the datasheet or an Allan-variance fit: rad/s/√Hz and m/s²/√Hz for white noise, rad/s²/√Hz and m/s³/√Hz for the bias random walks. These match Solà (2017), eqs. (247) and (259), with the state permuted and without his gravity state. They also match the 2007 MSCKF with Cᵀ → R̂ and the Earth-rate terms removed.

## 4. Discrete transition matrix Φ and noise Q

Φ maps the error at k to the error at k+1 exactly to first order, for any interval, given the preintegrated deltas. It is derived by perturbing the three propagation equations of Section 2.

**Bias Jacobians of the deltas.** Define the first-order bias corrections

```latex
\Delta R(\hat b + \delta b) \approx \Delta R\,\mathrm{Exp}(J^{R}_{g}\,\delta b_g), \qquad \Delta v(\hat b + \delta b) \approx \Delta v + J^{v}_{g}\delta b_g + J^{v}_{a}\delta b_a, \qquad \Delta p(\hat b + \delta b) \approx \Delta p + J^{p}_{g}\delta b_g + J^{p}_{a}\delta b_a
```

They start at zero and are accumulated per IMU sample alongside the deltas, using the values from before the sample in each line (Forster et al. 2017):

```latex
\begin{aligned}
J^{p}_{a} &\leftarrow J^{p}_{a} + J^{v}_{a}\,\delta t - \tfrac12\Delta R\,\delta t^2, &
J^{p}_{g} &\leftarrow J^{p}_{g} + J^{v}_{g}\,\delta t - \tfrac12\Delta R\lfloor\hat a_m\times\rfloor J^{R}_{g}\,\delta t^2, \\
J^{v}_{a} &\leftarrow J^{v}_{a} - \Delta R\,\delta t, &
J^{v}_{g} &\leftarrow J^{v}_{g} - \Delta R\lfloor\hat a_m\times\rfloor J^{R}_{g}\,\delta t, \\
J^{R}_{g} &\leftarrow \mathrm{Exp}(\hat\omega_m\delta t)^\top J^{R}_{g} - J_r(\hat\omega_m\delta t)\,\delta t
\end{aligned}
```

**Attitude.** Substitute R\_k = R̂\_k Exp(δθ\_k) and the bias correction into R\_{k+1} = R\_k ΔR, and move Exp past ΔR with Exp(a)ΔR = ΔR Exp(ΔRᵀa):

```latex
R_{k+1} \approx \hat R_k\,\Delta R\,\mathrm{Exp}(\Delta R^\top\delta\theta_k)\,\mathrm{Exp}(J^R_g\delta b_g)
\;\Rightarrow\;
\delta\theta_{k+1} = \Delta R^\top\delta\theta_k + J^R_g\,\delta b_g
```

**Velocity.** Substitute into v\_{k+1} = v\_k + gΔt + R\_k Δv, use R\_k ≈ R̂\_k(I + ⌊δθ\_k×⌋), keep first-order terms, and flip the cross product:

```latex
\delta v_{k+1} = \delta v_k - \hat R_k\lfloor\Delta v\times\rfloor\delta\theta_k + \hat R_k J^v_g\,\delta b_g + \hat R_k J^v_a\,\delta b_a
```

**Position.** In the same way, from p\_{k+1} = p\_k + v\_kΔt + ½gΔt² + R\_k Δp:

```latex
\delta p_{k+1} = \delta p_k + \Delta t\,\delta v_k - \hat R_k\lfloor\Delta p\times\rfloor\delta\theta_k + \hat R_k J^p_g\,\delta b_g + \hat R_k J^p_a\,\delta b_a
```

**Endpoint form.** Since R̂\_k⌊Δv×⌋ = ⌊(R̂\_kΔv)×⌋R̂\_k, and R̂\_kΔv and R̂\_kΔp are fixed by the nominal equations, the attitude columns can be written with the states at both ends of the interval. This is the form MSCKF 2.0 uses, and the one FEJ needs (Section 7):

```latex
\hat R_k\lfloor\Delta v\times\rfloor = \lfloor(\hat v_{k+1} - \hat v_k - g\Delta t)\times\rfloor\hat R_k, \qquad
\hat R_k\lfloor\Delta p\times\rfloor = \lfloor(\hat p_{k+1} - \hat p_k - \hat v_k\Delta t - \tfrac12 g\Delta t^2)\times\rfloor\hat R_k
```

**Φ, in the order \[δθ, δb\_g, δv, δb\_a, δp\]:**

```latex
\Phi_{k} = \begin{bmatrix}
\Phi_{\theta\theta} & J^R_g & 0 & 0 & 0 \\
0 & I & 0 & 0 & 0 \\
-\lfloor(\hat v_{k+1} - \hat v_k - g\Delta t)\times\rfloor\hat R_k & \hat R_k J^v_g & I & \hat R_k J^v_a & 0 \\
0 & 0 & 0 & I & 0 \\
-\lfloor(\hat p_{k+1} - \hat p_k - \hat v_k\Delta t - \tfrac12 g\Delta t^2)\times\rfloor\hat R_k & \hat R_k J^p_g & \Delta t\,I & \hat R_k J^p_a & I
\end{bmatrix},
\qquad \Phi_{\theta\theta} = \Delta R^\top = \hat R_{k+1}^\top\hat R_k
```

The last identity holds for the nominal propagation. Writing Φ\_θθ with the endpoint attitudes is what makes FEJ possible for a body-frame error. For a single IMU sample this reduces to I + Fδt, which is a quick unit test. MSCKF 2.0's own Φ has Φ\_θθ = I, because its error is global; that is the main block-level difference.

**Discrete noise.** Accumulate the noise over the interval per IMU sample m, starting from Q = 0, with Φ\_m the single-sample transition and G\_m evaluated at the current estimates:

```latex
Q \leftarrow \Phi_m\,Q\,\Phi_m^\top + G_m\,Q_c\,G_m^\top\,\delta t
```

Noise does not affect the nullspace, so Q needs no FEJ.

## 5. Covariance propagation and cloning

**Propagation.** Partition P into the IMU block (I), the calibration block (W) and the clones (C). Calibration and clones are constant between updates, so only the rows and columns that touch the IMU state change:

```latex
P = \begin{bmatrix} P_{II} & P_{IR} \\ P_{RI} & P_{RR} \end{bmatrix}
\;\rightarrow\;
\begin{bmatrix} \Phi_k P_{II}\Phi_k^\top + Q & \Phi_k P_{IR} \\ P_{RI}\Phi_k^\top & P_{RR} \end{bmatrix},
\qquad R = \{W, C\}
```

Propagate the covariance once per image interval with the closed-form Φ of Section 4, or per IMU sample with the single-sample Φ\_m. Both give the same result. The closed form is cheaper and is where FEJ is applied.

**Cloning.** At each image time t\_cam + t̂\_d, append a copy of the IMU pose. A small time-offset error moves the true clone time, and so the true clone pose, along the motion:

```latex
R(t + \delta t_d) \approx \hat R\,\mathrm{Exp}(\hat\omega\,\delta t_d), \qquad p(t + \delta t_d) \approx \hat p + \hat v\,\delta t_d
```

So the new clone's error is the IMU pose error plus a time-offset term:

```latex
\delta\theta_c = \delta\theta + \hat\omega\,\delta t_d, \qquad \delta p_c = \delta p + \hat v\,\delta t_d
```

That defines the augmentation Jacobian J, which is zero except for these blocks:

```latex
J = \big[\ J_I\ \ J_W\ \ 0_{6\times 6N}\ \big], \qquad
J_I = \begin{bmatrix} I & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & I \end{bmatrix}, \qquad
J_W = \begin{bmatrix} 0_{3\times6} & \hat\omega \\ 0_{3\times6} & \hat v \end{bmatrix}
```

The covariance grows to include the clone and its correlations with everything else:

```latex
P \leftarrow \begin{bmatrix} I \\ J \end{bmatrix} P \begin{bmatrix} I \\ J \end{bmatrix}^\top
= \begin{bmatrix} P & P J^\top \\ J P & J P J^\top \end{bmatrix}
```

Here ω̂ is the bias-corrected body rate and v̂ the world velocity at the clone time. Without time-offset estimation, J\_W = 0 and the clone is a pure copy. The clone's first estimate (Section 7) is the propagated IMU pose at this instant, R̂\_c\* = R̂\_{k|k−1} and p̂\_c\* = p̂\_{k|k−1}, stored alongside the clone and never updated.

## 6. Measurement model, Jacobians and update

**Model.** Feature j, with world position p\_f, is observed in clone i with pixel noise n\_ij:

```latex
{}^I p_f = R_i^\top (p_f - p_i), \qquad {}^C p_f = R_{IC}^\top({}^I p_f - p_{IC}), \qquad z_{ij} = \pi({}^C p_f;\ \zeta) + n_{ij}
```

π is the camera model (pinhole with Brown-Conrady, KB4, Double Sphere or EUCM) with intrinsics ζ. J\_π = ∂π/∂ᶜp\_f is its 2×3 Jacobian.

**Triangulation.** Estimate p̂\_f from all observations of the track by Gauss-Newton on anchored inverse depth, using the current clone estimates. Reject the track for low parallax, poor conditioning, or negative depth in any view.

**Residual.** Use current estimates throughout: r\_ij = z\_ij − π(ᶜp̂\_f).

**Jacobians.** Perturb the model with the error definitions of Section 1. Every block shares the factor J\_π R\_ICᵀ:

```latex
\begin{aligned}
H_{\theta_i} &= J_\pi R_{IC}^\top\lfloor{}^I\hat p_f\times\rfloor, &
H_{p_i} &= -J_\pi R_{IC}^\top\hat R_i^\top, &
H_{f} &= J_\pi R_{IC}^\top\hat R_i^\top, \\
H_{\phi} &= J_\pi\lfloor{}^C\hat p_f\times\rfloor, &
H_{p_{IC}} &= -J_\pi R_{IC}^\top, &
H_{\zeta} &= \partial\pi/\partial\zeta
\end{aligned}
```

FEJ rule: R̂\_i, p̂\_i and ⁱp̂\_f inside these blocks use the clone's first estimates R̂\_i\*, p̂\_i\*, together with the one triangulated p̂\_f. Section 7 shows why.

**Nullspace projection.** Stack the 2M\_j rows of feature j: r\_j = H\_{x,j} x̃ + H\_{f,j} δp\_f + n\_j. H\_f is 2M\_j×3 with rank 3. Let A be an orthonormal basis of its left nullspace, so AᵀH\_f = 0:

```latex
r_{o,j} = A^\top r_j = A^\top H_{x,j}\,\tilde x + A^\top n_j = H_{o,j}\,\tilde x + n_{o,j}, \qquad \mathrm{Cov}(n_{o,j}) = \sigma_{px}^2 I_{2M_j - 3}
```

Compute Aᵀ implicitly: apply Givens rotations that zero H\_f below its top 3 rows to \[H\_f | H\_x | r\] together, then drop those 3 rows. The same first-estimate H\_f must be used here and in H\_x.

**Gating.** Accept feature j only if

```latex
r_{o,j}^\top\big(H_{o,j} P H_{o,j}^\top + \sigma_{px}^2 I\big)^{-1} r_{o,j} < \chi^2_{0.95}(2M_j - 3)
```

**Compression and update.** Stack all accepted features into r\_o = H\_o x̃ + n\_o. If H\_o has more rows than columns, take a thin QR, H\_o = Q\_1 T\_H, and set r\_n = Q\_1ᵀ r\_o, H = T\_H; otherwise H = H\_o and r\_n = r\_o. Then:

```latex
S = H P H^\top + \sigma_{px}^2 I, \qquad K = P H^\top S^{-1}, \qquad \delta\hat x = K r_n, \qquad P \leftarrow (I - KH)P(I - KH)^\top + \sigma_{px}^2 K K^\top
```

**Injection.** Apply the correction to the nominal state. First-estimate copies are not touched.

```latex
\hat R \leftarrow \hat R\,\mathrm{Exp}(\delta\hat\theta), \quad \hat R_i \leftarrow \hat R_i\,\mathrm{Exp}(\delta\hat\theta_i), \quad \hat R_{IC} \leftarrow \hat R_{IC}\,\mathrm{Exp}(\delta\hat\phi), \quad \text{all other states } x \leftarrow x + \delta\hat x
```

**Reset (optional).** After injecting, the error is redefined around the new nominal value. To first order each rotation block of P is mapped by Γ = I − ⌊½δθ̂×⌋, so P ← Γ P Γᵀ with Γ block-diagonal (Solà 2017, Section 6.3). With small corrections Γ ≈ I, which is what the MSCKF papers use.

**Marginalization.** When the window is full, first use every track that touches the oldest clone, then delete that clone's rows and columns from P and drop its first-estimate copy.

### Anchored inverse-depth features

The features are parameterized by a unit bearing b in an anchor camera a and an inverse distance ρ, as in Civera or Basalt. For MSCKF features this gives the same update as the global form to first order. It changes how the Jacobians are computed, and it keeps low-parallax features usable.

**Why the update is unchanged.** The world point is p\_f = g(b, ρ, anchor pose), and ∂p\_f/∂(b, ρ) is an invertible 3×3 matrix for ρ > 0. So the anchored H\_f is the global H\_f times an invertible matrix, with the same left nullspace A. The anchor-pose terms this adds to H\_x have the form H\_f · (∂p\_f/∂anchor), and Aᵀ removes them. H\_o and r\_o are the same as in the global form.

**Homogeneous measurement.** All four camera models (pinhole with Brown-Conrady, KB4, Double Sphere, EUCM) depend only on the direction of the point, so π(λx) = π(x) for λ > 0. Multiplying the camera-frame point by ρ removes every division by depth:

```latex
w_i = \rho\,{}^{C_i}p_f = R_{IC}^\top\Big(R_i^\top\big(R_a R_{IC}\,b + \rho\,(R_a p_{IC} + p_a - p_i)\big) - \rho\,p_{IC}\Big), \qquad z_i = \pi(w_i) + n_i
```

Here (R\_a, p\_a) is the anchor IMU clone and (R\_i, p\_i) the observing clone. At ρ = 0, a point at infinity, the model stays finite and carries pure rotation information. Because it works on bearings, it holds beyond 180°.

**Feature error.** b lives on the unit sphere and has 2 degrees of freedom. With B a 3×2 orthonormal basis of the tangent plane at b̂:

```latex
b = \frac{\hat b + B\,\delta\beta}{\|\hat b + B\,\delta\beta\|}, \qquad \rho = \hat\rho + \delta\rho, \qquad \tilde f = \big[\delta\beta^\top\ \ \delta\rho\big]^\top \in \mathbb R^3
```

Keep b as an unknown. Fixing it to the anchor's measured pixel would turn the anchor observation into a hard constraint and bias the estimate.

**Jacobians.** Define y = ρ(p\_f − p\_i), the scaled world-frame vector from the observing clone to the point, y = R\_a R\_IC b + ρ(R\_a p\_IC + p\_a − p\_i), so that w\_i = R\_ICᵀ(R\_iᵀ y − ρ p\_IC). Each block below is multiplied on the left by J\_π(w\_i):

```latex
\begin{aligned}
\frac{\partial w_i}{\partial\delta\beta} &= R_{IC}^\top R_i^\top R_a R_{IC}\,B, &
\frac{\partial w_i}{\partial\delta\rho} &= R_{IC}^\top\big(R_i^\top(R_a p_{IC} + p_a - p_i) - p_{IC}\big), \\
\frac{\partial w_i}{\partial\delta\theta_i} &= R_{IC}^\top\lfloor(\hat R_i^\top y)\times\rfloor, &
\frac{\partial w_i}{\partial\delta p_i} &= -\rho\,R_{IC}^\top\hat R_i^\top, \\
\frac{\partial w_i}{\partial\delta\theta_a} &= -R_{IC}^\top\hat R_i^\top\hat R_a\lfloor(R_{IC}b + \rho\,p_{IC})\times\rfloor, &
\frac{\partial w_i}{\partial\delta p_a} &= \rho\,R_{IC}^\top\hat R_i^\top
\end{aligned}
```

The extrinsic and intrinsic blocks follow by the same chain rule. In the anchor's own view (i = a) the pose terms cancel and w\_a = b, so that row depends only on δβ: ∂w\_a/∂δβ = B. The anchor clone receives blocks from every observation of the feature, not only its own, so H\_x is denser. The result after projection is unchanged.

**Fixed anchor frame (the simpler variant, used in the code).** Take the anchor camera pose T\_GA at triangulation time as a constant, not as a function of the anchor clone. Then (β, ρ) are coordinates of a fixed world point p\_f = T\_GA·\[b/ρ\], an invertible reparameterization of the global point for ρ > 0, and no anchor-clone blocks exist: every observation, the anchor's own included, has only the observing-clone, extrinsic and feature blocks above. ∂p\_f/∂(β, ρ) is invertible, so H\_f has the same left nullspace as the global H\_f and H\_o is identical to the global and to the moving-anchor forms. The observability rows of N for the feature are ∂(β, ρ)/∂p\_f applied to the global rows d and −⌊p̂\_f×⌋g. Either variant is correct; this one has fewer blocks and no anchor FEJ.

**Rank-adaptive nullspace projection.** The δρ column of H\_f is J\_π R\_ICᵀ(R\_iᵀ(R\_a p\_IC + p\_a − p\_i) − p\_IC), the baseline from the anchor camera to camera i, and it does not depend on ρ. It does not vanish at ρ = 0: a point at infinity seen with baseline still has an observable inverse distance, with sensitivity f·|baseline| pixels per unit of ρ. It vanishes only when every baseline is zero or parallel to the bearing, that is, under pure rotation. Then H\_f drops to rank 2 and the δρ direction carries no information. So use a column-pivoted QR, or Givens rotations with a rank tolerance, on \[H\_f | H\_x | r\], and drop only rank(H\_f) rows. Rotation-only tracks then keep 2M − 2 residuals that constrain the clone rotations instead of being rejected, and the δρ row they would have lost, which under zero baseline is pure inconsistency, is kept where the χ² gate can see it. Use the matching degrees of freedom in the gate: 2M − rank(H\_f). A relative rank tolerance (1e-9 of the largest pivot) triggers only for an exactly zero baseline, which is the intended case; low but nonzero parallax keeps rank 3.

**Triangulation.** Gauss-Newton directly on (β, ρ), anchored at the first clone of the track, or the middle one for better conditioning. Initialize ρ from two views, allow ρ ≈ 0, and reject ρ < 0.

**FEJ.** With a moving anchor, the anchor clone appears in every observation's blocks, so its first estimate (R̂\_a\*, p̂\_a\*) is used there, as for the observing clones. The anchored parameters are then relative to the body, so their rows of N are zero, and condition (b) of Section 7 holds exactly whenever the same estimates are used in all blocks: substituting the blocks above into H\_θi R̂\_i\*ᵀg − H\_pi⌊p̂\_i\*×⌋g + H\_θa R̂\_a\*ᵀg − H\_pa⌊p̂\_a\*×⌋g gives J\_π R\_ICᵀ R̂\_i\*ᵀ⌊(y + ρp̂\_i\* − R̂\_a\*R\_IC b − ρR̂\_a\*p\_IC − ρp̂\_a\*)×⌋g = 0 by the definition of y. With the fixed anchor frame there is no anchor clone to freeze: the feature is a fixed global point and the global rule of Section 7.6 applies as written.

**Anchor changes (v2 only).** MSCKF features are used once and never change anchor. SLAM features kept in the state need a new anchor whenever their anchor clone is marginalized, with their covariance transformed through the change of anchor (sqrtVINS, Section IV-C).

## 7. Observability and FEJ (MSCKF 2.0, Section VI, in our convention)

The result: the linearized filter keeps the correct four unobservable directions only if every Jacobian uses one fixed linearization point per state. With a body-frame error the fixed point must include the attitude, and Φ\_θθ must be built from the endpoint attitudes. Section 7.9 gives the price of that choice.

### 7.1 The observability matrix

For the analysis, add the features to the state, as in EKF-SLAM. The MSCKF is equivalent to it after marginalizing the features. Measurements at times 1, 2, … depend on the error at time 1 through

```latex
\mathcal O = \begin{bmatrix} H_1 \\ H_2\,\Phi_{2,1} \\ H_3\,\Phi_{3,1} \\ \vdots \end{bmatrix}, \qquad \Phi_{k,1} = \Phi_{k-1}\cdots\Phi_1
```

A direction N is unobservable if 𝒪 N = 0. The data then cannot distinguish x̃ from x̃ + Nα.

### 7.2 The unobservable directions

The physical system does not change if the whole world is shifted by d, or rotated by an angle α about gravity. Express both motions in our error coordinates.

Translation: δp = δp\_i = δp\_f = d, everything else 0.

Rotation about gravity: R' = Exp(αĝ)R̂ = R̂ Exp(αR̂ᵀĝ), so δθ = αR̂ᵀĝ. Rotating v, p and p\_f gives δv = αĝ × v̂ = −α⌊v̂×⌋ĝ, and likewise for p and p\_f. Biases, extrinsics and t\_d are body-fixed, so they do not change. Scaling ĝ to g:

```latex
N_k = \begin{bmatrix}
0 & \hat R_k^\top g \\
0 & 0 \\
0 & -\lfloor\hat v_k\times\rfloor g \\
0 & 0 \\
I & -\lfloor\hat p_k\times\rfloor g \\
0 & 0 \\
I & \hat R_i^\top g \;\big|\; -\lfloor\hat p_i\times\rfloor g \\
I & -\lfloor\hat p_f\times\rfloor g
\end{bmatrix}
\quad
\begin{matrix} \delta\theta \\ \delta b_g \\ \delta v \\ \delta b_a \\ \delta p \\ \tilde x_W \\ (\delta\theta_i,\ \delta p_i) \\ \delta p_f \end{matrix}
```

The clone rows read: δθ\_i gets 0 and R̂\_iᵀg, δp\_i gets I and −⌊p̂\_i×⌋g. The difference from MSCKF 2.0: its global-frame error gives the constant g in the attitude rows. Ours gives R̂ᵀg, which depends on the attitude estimate.

### 7.3 The two conditions

𝒪 N\_1 = 0 for every sequence of measurements if and only if, at every step,

```latex
\text{(a)}\quad \Phi_k N_k = N_{k+1}, \qquad\qquad \text{(b)}\quad H_k N_k = 0
```

Then H\_k Φ\_{k,1} N\_1 = H\_k N\_k = 0 for every k. Each condition holds only if the estimates inside Φ\_k and H\_k equal the estimates inside N\_k.

### 7.4 Proof of (a) for propagation

Translation column: only the δp row of N is nonzero, and Φ\_pp = I. Holds.

Rotation column, attitude row. With Φ\_θθ = R̂\_{k+1}ᵀR̂\_k:

```latex
\Phi_{\theta\theta}\,\hat R_k^\top g = \hat R_{k+1}^\top\hat R_k\hat R_k^\top g = \hat R_{k+1}^\top g \qquad \checkmark
```

Rotation column, velocity row. Use (a − b) × g with g × g = 0:

```latex
-\lfloor(\hat v_{k+1} - \hat v_k - g\Delta t)\times\rfloor\hat R_k\hat R_k^\top g - \lfloor\hat v_k\times\rfloor g
= -\lfloor\hat v_{k+1}\times\rfloor g + \lfloor\hat v_k\times\rfloor g - \lfloor\hat v_k\times\rfloor g = -\lfloor\hat v_{k+1}\times\rfloor g \qquad \checkmark
```

Rotation column, position row:

```latex
-\lfloor(\hat p_{k+1} - \hat p_k - \hat v_k\Delta t - \tfrac12 g\Delta t^2)\times\rfloor g - \Delta t\lfloor\hat v_k\times\rfloor g - \lfloor\hat p_k\times\rfloor g = -\lfloor\hat p_{k+1}\times\rfloor g \qquad \checkmark
```

Bias columns of Φ multiply zero rows of N, so their linearization points never matter. Clone, calibration and feature rows are unchanged by propagation and by Φ, so they hold trivially.

Each check uses the same R̂\_k, v̂\_k, p̂\_k, v̂\_{k+1}, p̂\_{k+1} and R̂\_{k+1} inside Φ\_k as inside N\_k and N\_{k+1}.

### 7.5 Proof of (a) for cloning

The clone copies the IMU pose. Its N rows equal the IMU attitude and position rows at that instant, and the t\_d column of J multiplies the zero t\_d row of N. So \[I; J\]N\_k is the augmented N\_k. Holds, provided the clone's stored first estimate equals the IMU estimate used in N\_k at that time.

### 7.6 Proof of (b) for the measurement

Translation column: H\_{p\_i} + H\_f = −J\_πR\_ICᵀR̂\_iᵀ + J\_πR\_ICᵀR̂\_iᵀ = 0. Holds.

Rotation column. Use ⌊ⁱp̂\_f×⌋R̂\_iᵀg = R̂\_iᵀ⌊(R̂\_iⁱp̂\_f)×⌋g and R̂\_iⁱp̂\_f = p̂\_f − p̂\_i:

```latex
H_{\theta_i}\hat R_i^\top g - H_{p_i}\lfloor\hat p_i\times\rfloor g - H_f\lfloor\hat p_f\times\rfloor g
= J_\pi R_{IC}^\top\hat R_i^\top\big[\lfloor(\hat p_f - \hat p_i)\times\rfloor + \lfloor\hat p_i\times\rfloor - \lfloor\hat p_f\times\rfloor\big]g = 0 \qquad \checkmark
```

The extrinsic, intrinsic and t\_d blocks of H multiply zero rows of N. This holds only if ⁱp̂\_f is computed from the same R̂\_i, p̂\_i and p̂\_f that appear in N.

**The nullspace projection preserves it.** H\_x N\_x = −H\_f N\_f, so H\_o N\_x = AᵀH\_x N\_x = −AᵀH\_f N\_f = 0, provided the same H\_f builds A and H\_x.

### 7.7 What breaks without FEJ (the ΔΓ terms)

A standard EKF builds H\_k at the propagated estimate x̂\_{k|k−1}, so (b) holds with N\_k evaluated there. It then builds Φ\_k at the updated estimate x̂\_{k|k}. Write δx̂\_k = x̂\_{k|k} − x̂\_{k|k−1} for the correction. Condition (a) then fails by terms proportional to that correction. For the attitude row, with Φ\_θθ = ΔRᵀ and R̂\_{k+1|k} = R̂\_{k|k}ΔR:

```latex
\Phi_{\theta\theta}N_k^{\theta} - N_{k+1}^{\theta} = \Delta R^\top\big(\hat R_{k|k-1} - \hat R_{k|k}\big)^\top g \neq 0
```

The velocity and position rows leave similar residuals in δv̂\_k and δp̂\_k. These are MSCKF 2.0's ΔΓ terms. They make the rotation column observable: rank(𝒪) grows by one, and the filter gains information about yaw that the sensors never provided. The yaw covariance then shrinks without cause, and NEES grows over time.

### 7.8 The FEJ fix

Evaluate every Jacobian at first estimates, marked with a star:

```latex
\Phi_k^\star:\quad \Phi_{\theta\theta} = \hat R_{k+1|k}^\top\hat R_{k|k-1},\quad \hat R_k,\ \hat v_k,\ \hat p_k \to \hat x_{k|k-1},\quad \hat v_{k+1},\ \hat p_{k+1} \to \hat x_{k+1|k}
```

```latex
H^\star:\quad \hat R_i,\ \hat p_i \to \hat R_i^\star,\ \hat p_i^\star \ \text{(clone at creation)},\qquad {}^I\hat p_f = \hat R_i^{\star\top}(\hat p_f - \hat p_i^\star)
```

Then N\_k is evaluated at the first estimates everywhere, and the checks in 7.4 to 7.6 hold exactly. Between images no update happens, so x̂\_{k|k} = x̂\_{k|k−1} and Φ\* equals the ordinary Φ. FEJ changes only the first step after each update. This is the body-frame counterpart of MSCKF 2.0's eqs. (54)–(56). Theirs freezes only position and velocity, because their attitude row of N is constant. Ours must also freeze attitude, through Φ\_θθ and ⁱp̂\_f.

### 7.9 The cost of body-frame FEJ, and when it bites

Freezing the attitude is not free. Three terms appear that MSCKF 2.0's global error does not have:

- **Φ\_θθ\*** = R̂\_{k+1|k}ᵀR̂\_{k|k−1} = ΔRᵀ Exp(−δθ̂\_k): the attitude covariance and its cross terms are rotated by the correction δθ̂\_k on every step after an update. In global coordinates this Φ\_θθ\* is exactly the identity, so the body-frame FEJ filter is MSCKF 2.0's filter written in the frozen body frame; the mismatch is only that clone corrections are injected in the current frame. Second order in the corrections, and measured to cost nothing (Section 8, test 7).
- **R̂\_i\* inside H\_f** and inside the point at which J\_π is evaluated. The nullspace basis A is built from H\_f\*, but the residual's true sensitivity to the feature error is H\_f at the current estimate, so the projection leaks Aᵀ(H\_f − H\_f\*)δp\_f, proportional to the clone's correction since cloning times the feature's own error. MSCKF 2.0 keeps the current R̂\_i in H\_f and has no such term.
- **Stale pose Jacobians in low-translation motion.** H\_θi and the δρ column scale with the baseline between clones, not with the feature distance. A first-estimate gap of millimetres is negligible against a 5 m feature but not against a 10 cm window baseline. Fast rotation, hovering, stops and slow carts are this regime.

All three are proportional to the corrections applied after cloning, so they are large exactly when the filter is still converging: in the first seconds after initialization, while biases, velocity, extrinsics and t\_d are being pulled in, clone gaps reach degrees and centimetres. Afterwards they are of order 0.01° and millimetres and the terms are invisible. The damage done during the transient does not heal: global position and yaw are unobservable, so an error committed in the first seconds is carried for the whole run, and the position NEES stays high even though every later step is consistent. Measured on the fast-rotation simulation (50 runs × 60 s): FEJ in Φ alone costs nothing; FEJ in H carries the whole excess (position NEES 6.5 against 4.0 without FEJ); clones created in the first 5 s using current estimates and all later clones frozen gives 4.0; initial errors scaled by 0.2 gives 3.4 against 3.2. The cart, which starts stopped, diverges inside 3 s with FEJ for the same reason.

**Rule.** Freeze a clone only once the filter has converged: start FEJ when the bias, extrinsic and t\_d standard deviations have dropped below thresholds (for example 3× their steady-state values), or after a fixed warm-up, and use current estimates for the clones created before that. Spurious yaw information from the unfrozen start is negligible over the warm-up, while the stale-Jacobian error of a frozen transient is not. A better initial covariance from the initializer reduces the problem at the source. The right-invariant error of Stage 3 removes it: condition (a) holds there at the current estimates, with no frozen point and no leak.

## 8. FEJ implementation rules and tests

FEJ is bookkeeping: a frozen copy of a few states, used only where Jacobians are built. The nominal state and P are updated exactly as without FEJ.

| Quantity | In Φ and H use | Why |
| --- | --- | --- |
| IMU R̂, v̂, p̂ at the start of an interval | Propagated value x̂\_{k\|k−1}, saved before the update at k | They appear in N |
| IMU R̂, v̂, p̂ at the end of an interval | Propagated value x̂\_{k+1\|k} | They appear in N |
| Φ\_θθ | R̂\*\_{k+1}ᵀ R̂\*\_k, not ΔRᵀ, on the first step after an update | Body-frame attitude row of N is R̂ᵀg |
| Clone R̂\_i, p̂\_i | Value at cloning, stored with the clone, never updated | Clones appear in many updates |
| Feature p̂\_f (MSCKF track) | One triangulated value from current clones, used in every block of that feature | Used once, so any single value is consistent |
| Residual r = z − h(x̂) | Current estimates | It measures the actual error |
| Biases, ω̂, â, bias Jacobians, Q | Current estimates | Zero rows in N |
| Extrinsics, intrinsics, t\_d | Current estimates, the same value across one feature's blocks | Zero rows in N |
| SLAM features in the state (v2) | First estimate at initialization, like clones | They appear in many updates |
| Clones created before convergence | Current estimates, as without FEJ, until the warm-up gate of Section 7.9 opens | Stale transient Jacobians cost more than the spurious yaw information they prevent |

**Storage.** Each clone holds (R̂\*, p̂\*) next to (R̂, p̂). The IMU state keeps the propagated (R̂, v̂, p̂) from just before the latest update. One config flag switches every Jacobian between first and current estimates, so FEJ can be turned off for comparison. A second setting, the warm-up gate of Section 7.9, decides from which clone on the first estimates are used; each clone records whether it is frozen.

**Unit tests, in simulation with exact ground truth:**

1. Φ and H blocks against central finite differences of the nonlinear propagation and measurement functions: relative error below 1e-6.
2. Single-sample Φ against I + Fδt: agreement to O(δt²).
3. Nullspace conditions per step, with FEJ on: ‖Φ\_k N\_k − N\_{k+1}‖ and ‖H\_k N\_k‖ at machine precision.
4. Observability matrix over 20–50 steps with features in the state: numerical rank equal to dimension minus 4 with FEJ, minus 3 without.
5. Consistency: average NEES of attitude and position over 50 Monte Carlo runs, FEJ on vs off, judged per epoch (the per-epoch Monte Carlo average inside the band in at least 90 % of epochs), not by the run-long mean. With FEJ it should stay inside the χ² 95% band. Without it, the yaw NEES should grow over time.
6. Yaw covariance under a static or straight-line start: with FEJ it must not shrink below its initial value, because yaw is unobservable.
7. Transient cost of FEJ (Section 7.9): from a perturbed start, run FEJ in Φ only, FEJ in H only, full FEJ, and full FEJ with the warm-up gate, against no FEJ. FEJ in Φ alone must match no FEJ; the gated variant must match no FEJ on short runs and beat it on yaw NEES on long ones. Log, per observation, the gap between each clone's first and current estimate (rotation in degrees, position in metres and as a fraction of the window baseline) to see when the gate should open.

The same tests apply unchanged to the right-invariant variant planned in Stage 3. There, condition (a) should hold without any frozen values, and test 3 is the direct comparison.

References: Li & Mourikis, IJRR 2013 (MSCKF 2.0, Sections VI–VII); Mourikis & Roumeliotis, ICRA 2007; Solà, arXiv 1711.02508 (2017); Forster et al., T-RO 2017; Geneva et al., ICRA 2020.
