"""Frozen Acquisition-State Telemetry equations.

This module is intentionally substrate-independent. A binding supplies a bounded
per-schema scorer and a unit-normalised evidence representation; the monitor
computes the Bayesian Progress Estimator (PE^B) and geometric Stalling Index
(SI^perp) without inspecting the raw evidence modality.
"""

import numpy as np
import math

# ── Optional GPU acceleration ──────────────────────────────────────────────────
# torch is used only inside _compute_orthogonal_novelty to off-load the
# SVD-based orthogonal projection from the CPU.  If torch / CUDA is absent
# the code falls back transparently to the original NumPy lstsq path.
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore
    _TORCH_AVAILABLE = False


class GeometricTelemetryTool:
    """Frozen geometric and Bayesian telemetry computations for AST.

    Implements:
      - compute_bayesian_PE       — Bayesian Progress Estimator (PE^B)
      - compute_geometric_subspace_SI  — Geometric Subspace Stalling Index (SI^⊥_geom)

    Both signals are described in:
      the accompanying Acquisition-State Telemetry paper. The equations are
      substrate-independent; substrate bindings supply only scores and embeddings.
    """

    # ── GPU device (resolved once at class definition time) ────────────────
    # Points to the best available CUDA device, or None if running CPU-only.
    # All GPU paths in this class read from this single attribute so that
    # device selection is centralised and easy to override if needed.
    _torch_device: object = (
        _torch.device("cuda") if (_TORCH_AVAILABLE and _torch.cuda.is_available()) else None
    )

    # ══════════════════════════════════════════════════════════════════════
    #  BAYESIAN PROGRESS ESTIMATOR  (PE^B)
    # ══════════════════════════════════════════════════════════════════════
    #
    #  For each category i, we maintain a Beta(a_i, b_i) posterior over
    #  θ_i — the probability that the next query yields an informative
    #  response (Δυ_i > 0).
    #
    #  Prior:  Beta(1, 1) = Uniform
    #  Update: k_i successes in m_i queries → Beta(1+k_i, 1+m_i−k_i)
    #
    #  PE^B_i(t) = [ α · E[θ_i] · H_b(υ_i) · ε_i(t)
    #               + (1−α) · ξ_i(t) ] · w_i · g_i(t)
    #
    #  where:
    #    E[θ_i]   = a_i / (a_i + b_i)                — posterior mean
    #    σ²_i     = a_i·b_i / [(a_i+b_i)²(a_i+b_i+1)] — posterior variance
    #    H_b(υ_i) = binary entropy of completeness estimate
    #    ε_i(t)   = 1 + β · σ²_i / max_j σ²_j         — epistemic bonus, β=0.5
    #    ξ_i(t)   = trajectory exploration rate         — see _compute_exploration_rate
    #    w_i      = domain importance weight
    #    g_i(t)   = dependency gate = Π_{j∈dep(i)} E[θ_j]
    #
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def compute_bayesian_PE(state, dimension_importance, dependencies=None,
                            return_components=False):
        """Compute PE^B: Bayesian Progress Estimator (Eq. pe_bayesian).

        Args:
            state: HybridKnowledgeState with .values, .dimensions,
                   .dimension_mentions, .dimension_successes, .embeddings,
                   .embedding_history
            dimension_importance: dict {dim: w_i} importance weights
            dependencies: dict {dim: [prerequisite dims]} or None
            return_components: if True, return (pe, {'entropy': ..., 'semantic': ...})

        Returns:
            dict {dim: PE_value}
        """
        pe = {}
        eps = 1e-9
        alpha = 0.5   # balance: Bayesian EIG vs. trajectory exploration (Eq. pe_bayesian)
        beta  = 0.5   # epistemic bonus scale (Eq. epistemic_bonus)

        if return_components:
            entropy_comp  = {}
            semantic_comp = {}

        # ── Pass 1: compute max posterior variance for normalisation ───────
        max_posterior_var = eps
        posterior_params  = {}

        for dim in state.dimensions:
            m_i = int(state.dimension_mentions.get(dim, 0))
            k_i = int(getattr(state, 'dimension_successes', {}).get(dim, 0))

            # Beta posterior parameters: Beta(1+k_i, 1+m_i−k_i)  (Eq. beta_posterior)
            a_i = 1.0 + k_i
            b_i = 1.0 + (m_i - k_i)

            # Posterior variance  σ²_i  (Eq. beta_var)
            var_i = (a_i * b_i) / ((a_i + b_i) ** 2 * (a_i + b_i + 1.0))
            max_posterior_var = max(max_posterior_var, var_i)

            posterior_params[dim] = (a_i, b_i, var_i)

        # ── Pass 2: compute PE^B for each category ─────────────────────────
        for dim in state.dimensions:
            a_i, b_i, var_i = posterior_params[dim]

            # (1) Posterior mean E[θ_i]  (Eq. theta_mean)
            theta_mean = a_i / (a_i + b_i)

            # (2) Binary entropy H_b(υ_i)  (Eq. binary_entropy)
            v_i = float(state.values.get(dim, 0.0))
            v_i = min(max(v_i, eps), 1.0 - eps)
            H_b = -(v_i * math.log2(v_i) + (1.0 - v_i) * math.log2(1.0 - v_i))

            # (3) Epistemic bonus ε_i = 1 + β · σ²_i / max_j σ²_j  (Eq. epistemic_bonus)
            epistemic_bonus = 1.0 + beta * (var_i / max_posterior_var)  # ∈ [1, 1+β]

            # (4) Bayesian EIG term: E[θ_i] · H_b · ε_i
            bayesian_eig = theta_mean * H_b * epistemic_bonus

            # (5) Trajectory exploration rate ξ_i(t)  (Eq. exploration_rate)
            xi_i = GeometricTelemetryTool._compute_exploration_rate(state, dim)

            # (6) Combined PE^B  (Eq. pe_bayesian)
            combined = alpha * bayesian_eig + (1.0 - alpha) * xi_i

            # (7) Importance weight w_i
            w_i = dimension_importance.get(dim, 1.0)
            weighted = combined * w_i

            # (8) Dependency gate g_i(t) = Π_{j∈dep(i)} E[θ_j]  (Eq. gate)
            dep_gate = 1.0
            if dependencies and dim in dependencies:
                for dep_dim in dependencies[dim]:
                    a_j = 1.0 + getattr(state, 'dimension_successes', {}).get(dep_dim, 0)
                    b_j = 1.0 + (
                        state.dimension_mentions.get(dep_dim, 0)
                        - getattr(state, 'dimension_successes', {}).get(dep_dim, 0)
                    )
                    dep_gate *= a_j / (a_j + b_j)   # E[θ_j] posterior mean

            weighted *= dep_gate
            pe[dim] = float(weighted)

            if return_components:
                entropy_comp[dim]  = float(alpha       * bayesian_eig * w_i * dep_gate)
                semantic_comp[dim] = float((1.0 - alpha) * xi_i        * w_i * dep_gate)

        if return_components:
            return pe, {'entropy': entropy_comp, 'semantic': semantic_comp}
        return pe


    @staticmethod
    def _compute_exploration_rate(state, dim):
        """Trajectory exploration rate ξ_i(t)  (Eq. exploration_rate).

        Measures whether the gain-weighted PE-track trajectory is still
        advancing into new semantic territory.

        Returns:
            Float in [0, 1].
            1 = high residual exploration potential (no data, or looping trajectory)
            0 = well and productively explored (efficient straight-line progress)

        The gain-confidence gate γ_i = min(1, υ_i / υ*) ensures that a
        dimension with low cumulative gain retains high exploration potential
        regardless of trajectory shape — the trajectory is unreliable as a
        signal until enough has been learned.  (Eq. gain_confidence)
        """
        eps            = 1e-12
        GAIN_THRESHOLD = 0.5    # υ* — completeness at which gain_confidence reaches 1.0

        # γ_i(t) = min(1, υ_i / υ*)  (Eq. gain_confidence)
        cumulative_gain = float(state.values.get(dim, 0.0))
        gain_confidence = min(1.0, cumulative_gain / GAIN_THRESHOLD)

        # No PE-track history yet → maximum residual exploration potential
        if dim not in state.embedding_history or len(state.embedding_history[dim]) < 1:
            return 1.0

        # Build trajectory from PE-track snapshots + current embedding
        traj_points = [entry['embedding'] for entry in state.embedding_history[dim]]
        traj_points.append(state.embeddings[dim])

        if len(traj_points) < 2:
            return 1.0

        # Compute full path length (used for diagnostics; not the primary signal)
        path_length = 0.0
        for j in range(1, len(traj_points)):
            step = (np.asarray(traj_points[j],   dtype=np.float64) -
                    np.asarray(traj_points[j-1], dtype=np.float64))
            path_length += np.linalg.norm(step)

        if path_length < eps:
            return 1.0   # no movement → full potential

        # Recent window of n=4 points for recency sensitivity  (calibration: n=4)
        recent_n   = min(4, len(traj_points))
        recent_pts = traj_points[-recent_n:]

        if len(recent_pts) < 2:
            return 0.5   # neutral default

        # Recent path length L_i and net displacement Δ_i  (Eq. geodesic)
        recent_path = 0.0
        for j in range(1, len(recent_pts)):
            step = (np.asarray(recent_pts[j],   dtype=np.float64) -
                    np.asarray(recent_pts[j-1], dtype=np.float64))
            recent_path += np.linalg.norm(step)

        recent_disp = np.linalg.norm(
            np.asarray(recent_pts[-1], dtype=np.float64) -
            np.asarray(recent_pts[0],  dtype=np.float64)
        )

        if recent_path < eps:
            # Stationary trajectory → high remaining potential
            return 0.8

        # Raw geodesic efficiency ξ_raw = Δ_i / (L_i + ε)  (Eq. geodesic)
        recent_xi = recent_disp / (recent_path + eps)

        # ξ_i = 1 − ξ_raw · γ_i  (Eq. exploration_rate)
        adjusted_xi = recent_xi * gain_confidence
        return 1.0 - adjusted_xi


    # ══════════════════════════════════════════════════════════════════════
    #  DIAGNOSTIC: Per-category posterior summary
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def get_posterior_diagnostics(state):
        """Return per-category Beta posterior parameters and summary statistics.

        Returns:
            dict {dim: {alpha, beta, mean, variance, ci_95_low, ci_95_high,
                        attempts, successes}}
        """
        diagnostics = {}
        for dim in state.dimensions:
            m_i = int(state.dimension_mentions.get(dim, 0))
            k_i = int(getattr(state, 'dimension_successes', {}).get(dim, 0))

            a_i = 1.0 + k_i
            b_i = 1.0 + (m_i - k_i)

            mean = a_i / (a_i + b_i)
            var  = (a_i * b_i) / ((a_i + b_i) ** 2 * (a_i + b_i + 1.0))

            # 95% credible interval via Beta quantiles
            try:
                from scipy.stats import beta as beta_dist
                ci_low  = beta_dist.ppf(0.025, a_i, b_i)
                ci_high = beta_dist.ppf(0.975, a_i, b_i)
            except ImportError:
                std     = math.sqrt(var)
                ci_low  = max(0.0, mean - 1.96 * std)
                ci_high = min(1.0, mean + 1.96 * std)

            diagnostics[dim] = {
                'alpha':      a_i,
                'beta':       b_i,
                'mean':       mean,
                'variance':   var,
                'ci_95_low':  ci_low,
                'ci_95_high': ci_high,
                'attempts':   m_i,
                'successes':  k_i,
            }

        return diagnostics


    @staticmethod
    def get_trajectory_diagnostics(state):
        """Return per-category trajectory geometry statistics.

        Returns:
            dict {dim: {path_length, displacement, tortuosity,
                        exploration_rate, n_points}}
        """
        diagnostics = {}
        eps = 1e-12

        for dim in state.dimensions:
            if dim not in state.embedding_history or len(state.embedding_history[dim]) < 1:
                diagnostics[dim] = {
                    'path_length':    0.0,
                    'displacement':   0.0,
                    'tortuosity':     1.0,
                    'exploration_rate': 1.0,
                    'n_points':       0,
                }
                continue

            traj_points = [entry['embedding'] for entry in state.embedding_history[dim]]
            traj_points.append(state.embeddings[dim])

            path_length = 0.0
            for j in range(1, len(traj_points)):
                step = (np.asarray(traj_points[j],   dtype=np.float64) -
                        np.asarray(traj_points[j-1], dtype=np.float64))
                path_length += np.linalg.norm(step)

            displacement = np.linalg.norm(
                np.asarray(traj_points[-1], dtype=np.float64) -
                np.asarray(traj_points[0],  dtype=np.float64)
            )

            tortuosity = 1.0 if path_length < eps else path_length / (displacement + eps)
            xi = GeometricTelemetryTool._compute_exploration_rate(state, dim)

            diagnostics[dim] = {
                'path_length':      float(path_length),
                'displacement':     float(displacement),
                'tortuosity':       float(tortuosity),
                'exploration_rate': float(xi),
                'n_points':         len(traj_points),
            }

        return diagnostics


    # ══════════════════════════════════════════════════════════════════════
    #  ORTHOGONAL NOVELTY  η_i  (internal helper)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_orthogonal_novelty(state, dim, window=None):
        """Orthogonal subspace novelty η_i for a category  (Eq. eta_geom).

        Projects the most recent response embedding onto the subspace spanned
        by all prior response embeddings for dim.  η is the fraction of the
        new embedding that lies OUTSIDE that subspace.

        Reads from state.response_log (unconditional SI track) so that
        stall-turn responses are always represented.

        Args:
            state:  HybridKnowledgeState with .response_log
            dim:    Category name
            window: If set, restrict prior subspace to the last `window` entries.
                    None → full session history (recommended).

        Returns:
            Float in [0, 1]:
            1.0 → maximally novel (orthogonal to all prior responses)
            0.0 → fully redundant (lies in the span of prior responses)
        """
        eps = 1e-12

        if not hasattr(state, 'response_log'):
            return 1.0

        log = state.response_log.get(dim, [])
        if len(log) < 2:
            return 1.0   # first query — nothing to compare against

        prior_entries = log[:-1]
        if window is not None and len(prior_entries) > window:
            prior_entries = prior_entries[-window:]

        e_new    = np.asarray(log[-1]['embedding'], dtype=np.float64)
        norm_new = np.linalg.norm(e_new)

        if norm_new < eps:
            return 0.0   # zero-energy response — no information

        V_cols = [np.asarray(entry['embedding'], dtype=np.float64) for entry in prior_entries]
        V      = np.column_stack(V_cols)

        if GeometricTelemetryTool._torch_device is not None:
            # ── GPU path: rank-truncated SVD projection ───────────────────
            # Mathematically identical to lstsq: both compute the orthogonal
            # projection of e_new onto col(V).  float32 intentional (speed).
            V_t = _torch.from_numpy(V.astype(np.float32)).to(GeometricTelemetryTool._torch_device)
            e_t = _torch.from_numpy(e_new.astype(np.float32)).to(GeometricTelemetryTool._torch_device)

            U, S, _Vh = _torch.linalg.svd(V_t, full_matrices=False)
            tol       = S[0].item() * 1e-5 + 1e-12
            U_r       = U[:, S > tol]                      # keep numerical rank

            e_parallel_t = U_r @ (U_r.T @ e_t)
            e_perp_t     = e_t - e_parallel_t

            eta = _torch.linalg.norm(e_perp_t).item() / (_torch.linalg.norm(e_t).item() + eps)
        else:
            # ── CPU fallback: NumPy lstsq ─────────────────────────────────
            c, _, _, _ = np.linalg.lstsq(V, e_new, rcond=None)
            e_parallel  = V @ c
            e_perp      = e_new - e_parallel
            eta         = np.linalg.norm(e_perp) / (norm_new + eps)

        return float(min(max(eta, 0.0), 1.0))


    # ══════════════════════════════════════════════════════════════════════
    #  GEOMETRIC SUBSPACE SI  (SI^⊥_geom)
    # ══════════════════════════════════════════════════════════════════════
    #
    #  Removes the discrete question-stream counter entirely. Recurrence
    #  and redundancy are both derived from the same geometric object: the
    #  orthogonal projection of the current turn's response embedding r(t)
    #  onto each category's accumulated response subspace V_i.
    #
    #  For every category i ∈ M:
    #
    #      η_i(t) = ‖ P^⊥_i r(t) ‖ / ‖ r(t) ‖     — orthogonal novelty
    #      ρ_i(t) = 1 − η_i(t)                      — geometric membership
    #
    #  Because ρ_i = 1−η_i, membership and redundancy are the same scalar.
    #  Their product ρ_i·(1−η_i) = (1−η_i)² double-weights responses that
    #  are both strongly in a category's subspace AND add nothing new to it.
    #
    #  SI^⊥_geom(t) = Σ_i [ ρ_i·(1−η_i)·D(Δυ(t);λ,η_min)·(1−υ_i) ]
    #                 ────────────────────────────────────────────────
    #                               Σ_i ρ_i  +  ε
    #
    #  D(Δυ(t);λ,η_min) = max(η_min, 1−λ·Δυ(t))     — gain dampening
    #
    #  where Δυ(t) is the total knowledge gain at turn t, summed across
    #  all categories.  A productive turn suppresses the stall signal
    #  regardless of geometric redundancy.
    #
    #  Verified in Experiment 1:
    #    1. S1 (efficient): SI^⊥_geom stays below θ throughout.
    #    2. S2/S3 (agent repetition / interviewee degradation):
    #       SI^⊥_geom rises above θ; stall burden is elevated.
    #
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _project_probe_onto_category_subspace(state, dim, probe_embedding,
                                              window=None):
        """Project an arbitrary probe vector onto category dim's subspace.

        Unlike _compute_orthogonal_novelty (which reads r(t) from the log's
        last entry for dim), this helper accepts an external probe vector.
        Required by compute_geometric_subspace_SI, which tests the same r(t)
        against every category's subspace simultaneously.

        Args:
            state:           HybridKnowledgeState with .response_log
            dim:             Category whose prior responses span the subspace
            probe_embedding: Vector to project (numpy array, shape (d,))
            window:          If set, restrict subspace to last `window` entries

        Returns:
            η ∈ [0, 1]:
            1 = probe is orthogonal to V_dim (not in this category)
            0 = probe lies fully in span(V_dim)
            Returns 1.0 if V_dim has no entries.
        """
        eps = 1e-12

        if not hasattr(state, 'response_log'):
            return 1.0

        log = state.response_log.get(dim, [])
        if len(log) < 1:
            return 1.0   # empty subspace → no membership

        prior_entries = log
        if window is not None and len(prior_entries) > window:
            prior_entries = prior_entries[-window:]

        e_new    = np.asarray(probe_embedding, dtype=np.float64)
        norm_new = np.linalg.norm(e_new)
        if norm_new < eps:
            return 1.0   # degenerate probe — treat as non-member

        V_cols = [np.asarray(entry['embedding'], dtype=np.float64) for entry in prior_entries]
        V      = np.column_stack(V_cols)

        if GeometricTelemetryTool._torch_device is not None:
            V_t = _torch.from_numpy(V.astype(np.float32)).to(GeometricTelemetryTool._torch_device)
            e_t = _torch.from_numpy(e_new.astype(np.float32)).to(GeometricTelemetryTool._torch_device)

            U, S, _Vh = _torch.linalg.svd(V_t, full_matrices=False)
            tol       = S[0].item() * 1e-5 + 1e-12
            U_r       = U[:, S > tol]

            e_parallel_t = U_r @ (U_r.T @ e_t)
            e_perp_t     = e_t - e_parallel_t
            eta = _torch.linalg.norm(e_perp_t).item() / (_torch.linalg.norm(e_t).item() + eps)
        else:
            c, _, _, _  = np.linalg.lstsq(V, e_new, rcond=None)
            e_parallel   = V @ c
            e_perp       = e_new - e_parallel
            eta          = np.linalg.norm(e_perp) / (norm_new + eps)

        return float(min(max(eta, 0.0), 1.0))


    @staticmethod
    def compute_geometric_subspace_SI_diagnostic(state, history, window=None,
                                                 transformer_embeddings=True,
                                                 lambda_damp=5.0, eta_min=0.2,
                                                 membership_threshold=0.0):
        """Geometric Subspace Stalling Index SI^⊥_geom  (Eq. si_geom).

        Detects low-yield semantic recurrence entirely from response geometry,
        without a discrete question-stream counter.

        For each category i ∈ M, with V_i = prior response embeddings for i:
            η_i = ‖ P^⊥_i r(t) ‖ / ‖ r(t) ‖        — orthogonal novelty
            ρ_i = 1 − η_i                            — geometric membership

        SI^⊥_geom(t) = [ Σ_i ρ_i · (1−η_i) · D(Δυ(t)) · (1−υ_i) ]
                       ──────────────────────────────────────────────
                                    Σ_i ρ_i  +  ε

        where D(Δυ(t); λ, η_min) = max(η_min, 1 − λ·Δυ(t))  (Eq. dampening_geom)
        and Δυ(t) = total knowledge gain at turn t, summed across all categories.

        Note: ρ_i · (1−η_i) = (1−η_i)² — the same geometric scalar serves as
        both membership and redundancy measure.

        Args:
            state:                HybridKnowledgeState with .response_log
            history:              List of conversation turn dicts. Used only
                                  to identify the current response embedding;
                                  NOT for label-based repetition counting.
            window:               If set, restrict V_i to the last `window`
                                  entries per category. None = full history.
            transformer_embeddings: Signature parity only (unused).
            lambda_damp:          Gain dampening rate λ (default 5.0)
            eta_min:              Dampening floor η_min (default 0.2)
            membership_threshold: Hard floor on ρ_i below which a category is
                                  excluded (default 0.0 = soft; set e.g. 0.1
                                  to ignore categories with near-zero membership).

        Returns:
            dict with keys:
              si         float in [0, 1] — identical to compute_geometric_subspace_SI
              eta        {dim: η_i} orthogonal novelty, contributing dims only
              rho        {dim: ρ_i} geometric membership, contributing dims only
              dampening  float — D(Δυ(t)) for this exchange
              delta_upsilon_total  float — the Δυ(t) that produced D
              target     str — the category whose log supplied r(t)

        This is the single source of truth for SI^⊥. compute_geometric_subspace_SI
        is a thin wrapper that returns only `si`; the equations are unchanged.
        """
        eps = 1e-12

        if len(history) < 2:
            return GeometricTelemetryTool._empty_si_diagnostic()
        if not hasattr(state, 'response_log'):
            return GeometricTelemetryTool._empty_si_diagnostic()

        # ── Identify current turn's probe embedding r(t) ──────────────────
        # The current response sits at the tail of the labeled category's log.
        # We use history[-1]["target_dimension"] as a hint; fall back to scanning.
        latest_embedding  = None
        latest_upsilon_map = {}
        current_target    = None

        if history and isinstance(history[-1], dict):
            current_target = history[-1].get('target_dimension')

        if current_target and current_target in state.response_log \
                and len(state.response_log[current_target]) >= 1:
            latest_embedding = np.asarray(
                state.response_log[current_target][-1]['embedding'], dtype=np.float64)
        else:
            # Fallback: use the longest log (crude but safe when label is absent)
            longest_dim = None
            longest_len = -1
            for d, lg in state.response_log.items():
                if len(lg) > longest_len:
                    longest_len = len(lg)
                    longest_dim = d
            if longest_dim is None or longest_len < 1:
                return GeometricTelemetryTool._empty_si_diagnostic()
            current_target   = longest_dim
            latest_embedding = np.asarray(
                state.response_log[longest_dim][-1]['embedding'], dtype=np.float64)

        if latest_embedding is None or np.linalg.norm(latest_embedding) < eps:
            return GeometricTelemetryTool._empty_si_diagnostic()

        # ── Total knowledge gain Δυ(t): summed across all categories ──────
        # The paper defines Δυ(t) as the aggregate turn gain (Eq. dampening_geom).
        #
        # IMPORTANT: response_log[d][-1] is the last entry EVER recorded for
        # category d — it may be from a completely different turn.  Summing
        # response_log[d][-1]['gain'] across all d would aggregate gains from
        # many past turns, inflating total_gain and collapsing dampening to
        # eta_min on every turn, including genuine stalls.
        #
        # Correct approach: if the state carries last_total_gain (updated each
        # turn by the caller), use it.  Otherwise, the labeled category's
        # current-turn gain IS the total turn gain — each turn targets exactly
        # one category in this system, so no other category accrues gain this
        # turn and the sum equals the single-category gain.
        if hasattr(state, 'last_total_gain'):
            total_gain = float(state.last_total_gain)
        elif current_target and current_target in state.response_log \
                and len(state.response_log[current_target]) >= 1:
            total_gain = float(
                state.response_log[current_target][-1].get('gain', 0.0))
        else:
            total_gain = 0.0

        # D(Δυ(t); λ, η_min) = max(η_min, 1 − λ·Δυ(t))  (Eq. dampening_geom)
        # Computed once: it is the same scalar for all categories in this turn.
        dampening = max(eta_min, 1.0 - lambda_damp * total_gain)

        # ── Cache per-category υ_i(t) for saturation weight ───────────────
        for d in state.dimensions:
            if d in state.response_log and len(state.response_log[d]) >= 1:
                latest_upsilon_map[d] = state.response_log[d][-1].get(
                    'upsilon', state.values.get(d, 0.0))
            else:
                latest_upsilon_map[d] = state.values.get(d, 0.0)

        numerator   = 0.0
        denominator = 0.0
        eta_map     = {}
        rho_map     = {}

        ##############
        # ── For each category: test geometric membership + redundancy ──────
        for dim in state.dimensions:
            # if dim == 'general': 
            #     continue # "general" is a special catch-all category that should not contribute to stalling signals; skip it.

            # Build V_i from prior responses only.
            # For the labeled category: exclude the last entry (= current response).
            # For all others: use full log (current response is not in them).
            log = state.response_log.get(dim, [])
            prior_entries = log[:-1] if dim == current_target else log

            if len(prior_entries) < 1:
                continue   # no prior subspace → η=1, ρ=0 → no contribution

            if window is not None and len(prior_entries) > window:
                prior_entries = prior_entries[-window:]

            # ── Orthogonal projection onto V_i ─────────────────────────────
            V_cols = [np.asarray(entry['embedding'], dtype=np.float64)
                      for entry in prior_entries]
            V = np.column_stack(V_cols)

            if GeometricTelemetryTool._torch_device is not None:
                V_t = _torch.from_numpy(V.astype(np.float32)).to(
                    GeometricTelemetryTool._torch_device)
                e_t = _torch.from_numpy(
                    latest_embedding.astype(np.float32)).to(
                    GeometricTelemetryTool._torch_device)
                U, S, _Vh = _torch.linalg.svd(V_t, full_matrices=False)
                tol       = S[0].item() * 1e-5 + 1e-12
                U_r       = U[:, S > tol]
                e_perp_t  = e_t - U_r @ (U_r.T @ e_t)
                eta_i     = (_torch.linalg.norm(e_perp_t).item() /
                             (_torch.linalg.norm(e_t).item() + eps))
            else:
                c, _, _, _  = np.linalg.lstsq(V, latest_embedding, rcond=None)
                e_perp       = latest_embedding - V @ c
                eta_i        = (np.linalg.norm(e_perp) /
                                (np.linalg.norm(latest_embedding) + eps))

            eta_i = float(min(max(eta_i, 0.0), 1.0))
            rho_i = 1.0 - eta_i   # geometric membership ρ_i  (Eq. rho)

            if rho_i < membership_threshold:
                continue

            # ρ_i · (1−η_i) = (1−η_i)²  — membership × redundancy
            redundancy   = 1.0 - eta_i   # same scalar as rho_i; named for clarity
            upsilon_i    = latest_upsilon_map.get(dim, 0.0)
            saturation   = 1.0 - upsilon_i   # suppress stall on resolved categories

            contribution = rho_i * redundancy * dampening * saturation
            numerator   += contribution
            denominator += rho_i
            eta_map[dim] = eta_i
            rho_map[dim] = rho_i

        if denominator < eps:
            return GeometricTelemetryTool._empty_si_diagnostic(
                dampening=dampening, delta_upsilon_total=total_gain,
                target=current_target)

        si_geom = numerator / (denominator + eps)
        return {
            'si': float(max(0.0, min(1.0, si_geom))),
            'eta': eta_map,
            'rho': rho_map,
            'dampening': float(dampening),
            'delta_upsilon_total': float(total_gain),
            'target': current_target,
        }
    
   



    @staticmethod
    def _empty_si_diagnostic(dampening=1.0, delta_upsilon_total=0.0, target=None):
        """SI diagnostic for exchanges with no prior subspace (SI is 0 by definition)."""
        return {
            'si': 0.0,
            'eta': {},
            'rho': {},
            'dampening': float(dampening),
            'delta_upsilon_total': float(delta_upsilon_total),
            'target': target,
        }

    @staticmethod
    def compute_geometric_subspace_SI(state, history, window=None,
                                      transformer_embeddings=True,
                                      lambda_damp=5.0, eta_min=0.2,
                                      membership_threshold=0.0):
        """Geometric Subspace Stalling Index SI^⊥ (Eq. si_geom). Scalar only.

        Thin wrapper over compute_geometric_subspace_SI_diagnostic. The signature
        and the returned value are unchanged from the pre-refactor implementation;
        tests/test_frozen_core.py pins numerical equivalence.
        """
        return GeometricTelemetryTool.compute_geometric_subspace_SI_diagnostic(
            state, history, window=window,
            transformer_embeddings=transformer_embeddings,
            lambda_damp=lambda_damp, eta_min=eta_min,
            membership_threshold=membership_threshold,
        )['si']

    # ══════════════════════════════════════════════════════════════════════
    #  DIAGNOSTIC: Orthogonal novelty per category
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def get_subspace_diagnostics(state):
        """Return per-category orthogonal novelty diagnostics.

        Returns:
            dict {dim: {novelty_eta, redundancy, n_contributions, subspace_rank}}
        """
        diagnostics = {}
        eps = 1e-12

        for dim in state.dimensions:
            eta = GeometricTelemetryTool._compute_orthogonal_novelty(state, dim)

            n_contributions = 0
            subspace_rank   = 0
            if dim in state.embedding_history and len(state.embedding_history[dim]) >= 1:
                hist          = state.embedding_history[dim]
                contributions = []
                for idx in range(len(hist) - 1):
                    diff = (np.asarray(hist[idx + 1]['embedding'], dtype=np.float64) -
                            np.asarray(hist[idx]['embedding'],     dtype=np.float64))
                    if np.linalg.norm(diff) > eps:
                        contributions.append(diff)
                # Contribution from history tail to current embedding
                tail = np.asarray(hist[-1]['embedding'], dtype=np.float64)
                curr = np.asarray(state.embeddings[dim],  dtype=np.float64)
                diff = curr - tail
                if np.linalg.norm(diff) > eps:
                    contributions.append(diff)

                n_contributions = len(contributions)
                if contributions:
                    V             = np.column_stack(contributions)
                    subspace_rank = int(np.linalg.matrix_rank(V, tol=1e-6))

            diagnostics[dim] = {
                'novelty_eta':    float(eta),
                'redundancy':     float(1.0 - eta),
                'n_contributions': n_contributions,
                'subspace_rank':  subspace_rank,
            }

        return diagnostics