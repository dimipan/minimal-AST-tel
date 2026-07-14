"""Pre-refactor SI^perp implementation, vendored verbatim as a regression oracle.

Do not edit. tests/test_frozen_core.py asserts that the refactored
compute_geometric_subspace_SI (now a wrapper over the diagnostic variant)
returns bit-comparable values against this reference. The GPU branch is
disabled so the oracle is deterministic on any machine.
"""

import numpy as np


def reference_si(state, history, window=None,
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
        Float in [0, 1].  A stall is declared when the return value > θ.
    """
    eps = 1e-12

    if len(history) < 2:
        return 0.0
    if not hasattr(state, 'response_log'):
        return 0.0

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
            return 0.0
        current_target   = longest_dim
        latest_embedding = np.asarray(
            state.response_log[longest_dim][-1]['embedding'], dtype=np.float64)

    if latest_embedding is None or np.linalg.norm(latest_embedding) < eps:
        return 0.0

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

        if False:
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

    if denominator < eps:
        return 0.0

    si_geom = numerator / (denominator + eps)
    return float(max(0.0, min(1.0, si_geom)))

   


