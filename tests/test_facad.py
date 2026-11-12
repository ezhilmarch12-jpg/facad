import numpy as np
import pandas as pd
import pytest

from facad import FACAD, EffectTable, MondrianConformal, coherence, festival_contexts, festival_group, seasonal_z


def test_festival_group_keyword_fix():
    assert festival_group("Diwali") == "diwali"
    assert festival_group("Bhai Duj") == "diwali"
    assert festival_group("Sardar Vallabhbhai Patel's Jayanti") == "other"


def test_contexts_window_and_priority():
    c = festival_contexts(pd.date_range("2026-11-03", "2026-11-12"), {"2026-11-08": "diwali", "2026-11-11": "eid"})
    assert c[pd.Timestamp("2026-11-05")] == "diwali@-3"
    assert c[pd.Timestamp("2026-11-04")] is None
    assert c[pd.Timestamp("2026-11-08")] == "diwali@+0"
    assert c[pd.Timestamp("2026-11-10")] == "eid@-1"          # nearer festival wins


def test_coherence_bounds():
    assert coherence([1, 2, 1, 3], ["a", "a", "b", "b"]) == 1.0
    assert coherence([1, 1, -1, -1], ["a", "a", "b", "b"]) == 0.0


def _data(sign_b=1.0, n_days=40):
    rng = np.random.default_rng(1)
    rows = []
    for r, sg in [("a", 1.0), ("b", sign_b), ("c", 1.0)]:
        for d in range(n_days):
            c = "x@+0" if d % 10 == 0 else None
            z = rng.normal(0, 1) + (2.0 * sg if c else 0) + (rng.normal(0, 2) if c else 0)
            rows.append({"r": r, "d": d, "ctx": c, "z": z})
    return pd.DataFrame(rows)


def test_shrinkage_and_spread_floor():
    D = _data()
    t = EffectTable(lam=5).fit(D[["z"]], D.ctx, D.r, D.d)
    E, V, n, C = t.lookup("x@+0")["z"]
    assert n == 12 and 0 < E < np.median(D.z[D.ctx.notna()])  # shrunk towards 0
    assert V >= 1.0


def test_gate_removes_incoherent_shift():
    D = _data(sign_b=-1.0)
    D.loc[D.r == "c", "z"] *= -1                                # regions disagree on the direction
    t = EffectTable(lam=5, gate=True, tau=0.9).fit(D[["z"]], D.ctx, D.r, D.d)
    E, V, n, C = t.lookup("x@+0")["z"]
    assert C < 0.9 and E == 0.0 and V >= 1.0


def test_spread_only_never_raises_alarms():
    """Proposition 2: with E = 0 and V >= 1, |z|/V <= |z|, so no new false alarms."""
    D = _data()
    m = FACAD(mode="spread").fit(D[["z"]], D.ctx, D.r, D.d)
    adj = m.score(D[["z"]], D.ctx).values
    assert np.all(adj <= D.z.abs().values + 1e-12)


def test_hierarchical_and_prior():
    D = _data()
    m = FACAD(hierarchical=True, spread_prior={"x@+0": 3.0}).fit(
        D[["z"]], D.ctx, D.r, D.d, region_group={"a": "south", "b": "south", "c": "north"})
    E, V = m.effects(D.ctx, D.r)
    assert (V.z[D.ctx.notna().values] >= 3.0).all()


def test_mondrian_coverage():
    rng = np.random.default_rng(0)
    s = np.r_[rng.normal(0, 1, 4000), rng.normal(0, 3, 4000)]
    cat = np.r_[["normal"] * 4000, ["window"] * 4000]
    mc = MondrianConformal(alpha=0.05).fit(s, cat)
    s2 = np.r_[rng.normal(0, 1, 4000), rng.normal(0, 3, 4000)]
    alarms = mc.predict(s2, cat)
    for k in ("normal", "window"):
        assert abs(alarms[cat == k].mean() - 0.05) < 0.015


def test_seasonal_z_uses_past_only():
    rng = np.random.default_rng(2)
    y = pd.Series(rng.normal(100, 5, 80))
    mask = np.r_[np.ones(79, bool), False]          # scale from earlier rows only
    y2 = y.copy(); y2.iloc[-1] = 1e6                  # change the future
    a, b = seasonal_z(y, mask_scale=mask), seasonal_z(y2, mask_scale=mask)
    assert np.allclose(a.iloc[:-1], b.iloc[:-1], equal_nan=True)
