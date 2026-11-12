"""Calendar layer: festival groups, state festival lists and festival contexts.

A festival context is a label such as ``"diwali@-1"``: the festival group and the signed day offset
(negative = days before the festival, positive = days after). Days outside every window have no context.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Set

import pandas as pd

FESTIVAL_GROUPS: Dict[str, list] = {
    # "bhai duj" and variants, not plain "bhai", which also matches "Sardar Vallabhbhai Patel's Jayanti"
    "diwali": ["diwali", "deepavali", "govardhan", "bhai duj", "bhai dooj", "bhaiya dooj"],
    "harvest": ["pongal", "sankranti", "makar", "thiruvalluvar", "uzhavar", "lohri", "bihu", "magh"],
    "holi": ["holi"],
    "dussehra": ["dussehra", "vijaya", "ayudha", "durga", "navaratri", "maha navami", "mahanavami",
                 "saptami", "ashtami", "mahalaya"],
    "onam": ["onam", "thiruvonam"],
    "ganesh": ["ganesh", "vinayaka"],
    "eid": ["id-ul", "eid", "bakrid", "muharram", "milad"],
    "christmas": ["christmas"],
    "newyear_regional": ["ugadi", "puthandu", "gudi", "vishu", "baisakhi", "vaisakhi", "rongali",
                         "bengali new year", "cheti chand"],
    "national": ["republic", "independence", "gandhi", "ambedkar", "may day", "labour"],
}

PRIORITY = ["diwali", "harvest", "holi", "dussehra", "onam", "ganesh", "eid", "christmas",
            "newyear_regional", "national", "other"]

PRE, POST = 3, 2  # window: 3 days before to 2 days after a festival


def festival_group(name: str) -> str:
    """Map a holiday name to its festival group ("other" if none matches)."""
    n = str(name).lower()
    for g, keys in FESTIVAL_GROUPS.items():
        if any(k in n for k in keys):
            return g
    return "other"


def state_festivals(subdiv: Optional[str], years: Iterable[int]) -> Dict[pd.Timestamp, Set[str]]:
    """{date: {groups}} for one Indian state (ISO 3166-2 subdivision code, e.g. "AP"), from `holidays`."""
    import holidays  # imported lazily so the core works without it

    years = list(years)
    h = holidays.India(years=years, subdiv=subdiv) if subdiv else holidays.India(years=years)
    out: Dict[pd.Timestamp, Set[str]] = {}
    for d, name in h.items():
        for part in str(name).split(";"):
            out.setdefault(pd.Timestamp(d), set()).add(festival_group(part))
    return out


def festival_contexts(dates: Iterable, festivals: Mapping, pre: int = PRE, post: int = POST) -> pd.Series:
    """Label each date with its dominant festival context, or None.

    The nearest festival within ``-pre..+post`` days wins; ties are broken by ``PRIORITY``.
    ``festivals`` maps a date to a group name or a set of group names.
    """
    fest = []
    for d, g in festivals.items():
        groups = {g} if isinstance(g, str) else set(g)
        fest.append((pd.Timestamp(d).normalize(),
                     sorted(groups, key=lambda x: PRIORITY.index(x) if x in PRIORITY else len(PRIORITY))[0]))
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
    out = []
    for d in idx:
        best = None
        for fd, g in fest:
            k = (d - fd).days
            if -pre <= k <= post:
                key = (abs(k), PRIORITY.index(g) if g in PRIORITY else len(PRIORITY))
                if best is None or key < best[0]:
                    best = (key, f"{g}@{k:+d}")
        out.append(best[1] if best else None)
    return pd.Series(out, index=idx, dtype=object)


def hour_block(hour: int) -> str:
    """Four six-hour blocks used for hourly effect tables."""
    return ["00-06", "06-12", "12-18", "18-24"][int(hour) // 6]
