import numpy as np
from .predict import win_prob, wdl_probs

# ── Official 2026 FIFA World Cup knockout bracket ──────────────────────────────
# R32 slots: each entry is (spec_A, spec_B)
# spec = ("GROUP", 1|2) for group finishers  |  None = 3rd-place placeholder
_R32_SLOTS = [
    (("A", 2), ("B", 2)),   # M73 → R16 M90 → QF M97 → SF1
    (("E", 1), None),        # M74 → R16 M89 → QF M97 → SF1
    (("F", 1), ("C", 2)),   # M75 → R16 M90 → QF M97 → SF1
    (("C", 1), ("F", 2)),   # M76 → R16 M91 → QF M99 → SF2
    (("I", 1), None),        # M77 → R16 M89 → QF M97 → SF1
    (("E", 2), ("I", 2)),   # M78 → R16 M91 → QF M99 → SF2
    (("A", 1), None),        # M79 → R16 M92 → QF M99 → SF2
    (("L", 1), None),        # M80 → R16 M92 → QF M99 → SF2
    (("D", 1), None),        # M81 → R16 M94 → QF M98 → SF1
    (("G", 1), None),        # M82 → R16 M94 → QF M98 → SF1
    (("K", 2), ("L", 2)),   # M83 → R16 M93 → QF M98 → SF1
    (("H", 1), ("J", 2)),   # M84 → R16 M93 → QF M98 → SF1
    (("B", 1), None),        # M85 → R16 M96 → QF M100 → SF2
    (("J", 1), ("H", 2)),   # M86 → R16 M95 → QF M100 → SF2
    (("K", 1), None),        # M87 → R16 M96 → QF M100 → SF2
    (("D", 2), ("G", 2)),   # M88 → R16 M95 → QF M100 → SF2
]

# R16: which R32 match-winner indices (0-15) play each other
_R16_PAIRS = [
    (1, 4),    # M89 → QF M97 → SF1
    (0, 2),    # M90 → QF M97 → SF1
    (3, 5),    # M91 → QF M99 → SF2
    (6, 7),    # M92 → QF M99 → SF2
    (10, 11),  # M93 → QF M98 → SF1
    (8, 9),    # M94 → QF M98 → SF1
    (13, 15),  # M95 → QF M100 → SF2
    (12, 14),  # M96 → QF M100 → SF2
]

# QF: which R16 winner indices (0-7) play each other
_QF_PAIRS = [
    (0, 1),   # M97 → SF1
    (4, 5),   # M98 → SF1
    (2, 3),   # M99 → SF2
    (6, 7),   # M100 → SF2
]

# SF: which QF winner indices (0-3) play each other
_SF_PAIRS = [(0, 1), (2, 3)]


def _sim_group_match(a: str, b: str, team_elos: dict) -> tuple[int, int]:
    pw, pd_, _ = wdl_probs(team_elos[a], team_elos[b])
    r = np.random.random()
    if r < pw:
        return 3, 0
    elif r < pw + pd_:
        return 1, 1
    return 0, 3


def _sim_group(group: list[str], team_elos: dict) -> tuple[list[str], dict]:
    pts = {t: 0 for t in group}
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            pa, pb = _sim_group_match(group[i], group[j], team_elos)
            pts[group[i]] += pa
            pts[group[j]] += pb
    ranked = sorted(group, key=lambda t: (-pts[t], -team_elos[t]))
    return ranked, pts


def _sim_knockout_match(a: str, b: str, team_elos: dict, model) -> tuple[str, str]:
    p = win_prob(model, team_elos[a], team_elos[b])
    if np.random.random() < p:
        return a, b
    return b, a


def _record_matchup(matchup_counts: dict, round_name: str, a: str, b: str) -> None:
    key = (a, b) if a < b else (b, a)
    matchup_counts[round_name][key] = matchup_counts[round_name].get(key, 0) + 1


def sim_tournament(
    GROUPS: dict,
    team_elos: dict,
    model,
    n_sims: int = 10_000,
) -> dict:
    """Run full tournament (group stage + knockout) n_sims times.

    Returns a dict with:
        positions       {team: {1:count, 2:count, 3:count, 4:count}}
        group_finish    {team: {1:count, 2:count, 3:count, 4:count}}
        rounds_reached  {team: {R32:count, R16:count, QF:count, SF:count, Final:count}}
        matchup_counts  {round: {(teamA, teamB): count}}  — sorted tuple, teamA < teamB
        n_sims          int
    """
    all_teams = [t for g in GROUPS.values() for t in g]
    positions     = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t in all_teams}
    group_finish  = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t in all_teams}
    rounds_reached = {
        t: {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "Final": 0}
        for t in all_teams
    }
    matchup_counts = {"R32": {}, "R16": {}, "QF": {}, "SF": {}, "Final": {}}

    for _ in range(n_sims):
        # ── Group stage ───────────────────────────────────────────────────────
        qualifiers = {}   # {("GROUP", 1|2): team}
        third_pool = []
        grp_data   = {}

        for gname, group in GROUPS.items():
            ranked, pts = _sim_group(list(group), team_elos)
            grp_data[gname] = (ranked, pts)
            qualifiers[(gname, 1)] = ranked[0]
            qualifiers[(gname, 2)] = ranked[1]
            third_pool.append((ranked[2], pts[ranked[2]]))

        best_third = [t for t, _ in
                      sorted(third_pool, key=lambda x: (-x[1], -team_elos[x[0]]))[:8]]
        np.random.shuffle(best_third)   # FIFA assigns via lookup table; we randomise

        # ── Build R32 lineup using official bracket ───────────────────────────
        third_idx  = 0
        r32_pairs  = []   # 16 × (teamA, teamB)
        for spec_a, spec_b in _R32_SLOTS:
            if spec_a is not None:
                team_a = qualifiers[spec_a]
            else:
                team_a = best_third[third_idx]; third_idx += 1
            if spec_b is not None:
                team_b = qualifiers[spec_b]
            else:
                team_b = best_third[third_idx]; third_idx += 1
            r32_pairs.append((team_a, team_b))

        # ── R32 ───────────────────────────────────────────────────────────────
        r32_winners = []
        for a, b in r32_pairs:
            w, _ = _sim_knockout_match(a, b, team_elos, model)
            r32_winners.append(w)
            _record_matchup(matchup_counts, "R32", a, b)

        # ── R16 ───────────────────────────────────────────────────────────────
        r16_winners = []
        for ia, ib in _R16_PAIRS:
            a, b = r32_winners[ia], r32_winners[ib]
            w, _ = _sim_knockout_match(a, b, team_elos, model)
            r16_winners.append(w)
            _record_matchup(matchup_counts, "R16", a, b)

        # ── QF ────────────────────────────────────────────────────────────────
        qf_winners = []
        for ia, ib in _QF_PAIRS:
            a, b = r16_winners[ia], r16_winners[ib]
            w, _ = _sim_knockout_match(a, b, team_elos, model)
            qf_winners.append(w)
            _record_matchup(matchup_counts, "QF", a, b)

        # ── SF ────────────────────────────────────────────────────────────────
        sf_winners, sf_losers = [], []
        for ia, ib in _SF_PAIRS:
            a, b = qf_winners[ia], qf_winners[ib]
            w, l = _sim_knockout_match(a, b, team_elos, model)
            sf_winners.append(w); sf_losers.append(l)
            _record_matchup(matchup_counts, "SF", a, b)

        # ── Final ─────────────────────────────────────────────────────────────
        champion, runner_up = _sim_knockout_match(sf_winners[0], sf_winners[1], team_elos, model)
        _record_matchup(matchup_counts, "Final", sf_winners[0], sf_winners[1])
        third, fourth = _sim_knockout_match(sf_losers[0], sf_losers[1], team_elos, model)

        # ── Accumulate stats ──────────────────────────────────────────────────
        for gname, (ranked, _) in grp_data.items():
            for pos, team in enumerate(ranked, 1):
                group_finish[team][pos] += 1

        for a, b in r32_pairs:
            rounds_reached[a]["R32"] += 1
            rounds_reached[b]["R32"] += 1
        for t in r32_winners:
            rounds_reached[t]["R16"] += 1
        for t in r16_winners:
            rounds_reached[t]["QF"] += 1
        for t in qf_winners:
            rounds_reached[t]["SF"] += 1
        for t in sf_winners:
            rounds_reached[t]["Final"] += 1

        positions[champion][1]   += 1
        positions[runner_up][2]  += 1
        positions[third][3]      += 1
        positions[fourth][4]     += 1

    return {
        "positions":      positions,
        "group_finish":   group_finish,
        "rounds_reached": rounds_reached,
        "matchup_counts": matchup_counts,
        "n_sims":         n_sims,
    }


def sim_from_r32(
    r32_teams: list[str],
    team_elos: dict,
    model,
    n_sims: int = 10_000,
) -> dict:
    """Simulate the knockout stage from a given set of 32 teams.

    Used by the Bracket Builder. Returns championship probabilities.
    """
    counts = {t: 0 for t in r32_teams}

    for _ in range(n_sims):
        r = list(r32_teams)
        np.random.shuffle(r)

        while len(r) > 2:
            r = [_sim_knockout_match(r[i], r[i + 1], team_elos, model)[0]
                 for i in range(0, len(r), 2)]

        winner, _ = _sim_knockout_match(r[0], r[1], team_elos, model)
        counts[winner] += 1

    return {t: counts[t] / n_sims for t in r32_teams}
