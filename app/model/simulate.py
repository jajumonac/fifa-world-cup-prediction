import numpy as np
from .predict import win_prob, wdl_probs


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
        n_sims          int
    """
    all_teams = [t for g in GROUPS.values() for t in g]
    positions = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t in all_teams}
    group_finish = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t in all_teams}
    rounds_reached = {
        t: {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "Final": 0}
        for t in all_teams
    }

    for _ in range(n_sims):
        qualifiers, third_pool, grp_data = [], [], {}

        for gname, group in GROUPS.items():
            ranked, pts = _sim_group(list(group), team_elos)
            grp_data[gname] = (ranked, pts)
            qualifiers += ranked[:2]
            third_pool.append((ranked[2], pts[ranked[2]]))

        best_third = sorted(third_pool, key=lambda x: (-x[1], -team_elos[x[0]]))[:8]
        r32 = qualifiers + [t[0] for t in best_third]
        np.random.shuffle(r32)

        r16 = [_sim_knockout_match(r32[i], r32[i + 1], team_elos, model)[0]
               for i in range(0, 32, 2)]
        qf = [_sim_knockout_match(r16[i], r16[i + 1], team_elos, model)[0]
              for i in range(0, 16, 2)]
        sf = [_sim_knockout_match(qf[i], qf[i + 1], team_elos, model)[0]
              for i in range(0, 8, 2)]

        sf_w1, sf_l1 = _sim_knockout_match(sf[0], sf[1], team_elos, model)
        sf_w2, sf_l2 = _sim_knockout_match(sf[2], sf[3], team_elos, model)
        champion, runner_up = _sim_knockout_match(sf_w1, sf_w2, team_elos, model)
        third, fourth = _sim_knockout_match(sf_l1, sf_l2, team_elos, model)

        for gname, (ranked, _) in grp_data.items():
            for pos, team in enumerate(ranked, 1):
                group_finish[team][pos] += 1

        for team in set(r32):
            rounds_reached[team]["R32"] += 1
        for team in set(r16):
            rounds_reached[team]["R16"] += 1
        for team in set(qf):
            rounds_reached[team]["QF"] += 1
        for team in set(sf):
            rounds_reached[team]["SF"] += 1
        for team in {sf_w1, sf_w2}:
            rounds_reached[team]["Final"] += 1

        positions[champion][1] += 1
        positions[runner_up][2] += 1
        positions[third][3] += 1
        positions[fourth][4] += 1

    return {
        "positions": positions,
        "group_finish": group_finish,
        "rounds_reached": rounds_reached,
        "n_sims": n_sims,
    }


def sim_from_r32(
    r32_teams: list[str],
    team_elos: dict,
    model,
    n_sims: int = 10_000,
) -> dict:
    """Simulate the knockout stage from a given set of 32 teams.

    Used by the Bracket Builder and Live Tracker (knockout phase).
    Returns championship probabilities for each team.
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
