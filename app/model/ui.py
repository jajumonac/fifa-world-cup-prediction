"""Shared UI helpers -- injected on every page."""
import streamlit as st

GITHUB_REPO = "https://github.com/jajumonac/fifa-world-cup-prediction"
LINKEDIN    = "https://www.linkedin.com/in/jadiel-montero-230814142/"
GITHUB_USER = "https://github.com/jajumonac"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif !important; }

    /* App shell */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"]             { background: #05172E !important; }
    [data-testid="stHeader"]           { background: transparent !important; box-shadow: none !important; }
    [data-testid="stDecoration"]       { display: none !important; }

    /* Typography */
    h1 { font-size: 2rem !important; font-weight: 800 !important; color: #FFFFFF !important; letter-spacing: -0.3px; }
    h2 { font-size: 1.3rem !important; font-weight: 700 !important; color: #FFFFFF !important; }
    h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #E8F0FE !important; }
    h4 { font-size: 0.85rem !important; font-weight: 700 !important; color: #8BA3C1 !important;
         letter-spacing: 0.08em; text-transform: uppercase; }
    p, span, li, div { color: #C8D8EE; }
    .stMarkdown p { line-height: 1.7; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #0A1F3B !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 10px !important;
        padding: 18px 20px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #556B8A !important; font-size: 0.7rem !important;
        font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.06em;
    }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 800 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #030E1E !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    [data-testid="stSidebarNavLink"] {
        color: #6A8AAC !important; border-radius: 6px !important;
        margin: 2px 6px !important; font-size: 0.88rem !important;
    }
    [data-testid="stSidebarNavLink"]:hover {
        background: rgba(255,255,255,0.05) !important; color: #FFFFFF !important;
    }
    [data-testid="stSidebarNavLink"][aria-selected="true"] {
        background: rgba(184,0,42,0.18) !important; color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: #B8002A !important; color: #FFFFFF !important;
        border: none !important; border-radius: 6px !important;
        font-weight: 600 !important; letter-spacing: 0.02em;
        transition: background 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover { background: #D40030 !important; }
    .stButton > button {
        background: rgba(255,255,255,0.05) !important; color: #C8D8EE !important;
        border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 6px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background: rgba(255,255,255,0.09) !important; color: #FFFFFF !important;
        border-color: rgba(255,255,255,0.2) !important;
    }

    /* Radio */
    [data-testid="stRadio"] label         { color: #C8D8EE !important; font-size: 0.85rem !important; }
    [data-testid="stRadio"] > label       { color: #8BA3C1 !important; }
    [data-testid="stRadio"] div[role="radiogroup"] > label { font-size: 0.82rem !important; }

    /* Select box */
    [data-testid="stSelectbox"] > div > div {
        background: #0A1F3B !important; border-color: rgba(255,255,255,0.12) !important;
        color: #FFFFFF !important;
    }

    /* Sliders */
    [data-testid="stSlider"] > div { color: #C8D8EE !important; }

    /* Expander */
    [data-testid="stExpander"] {
        background: #0A1F3B !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary { color: #C8D8EE !important; font-weight: 600 !important; }

    /* Alert boxes */
    [data-testid="stAlert"] {
        background: rgba(240,180,41,0.08) !important;
        border-left: 3px solid #F0B429 !important;
        border-radius: 0 6px 6px 0 !important;
    }
    [data-testid="stAlert"] p, [data-testid="stAlert"] strong { color: #F0D080 !important; }

    /* Forms */
    [data-testid="stForm"] {
        background: #0A1F3B !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 10px !important;
        padding: 16px !important;
    }

    /* Dividers */
    hr { border-color: rgba(255,255,255,0.07) !important; margin: 2rem 0 !important; }

    /* Checkbox / multiselect */
    [data-testid="stMultiSelect"] > div {
        background: #0A1F3B !important; border-color: rgba(255,255,255,0.12) !important;
    }

    /* Footer */
    .wc-footer {
        margin-top: 3rem; padding: 1.2rem 0 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.07);
        text-align: center; color: #3A5070; font-size: 0.77rem; line-height: 2;
    }
    .wc-footer a { color: #4A7AAA !important; text-decoration: none; }
    .wc-footer a:hover { color: #6A9ACA !important; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page components
# ---------------------------------------------------------------------------
def render_footer():
    st.markdown(
        '<div class="wc-footer">'
        'Built by Jadiel Montero &nbsp;&middot;&nbsp; '
        '<a href="' + LINKEDIN + '" target="_blank">LinkedIn</a>'
        ' &nbsp;&middot;&nbsp; '
        '<a href="' + GITHUB_USER + '" target="_blank">GitHub</a>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_disclaimer():
    st.warning(
        "**Disclaimer:** This tool is for educational and entertainment purposes only. "
        "It is not financial or betting advice. Predictions are probabilistic estimates "
        "based on historical data and do not guarantee any specific outcome.",
        icon=None,
    )


def render_github_link():
    st.markdown(
        '<div style="text-align:center; padding:12px 0;">'
        '<a href="' + GITHUB_REPO + '" target="_blank" '
        'style="color:#4A7AAA; font-weight:600; font-size:0.9rem; text-decoration:none;">'
        'View full model documentation on GitHub</a>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Interactive bracket helpers
# ---------------------------------------------------------------------------
def _win_prob_between(model, team_elos, team_a, team_b):
    from model.predict import win_prob
    return win_prob(model, team_elos.get(team_a, 1500.0), team_elos.get(team_b, 1500.0))


def _default_winner(model, team_elos, team_a, team_b):
    p = _win_prob_between(model, team_elos, team_a, team_b)
    return team_a if p >= 0.5 else team_b


def _get_r16_pairs(matchup_counts, N):
    """Greedy: 8 most-likely non-overlapping R16 matchups."""
    pairs, used = [], set()
    for (a, b), cnt in sorted(matchup_counts["R16"].items(), key=lambda x: -x[1]):
        if a not in used and b not in used:
            pairs.append((a, b))
            used.update([a, b])
        if len(pairs) == 8:
            break
    return pairs


def _invalidate_downstream(pick_key):
    """Clear picks for all stages that depend on this match result."""
    bp = st.session_state.bracket_picks
    stage, idx_str = pick_key.rsplit("_", 1)
    idx = int(idx_str)
    if stage == "R16":
        qf_i = idx // 2
        bp.pop("QF_" + str(qf_i), None)
        sf_i = qf_i // 2
        bp.pop("SF_" + str(sf_i), None)
        bp.pop("Final_0", None)
    elif stage == "QF":
        sf_i = idx // 2
        bp.pop("SF_" + str(sf_i), None)
        bp.pop("Final_0", None)
    elif stage == "SF":
        bp.pop("Final_0", None)


def _fill_defaults(model, team_elos):
    """Auto-fill any missing picks with model predictions."""
    bp  = st.session_state.bracket_picks
    r16 = st.session_state.r16_bracket

    for i, (a, b) in enumerate(r16):
        bp.setdefault("R16_" + str(i), _default_winner(model, team_elos, a, b))

    for i in range(4):
        qa = bp.get("R16_" + str(2 * i))
        qb = bp.get("R16_" + str(2 * i + 1))
        if qa and qb:
            bp.setdefault("QF_" + str(i), _default_winner(model, team_elos, qa, qb))

    for i in range(2):
        sa = bp.get("QF_" + str(2 * i))
        sb = bp.get("QF_" + str(2 * i + 1))
        if sa and sb:
            bp.setdefault("SF_" + str(i), _default_winner(model, team_elos, sa, sb))

    fa = bp.get("SF_0")
    fb = bp.get("SF_1")
    if fa and fb:
        bp.setdefault("Final_0", _default_winner(model, team_elos, fa, fb))


def _match_card(team_a, team_b, pick_key, model, team_elos):
    """Render one match card + radio and handle picks."""
    prob_a  = _win_prob_between(model, team_elos, team_a, team_b)
    prob_b  = 1.0 - prob_a
    current = st.session_state.bracket_picks.get(
        pick_key, team_a if prob_a >= 0.5 else team_b
    )
    aw, bw = current == team_a, current == team_b

    def _row(name, prob, is_winner):
        border = "border-left:2px solid #F0B429; padding-left:7px; margin-left:-7px;" if is_winner else ""
        tc     = "#FFFFFF" if is_winner else "#8BA3C1"
        fw     = "700"     if is_winner else "400"
        pb_bg  = "rgba(240,180,41,0.12)" if is_winner else "rgba(255,255,255,0.05)"
        pb_c   = "#F0B429" if is_winner else "#445566"
        return (
            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'padding:5px 0;' + border + '">'
            '<span style="color:' + tc + ';font-weight:' + fw + ';font-size:0.87rem;">' + name + '</span>'
            '<span style="background:' + pb_bg + ';color:' + pb_c + ';border-radius:4px;'
            'padding:2px 7px;font-size:0.73rem;font-weight:700;">' + format(prob, '.0%') + '</span>'
            '</div>'
        )

    card = (
        '<div style="background:#0A1F3B;border:1px solid rgba(255,255,255,0.07);'
        'border-radius:8px;padding:10px 12px;margin-bottom:4px;">'
        + _row(team_a, prob_a, aw)
        + '<div style="text-align:center;color:rgba(255,255,255,0.18);font-size:0.65rem;'
          'letter-spacing:0.08em;padding:2px 0;">vs</div>'
        + _row(team_b, prob_b, bw)
        + '</div>'
    )
    st.markdown(card, unsafe_allow_html=True)

    new_pick = st.radio(
        "Winner",
        [team_a, team_b],
        index=0 if current == team_a else 1,
        key="radio_" + pick_key,
        label_visibility="collapsed",
        horizontal=True,
    )
    if new_pick != current:
        st.session_state.bracket_picks[pick_key] = new_pick
        _invalidate_downstream(pick_key)
        st.rerun()


def render_interactive_bracket(matchup_counts, model, team_elos, N):
    """Render the full interactive R16 -> QF -> SF -> Final bracket."""
    if "r16_bracket" not in st.session_state:
        st.session_state.r16_bracket = _get_r16_pairs(matchup_counts, N)
    if "bracket_picks" not in st.session_state:
        st.session_state.bracket_picks = {}

    _fill_defaults(model, team_elos)
    bp  = st.session_state.bracket_picks
    r16 = st.session_state.r16_bracket

    def _arrow():
        st.markdown(
            '<div style="text-align:center;color:rgba(255,255,255,0.12);'
            'font-size:1.3rem;padding:4px 0;">&#8595;</div>',
            unsafe_allow_html=True,
        )

    # Round of 16
    st.markdown("#### Round of 16")
    cols = st.columns(4)
    for i, (a, b) in enumerate(r16):
        with cols[i % 4]:
            _match_card(a, b, "R16_" + str(i), model, team_elos)

    _arrow()

    # Quarter-finals
    st.markdown("#### Quarter-finals")
    cols = st.columns(4)
    for i in range(4):
        qa = bp.get("R16_" + str(2 * i),     "TBD")
        qb = bp.get("R16_" + str(2 * i + 1), "TBD")
        with cols[i]:
            if qa != "TBD" and qb != "TBD":
                _match_card(qa, qb, "QF_" + str(i), model, team_elos)

    _arrow()

    # Semi-finals
    st.markdown("#### Semi-finals")
    _, c1, _, c2, _ = st.columns([1, 3, 1, 3, 1])
    for i, col in enumerate([c1, c2]):
        sa = bp.get("QF_" + str(2 * i),     "TBD")
        sb = bp.get("QF_" + str(2 * i + 1), "TBD")
        with col:
            if sa != "TBD" and sb != "TBD":
                _match_card(sa, sb, "SF_" + str(i), model, team_elos)

    _arrow()

    # Final
    st.markdown("#### Final")
    _, col, _ = st.columns([2, 3, 2])
    fa = bp.get("SF_0", "TBD")
    fb = bp.get("SF_1", "TBD")
    with col:
        if fa != "TBD" and fb != "TBD":
            _match_card(fa, fb, "Final_0", model, team_elos)

    # Champion card
    champion = bp.get("Final_0")
    if champion and fa != "TBD" and fb != "TBD":
        _arrow()
        _, col, _ = st.columns([2, 3, 2])
        with col:
            opponent = fb if champion == fa else fa
            p_win    = _win_prob_between(model, team_elos, champion, opponent)
            st.markdown(
                '<div style="background:linear-gradient(135deg,#1A1200,#251800);'
                'border:1px solid #F0B429;border-radius:10px;padding:22px;text-align:center;">'
                '<div style="color:#F0B429;font-size:0.72rem;font-weight:700;'
                'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;">'
                'Predicted Champion</div>'
                '<div style="color:#FFFFFF;font-size:1.4rem;font-weight:800;margin-bottom:4px;">'
                + champion +
                '</div>'
                '<div style="color:#C8A830;font-size:0.88rem;font-weight:600;">'
                + format(p_win, '.0%') + ' win probability in final'
                '</div></div>',
                unsafe_allow_html=True,
            )

    # Reset
    st.markdown("<br>", unsafe_allow_html=True)
    _, rc, _ = st.columns([2, 2, 2])
    with rc:
        if st.button("Reset to model predictions"):
            st.session_state.bracket_picks = {}
            st.rerun()
