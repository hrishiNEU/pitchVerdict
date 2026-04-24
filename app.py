"""
Pitch Verdict — Streamlit Web Application
No sidebar. Match cards on main page. Run button inline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Pitch Verdict",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "Pitch Verdict — AI-verified tactical match reports. INFO 7375."}
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --green-dark: #0f2318; --green-mid: #1a472a;
    --gold: #c9a84c; --verified: #27ae60; --flagged: #e74c3c; --uncertain: #f39c12;
    --bg: #0d1117; --bg-card: #161b22; --bg-card-2: #1c2128;
    --text: #e6edf3; --muted: #8b949e; --border: #30363d;
}

.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
section[data-testid="stSidebarContent"] { display: none !important; }
button[kind="header"] { display: none !important; }
h1,h2,h3 { font-family: 'Bebas Neue', cursive !important; letter-spacing: 0.05em; }

.topbar { display:flex; align-items:center; justify-content:space-between; padding:0.8rem 2rem; background:var(--green-dark); border-bottom:2px solid var(--gold); margin:-1rem -1rem 0; }
.topbar-brand { font-family:'Bebas Neue',cursive; font-size:1.6rem; letter-spacing:0.1em; color:#fff; line-height:1; }
.topbar-tag { font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:var(--gold); letter-spacing:0.1em; text-transform:uppercase; margin-top:2px; }
.topbar-pill { font-family:'JetBrains Mono',monospace; font-size:0.68rem; padding:3px 10px; border-radius:20px; letter-spacing:0.06em; }
.pill-ok { background:rgba(39,174,96,0.15); color:#2ecc71; border:1px solid #27ae60; }
.pill-warn { background:rgba(243,156,18,0.15); color:#f39c12; border:1px solid #e67e22; }

.hero { background:linear-gradient(160deg,var(--green-mid) 0%,var(--green-dark) 55%,var(--bg) 100%); padding:2.4rem 2rem 2rem; margin:0 -1rem 2rem; }
.hero-title { font-family:'Bebas Neue',cursive; font-size:3.6rem; letter-spacing:0.06em; color:#fff; line-height:1; margin:0; }
.hero-sub { font-family:'Source Serif 4',serif; font-style:italic; font-size:1.05rem; color:var(--gold); margin:0.4rem 0 1rem; }
.hero-desc { font-family:'Source Serif 4',serif; font-size:0.92rem; color:rgba(230,237,243,0.7); max-width:600px; line-height:1.6; }
.hero-pills { display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:1rem; }
.hero-pill { font-family:'JetBrains Mono',monospace; font-size:0.65rem; padding:3px 10px; border-radius:20px; border:1px solid rgba(201,168,76,0.35); color:rgba(201,168,76,0.8); background:rgba(201,168,76,0.05); letter-spacing:0.07em; text-transform:uppercase; }

.sec-label { font-family:'JetBrains Mono',monospace; font-size:0.65rem; letter-spacing:0.14em; text-transform:uppercase; color:var(--muted); margin-bottom:0.75rem; padding-bottom:0.45rem; border-bottom:1px solid var(--border); }

.match-card { background:var(--bg-card); border:1.5px solid var(--border); border-radius:10px; padding:1rem 1.2rem; position:relative; box-sizing:border-box; }
.match-card-sel { border-color:var(--gold) !important; background:rgba(201,168,76,0.05) !important; }
.match-comp { font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:var(--muted); letter-spacing:0.07em; text-transform:uppercase; margin-bottom:0.45rem; }
.match-score-row { display:flex; align-items:center; justify-content:space-between; margin:0.25rem 0; }
.match-team { font-family:'Bebas Neue',cursive; font-size:1.25rem; letter-spacing:0.03em; color:var(--text); flex:1; }
.match-score { font-family:'Bebas Neue',cursive; font-size:1.65rem; letter-spacing:0.08em; color:var(--gold); padding:0 0.6rem; min-width:54px; text-align:center; }
.match-meta { font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:var(--muted); margin-top:0.35rem; }
.match-narrative { font-family:'Source Serif 4',serif; font-style:italic; font-size:0.8rem; color:var(--muted); margin-top:0.5rem; line-height:1.5; padding-top:0.5rem; border-top:1px solid var(--border); }

.run-bar { background:var(--bg-card); border:1px solid var(--border); border-radius:8px; padding:0.6rem 1rem; font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:var(--muted); }

.pipe-step { display:flex; align-items:center; gap:0.65rem; padding:0.6rem 0.85rem; font-family:'JetBrains Mono',monospace; font-size:0.76rem; border-radius:6px; border:1px solid var(--border); background:var(--bg-card); margin-bottom:0.35rem; color:var(--muted); }
.pipe-active { border-color:var(--gold) !important; background:rgba(201,168,76,0.06) !important; color:var(--text) !important; }
.pipe-done { border-color:var(--verified) !important; background:rgba(39,174,96,0.05) !important; color:var(--text) !important; }

.result-hdr { background:linear-gradient(135deg,#161b22 0%,#0d2b1a 100%); border:1px solid var(--border); border-radius:10px; padding:1.3rem 1.8rem; margin-bottom:1.4rem; }
.result-comp { font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem; }
.result-teams { display:flex; align-items:center; justify-content:space-between; }
.result-team { font-family:'Bebas Neue',cursive; font-size:2rem; letter-spacing:0.03em; color:var(--text); }
.result-score { font-family:'Bebas Neue',cursive; font-size:2.8rem; letter-spacing:0.08em; color:var(--gold); text-align:center; padding:0 1rem; }
.result-model { font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:var(--muted); margin-top:0.4rem; }

.sec-hdr { font-family:'Bebas Neue',cursive; font-size:1.2rem; letter-spacing:0.07em; color:var(--gold); border-bottom:1px solid var(--border); padding-bottom:0.3rem; margin:1.2rem 0 0.8rem; }

.acc-row { display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace; font-size:0.73rem; margin-bottom:3px; }
.acc-track { background:var(--bg-card-2); border-radius:4px; height:6px; overflow:hidden; }
.acc-fill { height:100%; border-radius:4px; }
.acc-sub { font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:var(--muted); margin-top:3px; }

.ppda-chip { display:inline-block; border-radius:4px; padding:2px 9px; font-family:'JetBrains Mono',monospace; font-size:0.7rem; font-weight:600; }
.key-moment { background:rgba(201,168,76,0.06); border:1px solid rgba(201,168,76,0.25); border-radius:6px; padding:0.5rem 0.9rem; font-family:'JetBrains Mono',monospace; font-size:0.76rem; margin-bottom:0.35rem; }
.timeline-ev { display:flex; align-items:center; gap:0.9rem; padding:0.65rem 0.9rem; background:var(--bg-card); border:1px solid var(--border); border-radius:6px; margin-bottom:0.35rem; font-family:'JetBrains Mono',monospace; font-size:0.78rem; }

[data-testid="metric-container"] { background:var(--bg-card); border:1px solid var(--border); border-radius:8px; padding:0.75rem; }
.stProgress > div > div > div { background-color:var(--gold) !important; }
.stTabs [data-baseweb="tab"] { font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────

def api_status():
    if os.getenv('XAI_API_KEY'): return "ok", "xAI Grok"
    if os.getenv('ANTHROPIC_API_KEY'): return "ok", "Anthropic"
    if os.getenv('OPENAI_API_KEY'): return "ok", "OpenAI"
    return "warn", "Demo mode — no API key"

def ppda_color(v):
    return "#e74c3c" if v < 8 else "#f39c12" if v < 12 else "#3498db"

def ppda_label(v):
    return "HIGH PRESS" if v < 8 else "MID-BLOCK" if v < 12 else "DEEP BLOCK"

def render_ppda_chip(v):
    c = ppda_color(v)
    st.markdown(f'<span class="ppda-chip" style="background:rgba(0,0,0,0.3);color:{c};border:1px solid {c};">PPDA {v:.1f} — {ppda_label(v)}</span>', unsafe_allow_html=True)

def render_accuracy(acc):
    c = "#27ae60" if acc >= 0.95 else "#f39c12" if acc >= 0.80 else "#e74c3c"
    pct = int(acc * 100)
    st.markdown(f"""<div style="margin:.5rem 0;">
    <div class="acc-row"><span style="color:var(--muted);">FACTUAL ACCURACY</span><span style="color:{c};font-weight:600;">{pct}%</span></div>
    <div class="acc-track"><div class="acc-fill" style="width:{pct}%;background:{c};"></div></div>
    <div class="acc-sub">Target ≥ 95% · {"✅ PASSED" if acc >= .90 else "⚠️ BELOW TARGET"}</div>
    </div>""", unsafe_allow_html=True)

def render_pipeline(step, labels):
    icons = ["📥","✂️","🧠","✍️","🔍"]
    for i, lbl in enumerate(labels):
        cls = "pipe-done" if i < step else "pipe-active" if i == step else ""
        icon = "✅" if i < step else icons[i]
        st.markdown(f'<div class="pipe-step {cls}"><span>{icon}</span><span><b>Agent {i+1}:</b> {lbl}</span></div>', unsafe_allow_html=True)

def make_chart(phases, home, away, kind):
    try:
        import plotly.graph_objects as go
        names = [p.name[:18] for p in phases]
        fig = go.Figure()

        if kind == "possession":
            fig.add_trace(go.Bar(name=home, x=names, y=[p.home_possession for p in phases], marker_color='#c9a84c', text=[f"{v:.0f}%" for v in [p.home_possession for p in phases]], textposition='inside'))
            fig.add_trace(go.Bar(name=away, x=names, y=[p.away_possession for p in phases], marker_color='#2980b9', text=[f"{v:.0f}%" for v in [p.away_possession for p in phases]], textposition='inside'))
            fig.update_layout(barmode='stack', yaxis=dict(range=[0,100], ticksuffix='%', gridcolor='#30363d'), title=dict(text="Possession by Phase", font=dict(size=12, color='#c9a84c')))
        elif kind == "xg":
            fig.add_trace(go.Scatter(name=home, x=list(range(len(phases))), y=list(np.cumsum([p.home_xg for p in phases])), mode='lines+markers', line=dict(color='#c9a84c', width=2.5), marker=dict(size=8)))
            fig.add_trace(go.Scatter(name=away, x=list(range(len(phases))), y=list(np.cumsum([p.away_xg for p in phases])), mode='lines+markers', line=dict(color='#2980b9', width=2.5), marker=dict(size=8)))
            fig.update_layout(xaxis=dict(tickvals=list(range(len(phases))), ticktext=[p.name[:14] for p in phases], tickfont=dict(size=8), gridcolor='#30363d'), yaxis=dict(title='Cumulative xG', gridcolor='#30363d'), title=dict(text="Cumulative xG", font=dict(size=12, color='#c9a84c')))
        else:
            fig.add_hline(y=8, line_dash="dot", line_color="#e74c3c", opacity=0.4, annotation_text="High Press (< 8)", annotation_font_size=9)
            fig.add_hline(y=12, line_dash="dot", line_color="#f39c12", opacity=0.4, annotation_text="Mid-Block (< 12)", annotation_font_size=9)
            fig.add_trace(go.Scatter(name=home, x=names, y=[min(p.home_ppda,30) for p in phases], mode='lines+markers', line=dict(color='#c9a84c', width=2.5), marker=dict(size=10, symbol='diamond')))
            fig.add_trace(go.Scatter(name=away, x=names, y=[min(p.away_ppda,30) for p in phases], mode='lines+markers', line=dict(color='#2980b9', width=2.5), marker=dict(size=10, symbol='circle')))
            fig.update_layout(yaxis=dict(autorange='reversed', title='PPDA (lower = more pressing)', gridcolor='#30363d'), xaxis=dict(gridcolor='#30363d'), title=dict(text="Pressing Intensity (PPDA)", font=dict(size=12, color='#c9a84c')))

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='JetBrains Mono', color='#e6edf3', size=11), legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='#30363d'), margin=dict(l=0,r=0,t=30,b=0), height=260)
        return fig
    except ImportError:
        return None


# ── Top Nav ──────────────────────────────────────────────────────
status_cls, status_lbl = api_status()
pill_cls = "pill-ok" if status_cls == "ok" else "pill-warn"
st.markdown(f"""
<div class="topbar">
    <div>
        <div class="topbar-brand">⚽ PITCH VERDICT</div>
        <div class="topbar-tag">Five agents · Real data · Every claim verified</div>
    </div>
    <div style="display:flex;align-items:center;gap:1rem;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--muted);">INFO 7375 — Generative AI</span>
        <span class="topbar-pill {pill_cls}">● {status_lbl}</span>
    </div>
</div>""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">PITCH VERDICT</div>
    <div class="hero-sub">AI-Verified Tactical Match Reports</div>
    <div class="hero-desc">Every AI can write a match report. Only Pitch Verdict checks if it's true — extracting every number, cross-referencing it against StatsBomb source data, and flagging mismatches before you see a word.</div>
    <div class="hero-pills">
        <span class="hero-pill">StatsBomb Open Data</span>
        <span class="hero-pill">PPDA · xG · Possession</span>
        <span class="hero-pill">5-Agent Pipeline</span>
        <span class="hero-pill">Claim-by-Claim Verification</span>
        <span class="hero-pill">xAI Grok</span>
    </div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)


# ── Match Selection ───────────────────────────────────────────────
from agents.retriever import SAMPLE_MATCHES

if 'selected_match' not in st.session_state:
    st.session_state['selected_match'] = 'euro2024_final'

if 'result' not in st.session_state:

    st.markdown('<div class="sec-label">Select a Match to Analyse</div>', unsafe_allow_html=True)

    keys = list(SAMPLE_MATCHES.keys())
    row1 = st.columns(3, gap="medium")
    _, c4, c5, _ = st.columns([1, 3, 3, 1], gap="medium")
    slots = [row1[0], row1[1], row1[2], c4, c5]

    for i, key in enumerate(keys):
        m = SAMPLE_MATCHES[key]
        selected = (st.session_state['selected_match'] == key)
        card_cls = "match-card match-card-sel" if selected else "match-card"
        with slots[i]:
            st.markdown(f"""
            <div class="{card_cls}">
                <div class="match-comp">{m['competition']}</div>
                <div class="match-score-row">
                    <div class="match-team">{m['home_team']}</div>
                    <div class="match-score">{m['home_score']}–{m['away_score']}</div>
                    <div class="match-team" style="text-align:right;">{m['away_team']}</div>
                </div>
                <div class="match-meta">📅 {m['match_date']} &nbsp;·&nbsp; 🏟 {m['stadium'].split(',')[0]}</div>
                <div class="match-narrative">{m['narrative']}</div>
            </div>""", unsafe_allow_html=True)
            if st.button(
                "✓ Selected" if selected else "Select",
                key=f"sel_{key}",
                use_container_width=True,
                type="primary" if selected else "secondary"
            ):
                st.session_state['selected_match'] = key
                st.rerun()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Custom file upload ──
    with st.expander("📂  Or upload your own match JSON file", expanded=False):
        st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:var(--muted);line-height:1.8;">Upload a match file in Pitch Verdict JSON format. The file must have a <code>metadata</code> block and an <code>events</code> list. Download the sample file (mancity_vs_liverpool.json) to see the exact format expected.</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop your match JSON here", type=["json"], key="custom_match_file", label_visibility="collapsed")
        if uploaded_file:
            st.success(f"✅ Loaded: **{uploaded_file.name}** ({uploaded_file.size // 1024} KB)")
            st.session_state['custom_file_data'] = uploaded_file.read().decode('utf-8')
            st.session_state['custom_file_name'] = uploaded_file.name
            st.session_state['selected_match'] = '__custom__'
        else:
            if st.session_state.get('selected_match') == '__custom__':
                st.session_state['selected_match'] = 'euro2024_final'
            st.session_state.pop('custom_file_data', None)

    # ── Run button ──
    is_custom = st.session_state.get('selected_match') == '__custom__'
    if is_custom:
        btn_label = "▶  Analyse  Custom Match"
    else:
        sel = SAMPLE_MATCHES[st.session_state['selected_match']]
        btn_label = f"▶  Analyse  {sel['home_team']} vs {sel['away_team']}"

    run_col, info_col = st.columns([2, 5], gap="medium")
    with run_col:
        run_clicked = st.button(btn_label, use_container_width=True, type="primary")
    with info_col:
        if is_custom:
            fname = st.session_state.get('custom_file_name', 'custom.json')
            st.markdown(f'<div class="run-bar"><span style="color:var(--text);">Custom file:</span> <span style="color:var(--gold);">{fname}</span></div>', unsafe_allow_html=True)
        else:
            sel = SAMPLE_MATCHES[st.session_state['selected_match']]
            st.markdown(f'<div class="run-bar"><span style="color:var(--text);">Selected:</span> {sel["label"]} &nbsp;·&nbsp; <span style="color:var(--gold);">{sel["home_team"]} {sel["home_score"]}–{sel["away_score"]} {sel["away_team"]}</span></div>', unsafe_allow_html=True)

else:
    run_clicked = False
    if st.button("← New Analysis", type="secondary"):
        st.session_state.pop('result', None)
        st.rerun()


# ── Pipeline Run ─────────────────────────────────────────────────
if 'result' not in st.session_state and run_clicked:
    agent_labels = [
        "Retriever — Loading event data",
        "Phase Segmenter — Tactical chapters",
        "Tactical Classifier — PPDA · xG · Possession",
        "Writer — Generating report via Grok",
        "Verifier — Fact-checking every claim",
    ]
    left_col, right_col = st.columns([3, 1], gap="medium")
    with right_col:
        st.markdown('<div class="sec-hdr">PIPELINE STATUS</div>', unsafe_allow_html=True)
        pipe_ph = st.empty()
    with left_col:
        prog = st.progress(0, text="Initialising…")

    try:
        from pipeline import PitchVerdictPipeline
        pipe = PitchVerdictPipeline(verbose=False)
        sample_key = st.session_state.get('selected_match', 'euro2024_final')

        with pipe_ph.container(): render_pipeline(0, agent_labels)
        prog.progress(10, text="Agent 1: Loading match data…")
        # Load from custom file or sample
        is_custom = st.session_state.get('selected_match') == '__custom__'
        if is_custom and st.session_state.get('custom_file_data'):
            import tempfile, os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                tmp.write(st.session_state['custom_file_data'])
                tmp_path = tmp.name
            match_data = pipe.retriever.load_match(local_path=tmp_path)
            os.unlink(tmp_path)
        else:
            match_data = pipe.retriever.load_match(use_sample=True, sample_key=sample_key)
        pipe.retriever.get_match_summary(match_data)

        with pipe_ph.container(): render_pipeline(1, agent_labels)
        prog.progress(28, text="Agent 2: Segmenting phases…")
        segmentation = pipe.segmenter.segment(match_data)

        with pipe_ph.container(): render_pipeline(2, agent_labels)
        prog.progress(48, text="Agent 3: Classifying tactics…")
        tactical_analysis = pipe.classifier.classify(segmentation)
        structured_data = pipe.classifier.get_structured_data_for_writer(tactical_analysis, segmentation)

        match_meta = {
            'home_team': match_data.home_team, 'away_team': match_data.away_team,
            'home_score': match_data.home_score, 'away_score': match_data.away_score,
            'competition': match_data.competition, 'season': match_data.season,
            'match_date': match_data.match_date, 'stadium': match_data.stadium,
        }

        with pipe_ph.container(): render_pipeline(3, agent_labels)
        prog.progress(68, text="Agent 4: Writing report…")
        writer_output = pipe.writer.write_report(structured_data, match_meta)

        with pipe_ph.container(): render_pipeline(4, agent_labels)
        prog.progress(88, text="Agent 5: Verifying claims…")
        verification = pipe.verifier.verify(writer_output, tactical_analysis.ground_truth, match_meta)

        with pipe_ph.container(): render_pipeline(5, agent_labels)
        prog.progress(100, text="✅ Complete")

        st.session_state['result'] = {
            'match_data': match_data, 'segmentation': segmentation,
            'tactical_analysis': tactical_analysis, 'structured_data': structured_data,
            'writer_output': writer_output, 'verification': verification, 'match_meta': match_meta,
        }
        st.rerun()

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        import traceback; st.code(traceback.format_exc())


# ── Results ───────────────────────────────────────────────────────
if 'result' in st.session_state:
    r = st.session_state['result']
    md = r['match_data']; seg = r['segmentation']; ta = r['tactical_analysis']
    ver = r['verification']; mm = r['match_meta']; sd = r['structured_data']; wo = r['writer_output']
    home, away = md.home_team, md.away_team

    st.markdown(f"""
    <div class="result-hdr">
        <div class="result-comp">{md.competition} &nbsp;·&nbsp; {md.match_date} &nbsp;·&nbsp; {md.stadium}</div>
        <div class="result-teams">
            <div class="result-team">{home}</div>
            <div class="result-score">{md.home_score} — {md.away_score}</div>
            <div class="result-team" style="text-align:right;">{away}</div>
        </div>
        <div class="result-model">Model: {wo.model_used} &nbsp;·&nbsp; Claims checked: {ver.total_claims} &nbsp;·&nbsp; Accuracy: {ver.factual_accuracy:.1%}</div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Report", "🔍 Verification", "📊 Phase Analysis", "⚽ Key Events", "🧪 Adversarial Test"])

    with tab1:
        rep_col, stat_col = st.columns([3, 1], gap="medium")
        with stat_col:
            fm = sd.get('full_match_metrics', {}); hm = fm.get('home', {}); am = fm.get('away', {})
            st.markdown('<div class="sec-hdr">FULL MATCH STATS</div>', unsafe_allow_html=True)
            df_s = pd.DataFrame({'Metric': ['Possession','Pass Acc.','Shots','xG','PPDA'],
                home: [f"{hm.get('possession_pct',0):.1f}%", f"{hm.get('pass_completion_pct',0):.1f}%", str(hm.get('shots',0)), f"{hm.get('xg',0):.2f}", f"{hm.get('ppda',99):.1f}"],
                away: [f"{am.get('possession_pct',0):.1f}%", f"{am.get('pass_completion_pct',0):.1f}%", str(am.get('shots',0)), f"{am.get('xg',0):.2f}", f"{am.get('ppda',99):.1f}"],
            }).set_index('Metric')
            st.dataframe(df_s, use_container_width=True)
            st.markdown('<div class="sec-hdr">TACTICAL PROFILE</div>', unsafe_allow_html=True)
            if ta.overall_home_profile:
                st.markdown(f"**{home}**"); render_ppda_chip(ta.overall_home_profile.ppda)
                st.caption(f"{ta.overall_home_profile.buildup_pattern} · {ta.overall_home_profile.territorial_dominance}")
            if ta.overall_away_profile:
                st.markdown(f"**{away}**"); render_ppda_chip(ta.overall_away_profile.ppda)
                st.caption(f"{ta.overall_away_profile.buildup_pattern} · {ta.overall_away_profile.territorial_dominance}")
            st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
            render_accuracy(ver.factual_accuracy)
        with rep_col:
            st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.8rem;">Verified Match Report · Pitch Verdict · {wo.model_used}</div>', unsafe_allow_html=True)
            st.markdown(ver.revised_report)

    with tab2:
        st.markdown('<div class="sec-hdr">VERIFICATION DASHBOARD</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Claims", ver.total_claims)
        c2.metric("Verified ✅", ver.verified_claims, delta=f"{ver.factual_accuracy:.1%}")
        c3.metric("Flagged ⚠️", ver.flagged_claims)
        c4.metric("Uncertain ❓", ver.uncertain_claims)
        render_accuracy(ver.factual_accuracy)
        st.markdown("""<div style="font-family:'Source Serif 4',serif;font-style:italic;font-size:.88rem;color:var(--muted);margin:1rem 0;padding:.85rem 1rem;background:rgba(201,168,76,.05);border-left:3px solid var(--gold);border-radius:4px;">
        Every number in the report was extracted via regex, cross-referenced against StatsBomb source data, and either confirmed or flagged — before you saw a word. This is what separates Pitch Verdict from every other AI reporting tool.
        </div>""", unsafe_allow_html=True)
        st.markdown("**Claim-by-Claim Results**")
        if ver.results:
            for res in ver.results:
                icon = "✅" if res.status=="VERIFIED" else "⚠️" if res.status=="FLAGGED" else "❓"
                ic, cc = st.columns([0.4, 9.6])
                with ic: st.write(icon)
                with cc:
                    with st.expander(f"[{res.claim_type.upper()}] {res.explanation[:110]}…"):
                        a,b,c = st.columns(3)
                        a.markdown(f"**Type:** `{res.claim_type}`"); b.markdown(f"**Team:** {res.team or 'N/A'}"); c.markdown(f"**Status:** `{res.status}`")
                        st.code(res.claim_text[:280], language=None)
                        x,y,z = st.columns(3)
                        x.metric("Report says", res.extracted_value); y.metric("Data says", res.expected_value); z.metric("Deviation", f"{res.deviation:.3f}")
        else:
            st.info("No quantitative claims found. With a Grok API key, the generated report will have many more numbers to verify.")

    with tab3:
        st.markdown('<div class="sec-hdr">PHASE-BY-PHASE BREAKDOWN</div>', unsafe_allow_html=True)
        phases = seg.phases
        ca, cb = st.columns(2, gap="medium")
        with ca:
            f = make_chart(phases, home, away, "possession")
            if f: st.plotly_chart(f, use_container_width=True)
        with cb:
            f = make_chart(phases, home, away, "ppda")
            if f: st.plotly_chart(f, use_container_width=True)
        f = make_chart(phases, home, away, "xg")
        if f: st.plotly_chart(f, use_container_width=True)
        rows = [{'Phase': p.name, 'Min': f"{p.start_minute}'–{p.end_minute}'",
            f'{home} Poss': f"{p.home_possession:.1f}%", f'{away} Poss': f"{p.away_possession:.1f}%",
            f'{home} PPDA': f"{p.home_ppda:.1f}", f'{away} PPDA': f"{p.away_ppda:.1f}",
            f'{home} xG': f"{p.home_xg:.2f}", f'{away} xG': f"{p.away_xg:.2f}",
            f'{home} Press': ppda_label(p.home_ppda), f'{away} Press': ppda_label(p.away_ppda),
        } for p in phases]
        if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        themes = sd.get('key_tactical_themes', [])
        if themes:
            st.markdown('<div class="sec-hdr">KEY TACTICAL THEMES</div>', unsafe_allow_html=True)
            for t in themes: st.markdown(f'<div class="key-moment">🎯 {t}</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="sec-hdr">MATCH TIMELINE</div>', unsafe_allow_html=True)
        for ev in sorted(seg.key_events, key=lambda e: e['minute']):
            etype = ev.get('type','')
            icon = {"Goal":"⚽","Substitution":"🔄","Halftime":"🔔"}.get(etype,"📌")
            color = "#c9a84c" if etype=="Goal" else "#8b949e" if etype=="Halftime" else "#2980b9"
            xg_str = f"&nbsp;·&nbsp; xG: {ev['xg']:.2f}" if etype=="Goal" and 'xg' in ev else ""
            st.markdown(f'<div class="timeline-ev" style="border-left:3px solid {color};"><span style="color:{color};font-size:1rem;">{icon}</span><span style="color:var(--muted);min-width:30px;">{ev["minute"]}\'</span><span style="font-weight:600;color:var(--text);">{ev.get("description","")}</span><span style="color:var(--muted);">{xg_str}</span></div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="sec-hdr">ADVERSARIAL VERIFICATION TEST</div>', unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Source Serif 4',serif;font-size:.9rem;line-height:1.7;color:var(--muted);margin-bottom:1.2rem;">
        We deliberately inject wrong statistics into the generated report and re-run the Verifier.
        A system that only generates would silently pass these errors to the reader. This test proves the Verifier catches them.
        </div>""", unsafe_allow_html=True)
        if st.button("🧪 Inject Errors and Test Verifier", type="secondary"):
            from agents.verifier import VerifierAgent as VA
            from agents.writer import WriterOutput as WO
            va = VA(verbose=False)
            corrupted, injected = va.inject_errors(wo.report_text, ta.ground_truth, home, away)
            st.markdown("**Injected errors:**")
            for err in injected:
                st.markdown(f'<div style="background:rgba(231,76,60,.1);border:1px solid #e74c3c;border-radius:6px;padding:.6rem 1rem;margin-bottom:.4rem;font-family:\'JetBrains Mono\',monospace;font-size:.76rem;">⚠️ <b>{err["type"].upper()}</b> ({err.get("team","?")}): changed <code>{err["real"]}</code> → <code>{err["injected"]}</code></div>', unsafe_allow_html=True)
            adv_ver = va.verify(WO(report_text=corrupted, match_info=mm, structured_data=sd, model_used='adversarial'), ta.ground_truth, mm)
            caught = adv_ver.flagged_claims; total = len(injected)
            rate = caught / total if total else 0
            color = "#27ae60" if rate >= .9 else "#f39c12" if rate >= .5 else "#e74c3c"
            st.markdown(f'<div style="background:rgba(0,0,0,.3);border:2px solid {color};border-radius:10px;padding:1.3rem;text-align:center;margin:1rem 0;font-family:\'Bebas Neue\',cursive;font-size:1.9rem;letter-spacing:.05em;color:{color};">CAUGHT {caught} / {total} ERRORS — {rate:.0%} CATCH RATE</div>', unsafe_allow_html=True)
            for res in adv_ver.results:
                if res.is_flagged: st.error(f"⚠️ CAUGHT: {res.explanation}")