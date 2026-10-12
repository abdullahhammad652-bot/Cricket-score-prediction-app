import os
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="T20 Score Predictor", page_icon="🏏", layout="wide")

# ====================== STYLING ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;800&display=swap');
html, body, [class*="css"] {font-family:'Poppins',sans-serif;}
.stApp {background: radial-gradient(circle at 12% 8%, #11333f 0%, #0a1620 42%, #060b11 100%);}
#MainMenu, footer, header {visibility:hidden;}
.block-container {padding-top:1.2rem; max-width:1200px;}

/* sidebar */
section[data-testid="stSidebar"] {background:#08121a; border-right:1px solid rgba(255,255,255,.07);}
section[data-testid="stSidebar"] * {color:#cbd5e1;}
.side-logo {display:flex; align-items:center; gap:10px; margin-bottom:6px;}
.side-logo .ic {font-size:2rem;}
.side-logo .t {font-weight:800; font-size:1.15rem; color:#fff; line-height:1.1;}
.side-logo .s {font-size:.72rem; color:#64748b; letter-spacing:1.5px; text-transform:uppercase;}
.side-h {color:#5eead4 !important; font-weight:600; font-size:.8rem; letter-spacing:2px;
         text-transform:uppercase; margin:18px 0 4px 0;}

/* tabs */
button[data-baseweb="tab"] {font-weight:600; font-size:1rem; color:#94a3b8;}
button[data-baseweb="tab"][aria-selected="true"] {color:#5eead4;}
div[data-baseweb="tab-highlight"] {background:#14b8a6 !important;}

/* hero */
.hero {background:linear-gradient(120deg,#0d9488,#14b8a6 50%,#22d3ee); border-radius:24px;
       padding:30px 36px; color:#fff; margin-bottom:22px; position:relative; overflow:hidden;
       box-shadow:0 14px 44px rgba(20,184,166,.28);}
.hero:after {content:"🏏"; position:absolute; right:30px; top:-6px; font-size:7.5rem; opacity:.16;}
.hero h1 {margin:0; font-size:2.3rem; font-weight:800; letter-spacing:-.5px;}
.hero p {margin:6px 0 0 0; opacity:.93;}

/* versus banner */
.vs {display:flex; align-items:center; justify-content:space-between; gap:12px;
     background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09);
     border-radius:20px; padding:18px 26px; margin-bottom:20px;}
.team {display:flex; align-items:center; gap:14px;}
.team.r {flex-direction:row-reverse; text-align:right;}
.badge {width:58px; height:58px; border-radius:50%; display:grid; place-items:center;
        color:#fff; font-weight:800; font-size:1rem; letter-spacing:.5px;
        border:3px solid rgba(255,255,255,.25); box-shadow:0 6px 18px rgba(0,0,0,.4);}
.team .n {color:#fff; font-weight:700; font-size:1.15rem; line-height:1.15;}
.team .r2 {color:#94a3b8; font-size:.75rem; letter-spacing:1.4px; text-transform:uppercase;}
.vs-mid {color:#5eead4; font-weight:800; font-size:1.3rem; padding:6px 16px;
         border:1.5px dashed rgba(94,234,212,.5); border-radius:999px;}

/* cards */
.card {background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09);
       border-radius:20px; padding:22px 24px;}
.sec {color:#e2e8f0; font-weight:600; font-size:1.1rem; margin:6px 0 10px 0;}

/* score ring */
.result {text-align:center; background:linear-gradient(160deg,rgba(20,184,166,.18),rgba(34,211,238,.06));
         border:1.5px solid rgba(45,212,191,.5); border-radius:24px; padding:26px 16px;}
.result .lab {color:#99f6e4; font-size:.8rem; letter-spacing:2.5px; text-transform:uppercase; margin-bottom:14px;}
.ring {width:230px; height:230px; border-radius:50%; margin:0 auto; display:grid; place-items:center;
       box-shadow:0 0 40px rgba(34,211,238,.25);}
.ring-in {width:188px; height:188px; border-radius:50%; background:#0a1822;
          display:flex; flex-direction:column; align-items:center; justify-content:center;}
.ring-in .sc {color:#fff; font-size:4.2rem; font-weight:800; line-height:1;}
.ring-in .un {color:#94a3b8; font-size:.9rem; margin-top:2px;}
.pill {display:inline-block; margin-top:16px; padding:6px 16px; border-radius:999px;
       background:rgba(255,255,255,.08); color:#ccfbf1; font-size:.85rem;}
.verdict {margin-top:10px; font-weight:600; font-size:1.05rem;}

/* stat cards */
.stat {background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
       border-radius:16px; padding:16px; text-align:center; height:100%;}
.stat .ic {font-size:1.5rem;}
.stat .v {color:#fff; font-size:1.7rem; font-weight:700; line-height:1.2;}
.stat .k {color:#94a3b8; font-size:.72rem; text-transform:uppercase; letter-spacing:1.2px;}
.up {color:#4ade80 !important;} .down {color:#f87171 !important;}

.chip {display:inline-block; background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
       border-radius:999px; padding:5px 14px; margin:3px 4px 3px 0; color:#cbd5e1; font-size:.82rem;}
.foot {color:#64748b; text-align:center; font-size:.8rem; margin-top:30px;}
label, .stMarkdown p {color:#cbd5e1;}
</style>
""", unsafe_allow_html=True)


def html(s: str) -> str:
    """Strip indentation so markdown never treats the HTML as a code block."""
    return "".join(line.strip() for line in s.splitlines())


# ====================== MODEL ======================
@st.cache_resource
def load_assets():
    base = os.path.dirname(os.path.abspath(__file__))
    return (
        joblib.load(os.path.join(base, "model.pkl")),
        joblib.load(os.path.join(base, "poly.pkl")),
        joblib.load(os.path.join(base, "columns.pkl")),
    )


model, poly, columns = load_assets()


def predict_many(params):
    """params: list of (overs, wickets, run_rate, opp, venue, pitch, weather)."""
    cols = list(zip(*params))
    df = pd.DataFrame(0.0, index=range(len(params)), columns=columns)
    df["Overs Played"] = cols[0]
    df["Wickets Lost"] = cols[1]
    df["Run Rate"] = cols[2]
    df["Opponent Strength"] = cols[3]
    df["Home/Away_Home"] = [int(v == "Home") for v in cols[4]]
    df["Pitch Condition_Batting"] = [int(p == "Batting") for p in cols[5]]
    df["Pitch Condition_Bowling"] = [int(p == "Bowling") for p in cols[5]]
    df["Weather_Overcast"] = [int(w == "Overcast") for w in cols[6]]
    df["Weather_Sunny"] = [int(w == "Sunny") for w in cols[6]]
    return np.maximum(model.predict(poly.transform(df)), 0)


def predict_one(*p):
    return float(predict_many([p])[0])


# ====================== TEAMS (display only) ======================
TEAMS = {
    "Pakistan": ("PAK", "#01411C"), "India": ("IND", "#1d4ed8"),
    "Australia": ("AUS", "#ca8a04"), "England": ("ENG", "#dc2626"),
    "South Africa": ("RSA", "#15803d"), "New Zealand": ("NZ", "#334155"),
    "Sri Lanka": ("SL", "#1e3a8a"), "Bangladesh": ("BAN", "#166534"),
    "West Indies": ("WI", "#7f1d1d"), "Afghanistan": ("AFG", "#0369a1"),
    "Lahore Qalandars": ("LQ", "#16a34a"), "Karachi Kings": ("KK", "#2563eb"),
    "Islamabad United": ("IU", "#dc2626"), "Peshawar Zalmi": ("PZ", "#ca8a04"),
    "Quetta Gladiators": ("QG", "#7c3aed"), "Multan Sultans": ("MS", "#0ea5e9"),
}


def team_html(name, role, right=False):
    code, color = TEAMS[name]
    cls = "team r" if right else "team"
    return (f'<div class="{cls}"><div class="badge" style="background:{color}">{code}</div>'
            f'<div><div class="n">{name}</div><div class="r2">{role}</div></div></div>')


# ====================== SIDEBAR INPUTS ======================
with st.sidebar:
    st.markdown(html("""
    <div class="side-logo"><div class="ic">🏏</div>
    <div><div class="t">T20 Predictor</div><div class="s">Score Analytics</div></div></div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="side-h">🏟️ Teams</div>', unsafe_allow_html=True)
    team_names = list(TEAMS.keys())
    bat_team = st.selectbox("Batting team", team_names, index=0)
    bowl_team = st.selectbox("Bowling team", team_names, index=1)

    st.markdown('<div class="side-h">🎛️ Match situation</div>', unsafe_allow_html=True)
    overs = st.slider("Overs played", 1, 20, 10)
    wickets = st.slider("Wickets lost", 0, 10, 3)
    run_rate = st.number_input("Current run rate", 4.0, 16.0, 8.5, step=0.1)
    opp = st.slider("Opponent strength (1-10)", 1, 10, 5)

    st.markdown('<div class="side-h">🌦️ Conditions</div>', unsafe_allow_html=True)
    venue = st.radio("Venue", ["Home", "Away"], horizontal=True,
                     format_func=lambda x: {"Home": "🏠 Home", "Away": "✈️ Away"}[x])
    pitch = st.selectbox("Pitch", ["Balanced", "Batting", "Bowling"],
                         format_func=lambda x: {"Balanced": "⚖️ Balanced", "Batting": "🏏 Batting friendly",
                                                "Bowling": "🎯 Bowling friendly"}[x])
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Overcast"],
                           format_func=lambda x: {"Sunny": "☀️ Sunny", "Cloudy": "☁️ Cloudy",
                                                  "Overcast": "🌧️ Overcast"}[x])

state = (overs, wickets, run_rate, opp, venue, pitch, weather)
score = predict_one(*state)

# ====================== HEADER ======================
st.markdown(html("""
<div class="hero"><h1>🏏 T20 Cricket Score Predictor</h1>
<p>Set the match situation in the sidebar and watch the predicted final score update instantly.</p></div>
"""), unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯  Predictor", "📊  Analysis", "ℹ️  About"])

# ====================== TAB 1: PREDICTOR ======================
with tab1:
    st.markdown(html(f"""
    <div class="vs">{team_html(bat_team, "Batting")}<div class="vs-mid">VS</div>
    {team_html(bowl_team, "Bowling", right=True)}</div>
    """), unsafe_allow_html=True)
    if bat_team == bowl_team:
        st.warning("Batting and bowling team are the same. Pick two different teams.")

    if score < 120:
        verdict, vcol = "Low-scoring innings", "#f87171"
    elif score < 170:
        verdict, vcol = "Competitive total", "#fbbf24"
    elif score < 210:
        verdict, vcol = "Strong total", "#4ade80"
    else:
        verdict, vcol = "Mammoth total", "#22d3ee"
    pct = min(score / 300 * 100, 100)

    left, right = st.columns([4, 5], gap="large")
    with left:
        st.markdown(html(f"""
        <div class="result"><div class="lab">Predicted final score</div>
        <div class="ring" style="background:conic-gradient(#22d3ee {pct}%, rgba(255,255,255,.08) 0)">
        <div class="ring-in"><div class="sc">{score:.0f}</div><div class="un">runs</div></div></div>
        <div class="pill">Likely range: {max(score-3, 0):.0f} – {score+3:.0f}</div>
        <div class="verdict" style="color:{vcol}">{verdict}</div></div>
        """), unsafe_allow_html=True)

    with right:
        st.markdown('<div class="sec">📌 Match snapshot</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        s1.markdown(html(f'<div class="stat"><div class="ic">⏱️</div><div class="v">{20-overs}</div><div class="k">Overs left</div></div>'), unsafe_allow_html=True)
        s2.markdown(html(f'<div class="stat"><div class="ic">🧤</div><div class="v">{10-wickets}</div><div class="k">Wickets in hand</div></div>'), unsafe_allow_html=True)
        s3.markdown(html(f'<div class="stat"><div class="ic">⚡</div><div class="v">{run_rate:.1f}</div><div class="k">Run rate</div></div>'), unsafe_allow_html=True)

        st.markdown('<div class="sec" style="margin-top:18px">🔥 What moves the score</div>', unsafe_allow_html=True)
        base = score
        d_rr = predict_one(overs, wickets, min(run_rate + 1, 16), opp, venue, pitch, weather) - base
        d_wk = predict_one(overs, min(wickets + 1, 10), run_rate, opp, venue, pitch, weather) - base
        d_ov = predict_one(min(overs + 1, 20), wickets, run_rate, opp, venue, pitch, weather) - base

        def delta_card(icon, label, d):
            cls = "up" if d >= 0 else "down"
            sign = "+" if d >= 0 else ""
            return html(f'<div class="stat"><div class="ic">{icon}</div><div class="v {cls}">{sign}{d:.0f}</div><div class="k">{label}</div></div>')

        d1, d2, d3 = st.columns(3)
        d1.markdown(delta_card("📈", "+1 run rate", d_rr), unsafe_allow_html=True)
        d2.markdown(delta_card("❌", "+1 wicket", d_wk), unsafe_allow_html=True)
        d3.markdown(delta_card("⏩", "+1 over", d_ov), unsafe_allow_html=True)

        st.markdown(html(f"""
        <div style="margin-top:18px">
        <span class="chip">{'🏠' if venue == 'Home' else '✈️'} {venue}</span>
        <span class="chip">🏟️ {pitch} pitch</span>
        <span class="chip">{ {'Sunny':'☀️','Cloudy':'☁️','Overcast':'🌧️'}[weather] } {weather}</span>
        <span class="chip">💪 Opponent {opp}/10</span></div>
        """), unsafe_allow_html=True)

# ====================== CHART HELPERS ======================
def style(chart):
    return (chart.configure(background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor="#94a3b8", titleColor="#cbd5e1", gridColor="#ffffff14",
                            domainColor="#ffffff22", tickColor="#ffffff22")
            .configure_legend(labelColor="#cbd5e1", titleColor="#cbd5e1"))


def area_line(df, x, y, xt, yt):
    base = alt.Chart(df).encode(x=alt.X(x, title=xt), y=alt.Y(y, title=yt))
    return style((base.mark_area(color="#22d3ee", opacity=0.15) +
                  base.mark_line(color="#22d3ee", strokeWidth=3, point=alt.OverlayMarkDef(color="#5eead4", size=45)))
                 .properties(height=270))


# ====================== TAB 2: ANALYSIS ======================
with tab2:
    st.markdown(html(f"""
    <div class="vs">{team_html(bat_team, "Batting")}<div class="vs-mid">{score:.0f} runs</div>
    {team_html(bowl_team, "Bowling", right=True)}</div>
    """), unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="sec">📈 Score vs run rate</div>', unsafe_allow_html=True)
        rates = [round(4 + 0.5 * i, 1) for i in range(25)]
        df_rr = pd.DataFrame({"Run rate": rates,
                              "Predicted score": predict_many([(overs, wickets, r, opp, venue, pitch, weather) for r in rates])})
        st.altair_chart(area_line(df_rr, "Run rate:Q", "Predicted score:Q", "Run rate", "Predicted score"),
                        use_container_width=True, theme=None)

    with c2:
        st.markdown('<div class="sec">⏱️ Score vs overs played</div>', unsafe_allow_html=True)
        ovs = list(range(1, 21))
        df_ov = pd.DataFrame({"Overs played": ovs,
                              "Predicted score": predict_many([(o, wickets, run_rate, opp, venue, pitch, weather) for o in ovs])})
        st.altair_chart(area_line(df_ov, "Overs played:Q", "Predicted score:Q", "Overs played", "Predicted score"),
                        use_container_width=True, theme=None)

    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown('<div class="sec">❌ Score vs wickets lost</div>', unsafe_allow_html=True)
        wk = list(range(0, 11))
        df_wk = pd.DataFrame({"Wickets lost": wk,
                              "Predicted score": predict_many([(overs, w, run_rate, opp, venue, pitch, weather) for w in wk])})
        bars = (alt.Chart(df_wk).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(x=alt.X("Wickets lost:O", title="Wickets lost", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("Predicted score:Q", title="Predicted score"),
                        color=alt.condition(alt.datum["Wickets lost"] == wickets,
                                            alt.value("#22d3ee"), alt.value("#14b8a655")),
                        tooltip=["Wickets lost", alt.Tooltip("Predicted score:Q", format=".0f")])
                .properties(height=270))
        st.altair_chart(style(bars), use_container_width=True, theme=None)

    with c4:
        st.markdown('<div class="sec">🌦️ Pitch × weather impact</div>', unsafe_allow_html=True)
        combos = [(p, w) for p in ["Balanced", "Batting", "Bowling"] for w in ["Sunny", "Cloudy", "Overcast"]]
        vals = predict_many([(overs, wickets, run_rate, opp, venue, p, w) for p, w in combos])
        df_pw = pd.DataFrame({"Pitch": [c[0] for c in combos], "Weather": [c[1] for c in combos], "Score": vals})
        heat = alt.Chart(df_pw).mark_rect(cornerRadius=8).encode(
            x=alt.X("Weather:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Pitch:N", title=None),
            color=alt.Color("Score:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
            tooltip=["Pitch", "Weather", alt.Tooltip("Score:Q", format=".0f")])
        text = alt.Chart(df_pw).mark_text(fontWeight="bold", fontSize=15).encode(
            x="Weather:N", y="Pitch:N", text=alt.Text("Score:Q", format=".0f"),
            color=alt.condition(alt.datum.Score > df_pw["Score"].median(), alt.value("#04202a"), alt.value("#ffffff")))
        st.altair_chart(style((heat + text).properties(height=270)), use_container_width=True, theme=None)

    st.markdown('<div class="sec">🔥 Overs × wickets heatmap (at current run rate)</div>', unsafe_allow_html=True)
    grid = [(o, w) for o in range(1, 21) for w in range(0, 11)]
    gv = predict_many([(o, w, run_rate, opp, venue, pitch, weather) for o, w in grid])
    df_g = pd.DataFrame({"Overs": [g[0] for g in grid], "Wickets": [g[1] for g in grid], "Score": gv})
    hm = (alt.Chart(df_g).mark_rect()
          .encode(x=alt.X("Overs:O", title="Overs played", axis=alt.Axis(labelAngle=0)),
                  y=alt.Y("Wickets:O", title="Wickets lost", sort="descending"),
                  color=alt.Color("Score:Q", scale=alt.Scale(scheme="viridis"), title="Score"),
                  tooltip=["Overs", "Wickets", alt.Tooltip("Score:Q", format=".0f")])
          .properties(height=330))
    st.altair_chart(style(hm), use_container_width=True, theme=None)
    st.caption("All charts change live when you move the sidebar controls.")

# ====================== TAB 3: ABOUT ======================
with tab3:
    a, b = st.columns([3, 2], gap="large")
    with a:
        st.markdown(html("""
        <div class="card"><div class="sec">🧠 How it works</div>
        <p>This app predicts the final T20 innings score from the current match situation.
        A <b>Linear Regression</b> model with <b>degree-2 polynomial features</b> was trained on 1,500 matches.</p>
        <p><b>Inputs used by the model:</b> overs played, wickets lost, run rate, opponent strength,
        venue, pitch condition and weather.</p>
        <p><b>Pipeline:</b> missing values dropped → outliers handled with the IQR rule (median) →
        categorical columns one-hot encoded → 80/20 train/test split → model training and evaluation.</p>
        <p style="color:#94a3b8">Note: team names are for display only. The model was not trained on team identity.</p></div>
        """), unsafe_allow_html=True)
    with b:
        m1, m2 = st.columns(2)
        m1.markdown(html('<div class="stat"><div class="ic">🎯</div><div class="v">0.997</div><div class="k">R² (poly)</div></div>'), unsafe_allow_html=True)
        m2.markdown(html('<div class="stat"><div class="ic">📏</div><div class="v">2.5</div><div class="k">MAE (runs)</div></div>'), unsafe_allow_html=True)
        st.write("")
        m3, m4 = st.columns(2)
        m3.markdown(html('<div class="stat"><div class="ic">📉</div><div class="v">0.915</div><div class="k">R² (linear)</div></div>'), unsafe_allow_html=True)
        m4.markdown(html('<div class="stat"><div class="ic">📏</div><div class="v">15.6</div><div class="k">MAE (linear)</div></div>'), unsafe_allow_html=True)

st.markdown('<div class="foot">Built with Python · scikit-learn · Streamlit</div>', unsafe_allow_html=True)