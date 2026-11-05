"""Streamlit dashboard for the Quantitative Finance AI Copilot.

Run:  streamlit run dashboard.py
"""
from __future__ import annotations

import sys
import os

# Allow imports from app/
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Risk Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── colour tokens ────────────────────────────────────────────────────────────
BLUE   = "#4A90D9"
RED    = "#E05C5C"
GREEN  = "#5CB85C"
AMBER  = "#F0A500"
GREY   = "#6C757D"
BG     = "#0E1117"
CARD   = "#1A1F2E"
TEXT   = "#E0E0E0"
BORDER = "#2D3348"

ASSET_COLOURS = {
    "NIFTY50":   BLUE,
    "GOVT_BOND": GREEN,
    "GOLD":      AMBER,
    "USD_INR":   "#A78BFA",
}

# ── data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
PORTFOLIO_CSV   = os.environ.get("PORTFOLIO_CSV",    os.path.join(DATA_DIR, "sample_portfolio.csv"))
ASSET_RET_CSV   = os.environ.get("ASSET_RETURNS_CSV",os.path.join(DATA_DIR, "sample_asset_returns.csv"))
BASE_CURRENCY   = os.environ.get("BASE_CURRENCY", "INR")

# ── load data (cached) ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    from app.portfolio.portfolio_engine import (
        load_portfolio, load_asset_returns,
        weights_from_portfolio, calculate_portfolio_returns, covariance_matrix,
    )
    portfolio  = load_portfolio(PORTFOLIO_CSV)
    ret_df     = load_asset_returns(ASSET_RET_CSV)
    weights    = weights_from_portfolio(portfolio)
    port_ret   = calculate_portfolio_returns(ret_df, weights)
    cov        = covariance_matrix(ret_df)
    return portfolio, ret_df, weights, port_ret, cov

# ── helper: metric card ──────────────────────────────────────────────────────
def metric(label: str, value: str, delta: str = "", colour: str = TEXT):
    st.markdown(
        f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;
            padding:14px 18px;margin-bottom:8px;">
            <div style="font-size:11px;color:{GREY};text-transform:uppercase;
                letter-spacing:.8px;">{label}</div>
            <div style="font-size:22px;font-weight:700;color:{colour};margin:4px 0 2px;">{value}</div>
            <div style="font-size:11px;color:{GREY};">{delta}</div>
        </div>""",
        unsafe_allow_html=True,
    )

# ── helper: plotly defaults ───────────────────────────────────────────────────
def _fig_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT)),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=TEXT, size=11),
        margin=dict(l=50, r=20, t=40 if title else 20, b=40),
        height=height,
        legend=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(size=11)),
        xaxis=dict(gridcolor=BORDER, showgrid=True),
        yaxis=dict(gridcolor=BORDER, showgrid=True),
    )
    return fig

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f"""<div style="font-size:20px;font-weight:800;color:{BLUE};margin-bottom:4px;">
        📊 Quant Risk Copilot</div>
        <div style="font-size:11px;color:{GREY};margin-bottom:20px;">
        Multi-asset Portfolio Risk Engine</div>""",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["Portfolio Overview", "VaR & ES", "Risk Attribution",
         "Stress Testing", "Volatility Models", "Backtesting"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    confidence = st.select_slider(
        "Confidence level", options=[0.90, 0.95, 0.99], value=0.99,
        format_func=lambda x: f"{int(x*100)}%",
    )
    horizon = st.select_slider(
        "Horizon (days)", options=[1, 5, 10, 20], value=1,
    )
    st.markdown("---")
    st.markdown(
        f"<div style='font-size:10px;color:{GREY};'>Assets: NIFTY50 · GOVT_BOND · GOLD · USD/INR<br>"
        f"Currency: {BASE_CURRENCY} · 756 trading days</div>",
        unsafe_allow_html=True,
    )

# ── load ─────────────────────────────────────────────────────────────────────
try:
    portfolio, ret_df, weights, port_ret, cov = load_data()
except FileNotFoundError:
    st.error("⚠️  Sample data not found. Run `python data/generate_sample_data.py` first.")
    st.stop()

assets       = list(weights.keys())
total_value  = float(portfolio["market_value"].sum())

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PORTFOLIO OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "Portfolio Overview":
    st.markdown(f"## Portfolio Overview")

    # ── top metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    ann_vol  = float(port_ret.std() * np.sqrt(252))
    ann_ret  = float(port_ret.mean() * 252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd   = float((port_ret.cumsum() - port_ret.cumsum().cummax()).min())

    with c1: metric("Total Value",     f"₹{total_value/1e6:.2f} M",  f"{len(assets)} assets", BLUE)
    with c2: metric("Ann. Return",     f"{ann_ret*100:.2f}%",         "252-day annualised", GREEN if ann_ret > 0 else RED)
    with c3: metric("Ann. Volatility", f"{ann_vol*100:.2f}%",         "1-day σ × √252", AMBER)
    with c4: metric("Sharpe Ratio",    f"{sharpe:.2f}",               "Return / Vol", GREEN if sharpe > 1 else AMBER)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 1])

    # ── cumulative returns chart ──────────────────────────────────────────────
    with col_l:
        fig = go.Figure()
        cum_port = (1 + port_ret).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum_port.index, y=cum_port.values * 100,
            name="Portfolio", line=dict(color=BLUE, width=2), fill="tozeroy",
            fillcolor="rgba(74,144,217,0.08)",
        ))
        for a in assets:
            s = ret_df[a]
            cum_a = (1 + s).cumprod() - 1
            fig.add_trace(go.Scatter(
                x=cum_a.index, y=cum_a.values * 100,
                name=a, line=dict(color=ASSET_COLOURS.get(a, GREY), width=1, dash="dot"),
                opacity=0.65,
            ))
        fig.update_yaxes(title_text="Cumulative Return (%)")
        _fig_layout(fig, "Cumulative Returns", 350)
        st.plotly_chart(fig, use_container_width=True)

    # ── allocation pie ────────────────────────────────────────────────────────
    with col_r:
        fig2 = go.Figure(go.Pie(
            labels=assets,
            values=[portfolio.set_index("asset").loc[a, "market_value"] for a in assets],
            hole=0.5,
            marker_colors=[ASSET_COLOURS.get(a, GREY) for a in assets],
            textinfo="label+percent", textfont_size=12,
        ))
        _fig_layout(fig2, "Allocation", 350)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── rolling volatility ────────────────────────────────────────────────────
    roll_vol = port_ret.rolling(21).std() * np.sqrt(252) * 100
    fig3 = go.Figure(go.Scatter(
        x=roll_vol.index, y=roll_vol,
        line=dict(color=AMBER, width=1.5), fill="tozeroy",
        fillcolor="rgba(240,165,0,0.08)",
    ))
    fig3.update_yaxes(title_text="Annualised Vol (%)")
    _fig_layout(fig3, "21-Day Rolling Volatility", 240)
    st.plotly_chart(fig3, use_container_width=True)

    # ── return distribution ───────────────────────────────────────────────────
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(
        x=port_ret.values * 100, nbinsx=60,
        marker_color=BLUE, opacity=0.75, name="Daily returns",
    ))
    fig4.update_xaxes(title_text="Daily Return (%)")
    fig4.update_yaxes(title_text="Frequency")
    _fig_layout(fig4, "Return Distribution", 240)
    st.plotly_chart(fig4, use_container_width=True)

    # ── correlation heatmap ───────────────────────────────────────────────────
    corr = ret_df[assets].corr()
    fig5 = go.Figure(go.Heatmap(
        z=corr.values, x=assets, y=assets,
        colorscale=[[0, RED], [0.5, CARD], [1, BLUE]],
        zmid=0, text=corr.round(2).values, texttemplate="%{text}",
        textfont_size=12, showscale=True,
    ))
    _fig_layout(fig5, "Asset Correlation Matrix", 320)
    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — VaR & ES
# ════════════════════════════════════════════════════════════════════════════
elif page == "VaR & ES":
    from app.portfolio.risk_metrics import historical_var_es, parametric_var_es, scale_horizon

    st.markdown("## Value at Risk & Expected Shortfall")

    h_var = historical_var_es(port_ret, total_value, confidence)
    p_var = parametric_var_es(port_ret, total_value, confidence)

    h_var_sc = scale_horizon(h_var["var"], horizon)
    p_var_sc = scale_horizon(p_var["var"], horizon)
    h_es_sc  = scale_horizon(h_var["es"],  horizon)
    p_es_sc  = scale_horizon(p_var["es"],  horizon)

    conf_lbl = f"{int(confidence*100)}%"
    hor_lbl  = f"{horizon}-day"

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric(f"Historical VaR ({conf_lbl}, {hor_lbl})", f"₹{h_var_sc/1e3:.1f}K",
                    f"{h_var['var_pct']*100:.2f}% of portfolio", RED)
    with c2: metric(f"Historical ES ({conf_lbl}, {hor_lbl})",  f"₹{h_es_sc/1e3:.1f}K",
                    "Mean tail loss", RED)
    with c3: metric(f"Parametric VaR ({conf_lbl}, {hor_lbl})", f"₹{p_var_sc/1e3:.1f}K",
                    f"{p_var['var_pct']*100:.2f}% of portfolio", AMBER)
    with c4: metric(f"Parametric ES ({conf_lbl}, {hor_lbl})",  f"₹{p_es_sc/1e3:.1f}K",
                    "Normal distribution", AMBER)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── loss distribution with VaR lines ─────────────────────────────────────
    losses = -port_ret.values * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=losses, nbinsx=60, marker_color=BLUE, opacity=0.6, name="Daily losses",
    ))
    fig.add_vline(x=h_var["var_pct"]*100, line=dict(color=RED, width=2, dash="solid"),
                  annotation_text=f"Hist VaR {conf_lbl}", annotation_position="top right",
                  annotation_font_color=RED)
    fig.add_vline(x=h_var["es_pct"]*100,  line=dict(color=RED, width=1.5, dash="dash"),
                  annotation_text=f"Hist ES", annotation_position="top right",
                  annotation_font_color=RED)
    fig.add_vline(x=p_var["var_pct"]*100, line=dict(color=AMBER, width=2, dash="solid"),
                  annotation_text=f"Param VaR {conf_lbl}", annotation_position="top left",
                  annotation_font_color=AMBER)
    fig.update_xaxes(title_text="Daily Loss (% of portfolio)")
    fig.update_yaxes(title_text="Frequency")
    _fig_layout(fig, "Loss Distribution", 350)
    st.plotly_chart(fig, use_container_width=True)

    # ── VaR sensitivity table ─────────────────────────────────────────────────
    st.markdown("#### Sensitivity across confidence levels")
    rows = []
    for c in [0.90, 0.95, 0.99]:
        hv = historical_var_es(port_ret, total_value, c)
        pv = parametric_var_es(port_ret, total_value, c)
        rows.append({
            "Confidence": f"{int(c*100)}%",
            "Hist VaR (₹K)":  f"{hv['var']/1e3:.1f}",
            "Hist ES (₹K)":   f"{hv['es']/1e3:.1f}",
            "Param VaR (₹K)": f"{pv['var']/1e3:.1f}",
            "Param ES (₹K)":  f"{pv['es']/1e3:.1f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── horizon scaling ───────────────────────────────────────────────────────
    st.markdown("#### Horizon scaling (√t approximation)")
    hrows = []
    for h in [1, 5, 10, 20]:
        hrows.append({
            "Horizon": f"{h}d",
            "Hist VaR (₹K)":  f"{scale_horizon(h_var['var'],h)/1e3:.1f}",
            "Hist ES (₹K)":   f"{scale_horizon(h_var['es'],h)/1e3:.1f}",
            "Param VaR (₹K)": f"{scale_horizon(p_var['var'],h)/1e3:.1f}",
            "Param ES (₹K)":  f"{scale_horizon(p_var['es'],h)/1e3:.1f}",
        })
    st.dataframe(pd.DataFrame(hrows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — RISK ATTRIBUTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "Risk Attribution":
    from app.portfolio.risk_attribution import calculate_risk_contribution, concentration_metrics

    st.markdown("## Euler Risk Attribution")

    rc   = calculate_risk_contribution(weights, cov)
    conc = concentration_metrics(weights, cov)

    c1, c2, c3 = st.columns(3)
    port_vol = sum(rc[a]["risk_contribution"] for a in assets)
    with c1: metric("Portfolio Volatility", f"{port_vol*100:.4f}%", "1-day", BLUE)
    with c2: metric("Diversification Ratio", f"{conc['diversification_ratio']:.3f}",
                    "Weighted vol / Port vol", GREEN)
    with c3: metric("Weight HHI", f"{conc['weight_hhi']:.3f}",
                    "1/N=equal, 1=concentrated", AMBER)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── risk contribution bar chart ───────────────────────────────────────────
    rc_df = pd.DataFrame([
        {"Asset": a, "Weight %": rc[a]["weight"]*100,
         "Risk Contrib %": rc[a]["percentage"],
         "Marginal Contrib": rc[a]["marginal_contribution"]}
        for a in assets
    ])

    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rc_df["Asset"], y=rc_df["Weight %"],
            name="Weight", marker_color=[ASSET_COLOURS.get(a, GREY) for a in assets],
            opacity=0.75,
        ))
        fig.add_trace(go.Bar(
            x=rc_df["Asset"], y=rc_df["Risk Contrib %"],
            name="Risk Contribution", marker_color=[ASSET_COLOURS.get(a, GREY) for a in assets],
            opacity=1.0,
        ))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title_text="%")
        _fig_layout(fig, "Weight vs Risk Contribution", 350)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = go.Figure(go.Pie(
            labels=rc_df["Asset"],
            values=rc_df["Risk Contrib %"],
            hole=0.5,
            marker_colors=[ASSET_COLOURS.get(a, GREY) for a in assets],
            textinfo="label+percent", textfont_size=12,
        ))
        _fig_layout(fig2, "Risk Contribution Share", 350)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── table ─────────────────────────────────────────────────────────────────
    st.markdown("#### Euler decomposition")
    st.dataframe(
        rc_df.round(4).rename(columns={
            "Weight %": "Weight (%)",
            "Risk Contrib %": "Risk Contrib (%)",
            "Marginal Contrib": "Marginal Contrib (σ)",
        }),
        use_container_width=True, hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — STRESS TESTING
# ════════════════════════════════════════════════════════════════════════════
elif page == "Stress Testing":
    from app.portfolio.stress_testing import run_stress_test, DEFAULT_SCENARIOS

    st.markdown("## Scenario Stress Testing")

    market_values  = dict(zip(portfolio["asset"], portfolio["market_value"]))
    asset_classes  = dict(zip(portfolio["asset"], portfolio["asset_class"]))
    results        = run_stress_test(market_values, asset_classes)

    c_cols = st.columns(len(results))
    for i, r in enumerate(results):
        colour = GREEN if r["pnl"] >= 0 else RED
        sign   = "+" if r["pnl"] >= 0 else ""
        with c_cols[i]:
            metric(r["scenario"], f"{sign}{r['pnl_pct']:.1f}%",
                   f"₹{r['pnl']/1e3:.0f}K P&L", colour)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── scenario P&L bar ─────────────────────────────────────────────────────
    fig = go.Figure()
    for r in results:
        pnl_vals = [r["pnl_by_asset"][a]/1e3 for a in assets]
        fig.add_trace(go.Bar(
            x=assets, y=pnl_vals,
            name=r["scenario"],
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="P&L (₹K)")
    _fig_layout(fig, "Scenario P&L by Asset", 380)
    st.plotly_chart(fig, use_container_width=True)

    # ── waterfall for selected scenario ──────────────────────────────────────
    sel = st.selectbox("Waterfall for scenario", [r["scenario"] for r in results])
    sel_r = next(r for r in results if r["scenario"] == sel)
    asset_pnl = [(a, sel_r["pnl_by_asset"][a]/1e3) for a in assets]

    fig2 = go.Figure(go.Waterfall(
        x=[a for a, _ in asset_pnl] + ["Total"],
        y=[v for _, v in asset_pnl] + [sel_r["pnl"]/1e3],
        measure=["relative"]*len(assets) + ["total"],
        connector=dict(line=dict(color=BORDER)),
        increasing=dict(marker_color=GREEN),
        decreasing=dict(marker_color=RED),
        totals=dict(marker_color=BLUE),
    ))
    fig2.update_yaxes(title_text="P&L (₹K)")
    _fig_layout(fig2, f"P&L Waterfall — {sel}", 350)
    st.plotly_chart(fig2, use_container_width=True)

    # ── shock table ───────────────────────────────────────────────────────────
    st.markdown("#### Scenario shocks applied")
    shock_rows = []
    for name, shocks in DEFAULT_SCENARIOS.items():
        row = {"Scenario": name}
        row.update({k: f"{v*100:+.0f}%" for k, v in shocks.items()})
        shock_rows.append(row)
    st.dataframe(pd.DataFrame(shock_rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — VOLATILITY MODELS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Volatility Models":
    from app.portfolio.volatility_models import (
        ewma_variance_series, ewma_volatility, fit_garch11, garch_forecast,
    )

    st.markdown("## Volatility Models — EWMA & GARCH(1,1)")

    with st.spinner("Fitting GARCH(1,1)…"):
        ewma_res   = ewma_volatility(port_ret)
        garch_res  = fit_garch11(port_ret)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric("EWMA Daily Vol",   f"{ewma_res['daily_vol_forecast']*100:.4f}%",  "λ = 0.94",   BLUE)
    with c2: metric("EWMA Ann. Vol",    f"{ewma_res['annualized_vol_forecast']*100:.2f}%", "×√252",  BLUE)
    with c3: metric("GARCH Daily Vol",  f"{garch_res['long_run_daily_vol']*100:.4f}%", "long-run σ", AMBER)
    with c4: metric("GARCH Next-day Vol",f"{garch_forecast(garch_res)['next_day_vol']*100:.4f}%", "σ_t+1", GREEN)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── EWMA vol series ────────────────────────────────────────────────────────
    s2 = ewma_variance_series(port_ret.values)
    ewma_vol_series = pd.Series(np.sqrt(s2[1:]) * 100, index=port_ret.index)

    roll_vol = port_ret.rolling(21).std() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll_vol.index, y=roll_vol,
        name="21-day Rolling", line=dict(color=GREY, width=1, dash="dot"), opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=ewma_vol_series.index, y=ewma_vol_series,
        name="EWMA (λ=0.94)", line=dict(color=BLUE, width=2),
    ))
    fig.update_yaxes(title_text="Daily Volatility (%)")
    _fig_layout(fig, "EWMA vs Rolling Volatility", 350)
    st.plotly_chart(fig, use_container_width=True)

    # ── GARCH params & diagnostics ────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### GARCH(1,1) Parameters")
        gp = {
            "ω (omega)":        f"{garch_res['omega']:.6f}",
            "α (alpha)":        f"{garch_res['alpha']:.4f}",
            "β (beta)":         f"{garch_res['beta']:.4f}",
            "Persistence α+β":  f"{garch_res['persistence']:.4f}",
            "Half-life (days)": f"{garch_res.get('half_life', 'N/A')}",
            "Log-likelihood":   f"{garch_res.get('log_likelihood', 'N/A')}",
        }
        for k, v in gp.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;border-bottom:1px solid {BORDER};'>"
                f"<span style='color:{GREY};font-size:12px;'>{k}</span>"
                f"<span style='font-weight:600;font-size:12px;'>{v}</span></div>",
                unsafe_allow_html=True,
            )

    with col_r:
        # GARCH multi-step forecast
        try:
            fc = garch_forecast(garch_res, steps=20)
            fc_vols = [v * 100 for v in fc["vol_forecast"]]
            fig2 = go.Figure(go.Scatter(
                x=list(range(1, len(fc_vols)+1)), y=fc_vols,
                mode="lines+markers", line=dict(color=AMBER, width=2),
                marker=dict(size=5),
            ))
            fig2.add_hline(y=garch_res["long_run_vol"]*100,
                           line=dict(color=GREEN, dash="dash"),
                           annotation_text="Long-run vol", annotation_font_color=GREEN)
            fig2.update_xaxes(title_text="Step ahead")
            fig2.update_yaxes(title_text="Daily Vol (%)")
            _fig_layout(fig2, "GARCH 20-Step Forecast", 300)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.info("GARCH forecast unavailable.")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — BACKTESTING
# ════════════════════════════════════════════════════════════════════════════
elif page == "Backtesting":
    from app.portfolio.backtesting import backtest_var, rolling_var_forecasts
    from app.portfolio.risk_metrics import historical_var_es

    st.markdown("## VaR Backtesting")
    st.caption(
        "Rolling-window out-of-sample VaR. Window = 250 days. "
        "Kupiec POF tests unconditional coverage; "
        "Christoffersen tests independence of exceptions."
    )

    with st.spinner("Running backtests…"):
        results = {m: backtest_var(port_ret, confidence=confidence, method=m)
                   for m in ["historical", "parametric", "ewma"]}

    # ── Basel traffic light summary ───────────────────────────────────────────
    st.markdown("#### Basel Traffic Light")
    tl_cols = st.columns(3)
    for i, (m, r) in enumerate(results.items()):
        tl  = r["basel_traffic_light"]
        col = {"Green": GREEN, "Amber": AMBER, "Red": RED}.get(tl["zone"], GREY)
        with tl_cols[i]:
            metric(
                m.capitalize(),
                tl["zone"],
                f"{tl['exceptions']} exceptions / {r['n_obs']} days",
                col,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── test results table ────────────────────────────────────────────────────
    rows = []
    for m, r in results.items():
        kup  = r["kupiec_pof"]
        chri = r["christoffersen_independence"]
        exc  = r["n_exceptions"]
        exp  = round(r["n_obs"] * (1 - confidence), 1)
        rows.append({
            "Model":           m.capitalize(),
            "Exceptions":      exc,
            "Expected":        exp,
            "Exception Rate":  f"{r['exception_rate']*100:.2f}%",
            "Kupiec LR":       f"{kup['statistic']:.3f}",
            "Kupiec p-val":    f"{kup['p_value']:.4f}",
            "CC p-val":        f"{chri['p_value']:.4f}",
            "Basel Zone":      r["basel_traffic_light"]["zone"],
        })
    tbl = pd.DataFrame(rows)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── exception plot ────────────────────────────────────────────────────────
    sel_method = st.selectbox("Plot exceptions for", ["historical", "parametric", "ewma"])
    var_series = rolling_var_forecasts(port_ret, confidence=confidence, method=sel_method)
    actual_loss = -port_ret.reindex(var_series.index)
    hits        = actual_loss > var_series

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual_loss.index, y=actual_loss * 100,
        name="Actual loss", line=dict(color=GREY, width=1), opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=var_series.index, y=var_series * 100,
        name=f"VaR ({int(confidence*100)}%)", line=dict(color=BLUE, width=2),
    ))
    exc_dates = actual_loss.index[hits]
    fig.add_trace(go.Scatter(
        x=exc_dates, y=actual_loss[hits] * 100,
        mode="markers", name="Exception",
        marker=dict(color=RED, size=7, symbol="x"),
    ))
    fig.update_yaxes(title_text="Loss (% of portfolio)")
    _fig_layout(fig, f"Exceptions — {sel_method.capitalize()} VaR", 400)
    st.plotly_chart(fig, use_container_width=True)

    # ── exception clustering ──────────────────────────────────────────────────
    st.markdown("#### Exception timeline")
    hit_series = hits.astype(int)
    fig2 = go.Figure(go.Bar(
        x=hit_series.index, y=hit_series,
        marker_color=RED, name="Exception day",
    ))
    fig2.update_yaxes(tickvals=[0, 1], title_text="Exception")
    _fig_layout(fig2, "Exception Clustering", 160)
    st.plotly_chart(fig2, use_container_width=True)
