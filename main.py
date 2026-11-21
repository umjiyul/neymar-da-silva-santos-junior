import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── 피어슨 상관계수 헬퍼 (scipy 없으면 numpy로 대체) ──────────────────────
def pearsonr_safe(x, y):
    try:
        from scipy.stats import pearsonr as pr
        r, _ = pr(x, y)
        return r, None
    except ImportError:
        mx, my = x.mean(), y.mean()
        n = len(x)
        if n < 2:
            return 0.0, None
        sxy = ((x - mx) * (y - my)).sum()
        sxx = ((x - mx) ** 2).sum()
        syy = ((y - my) ** 2).sum()
        denom = np.sqrt(sxx * syy)
        r = float(sxy / denom) if denom != 0 else 0.0
        return r, None


# ── CSV 읽기 함수 ───────────────────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ── 열 이름 표준화 ───────────────────────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


# ── 사이드바: 파일 업로드 ────────────────────────────────────────────────────
st.sidebar.header("📁 CSV 업로드")
env_file    = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

st.sidebar.header("📏 생육 측정 항목")
measure_cols = st.sidebar.multiselect(
    "분석할 항목 선택",
    ["잎 수", "지상부 길이", "지하부 길이", "생중량"],
    default=["잎 수", "지상부 길이", "지하부 길이", "생중량"],
)


# ── 데이터 불러오기 (한 번만, n_env·n_growth는 바깥 스코프에서 정의) ──────
def school_name(file):
    if file is None:
        return None
    return os.path.basename(file.name).split("_")[0].split("-")[0]


sn_env    = school_name(env_file)
sn_growth = school_name(growth_file)
school = sn_env or sn_growth or "우리 학교"

with st.spinner("데이터를 불러오는 중..."):
    env_df_raw    = read_csv(env_file)
    growth_df_raw = read_csv(growth_file)

env_df    = std_cols(env_df_raw)    if not env_df_raw.empty    else pd.DataFrame()
growth_df = std_cols(growth_df_raw) if not growth_df_raw.empty else pd.DataFrame()

n_env    = len(env_df)
n_growth = len(growth_df)


# ── 헬퍼 ────────────────────────────────────────────────────────────────────
def daily_series(df, col):
    sub = df[["time", col]].dropna()
    if sub.empty:
        return None
    sub = sub.copy()
    sub["day"] = sub["time"].dt.date
    return sub.groupby("day")[col].mean()


def trendline(x, y):
    mask = pd.notna(x) & pd.notna(y)
    x2, y2 = x[mask], y[mask]
    if len(x2) < 2:
        return None, None
    a, b = np.polyfit(x2, y2, 1)
    return a, b


def plotly_font(fig):
    fig.update_layout(font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))


def csv_download(df, label):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"📥 {label} CSV 다운로드",
        data=csv_bytes,
        file_name=f"{label}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📖 연구 소개 및 재배 환경",
    "📊 생육 길이와 생중량의 관계",
    "🏅 생중량 상·하위 개체 비교 및 결론",
])


# ── 탭 1: 연구 소개 및 재배 환경 ────────────────────────────────────────────
with tab1:
    try:
        st.markdown("""\
**연구 목적**
식물 생육은 개체마다 차이를 보이며, 그 차이는 재배 환경과 식물의 구조적 특성(길이·잎 수)에 영향을 받습니다.
이 대시보드에서는 {school} 학교 극지식물 개체들의 생중량 차이를 분석하고,
생중량에 가장 큰 영향을 미치는 생육 요인이 무엇인지 알아봅니다.

**핵심 질문**
1. 생중량 상위 25% 개체와 하위 25% 개체는 지상부 길이·지하부 길이에서 어떤 차이를 보일까요?
2. 지상부 길이와 지하부 길이 중 어떤 요인이 생중량과 더 강한 상관관계를 가질까요?
""".format(school=school).strip())

        if "temperature" in env_df.columns and n_env > 0:
            t = pd.to_numeric(env_df["temperature"], errors="coerce").dropna()
            h = pd.to_numeric(env_df["humidity"], errors="coerce").dropna()
            p = pd.to_numeric(env_df["ph"], errors="coerce").dropna() if "ph" in env_df.columns else None
            e = pd.to_numeric(env_df["ec"], errors="coerce").dropna() if "ec" in env_df.columns else None

            st.subheader(f"🌡️ {school} 재배 환경 평균값")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "평균 온도",
                f"{round(t.mean(), 2)}°C" if len(t) > 0 else "정보 없음",
                help="온도는 식물의 광합성·호흡·생장 속도에 직접 영향을 줍니다.",
            )
            m2.metric(
                "평균 습도",
                f"{round(h.mean(), 2)}%" if len(h) > 0 else "정보 없음",
                help="습도는 증산작용과 기공 개폐에 영향을 주어 생육 차이를 만들 수 있습니다.",
            )
            if p is not None and len(p) > 0:
                m3.metric("평균 pH", f"{round(p.mean(), 2)}",
                          help="pH는 토양·수경의 산도로, 영양분 흡수에 영향을 줍니다.")
            else:
                m3.metric("평균 pH", "데이터 없음")
            if e is not None and len(e) > 0:
                m4.metric("평균 EC", f"{round(e.mean(), 2)} mS/cm",
                          help="EC(전기전도도)는 수경·양액의 염류 농도로, 영양 상태를 나타냅니다.")
            else:
                m4.metric("평균 EC", "데이터 없음")
        else:
            st.info("🌡️ 환경 데이터 CSV를 사이드바에서 업로드하면 환경 평균값과 그래프가 표시됩니다.")

        if "time" in env_df.columns and n_env > 0:
            t_daily = daily_series(env_df, "temperature") if "temperature" in env_df.columns else None
            h_daily = daily_series(env_df, "humidity")    if "humidity"    in env_df.columns else None

            fig_env = go.Figure()
            has_any = False
            if t_daily is not None and len(t_daily) >= 2:
                fig_env.add_trace(go.Scatter(
                    x=list(t_daily.index), y=list(t_daily.values),
                    mode="lines", name="온도 (°C)",
                    line=dict(color="#e74c3c", width=2),
                ))
                has_any = True
            if h_daily is not None and len(h_daily) >= 2:
                fig_env.add_trace(go.Scatter(
                    x=list(h_daily.index), y=list(h_daily.values),
                    mode="lines", name="습도 (%)",
                    line=dict(color="#3498db", width=2),
                ))
                has_any = True

            if has_any:
                fig_env.update_layout(
                    height=400,
                    xaxis_title="날짜",
                    yaxis_title="값",
                    legend=dict(orientation="h", yanchor="top", y=1.12,
                                xanchor="center", x=0.5),
                    hovermode="x unified",
                )
                plotly_font(fig_env)
                st.plotly_chart(fig_env, width="stretch")
            else:
                st.info("일별 평균 꺾은선 그래프를 그릴 수 있는 데이터가 충분하지 않습니다.")
        else:
            st.info("환경 데이터에 시간 정보가 없어 꺾은선 그래프를 표시할 수 없습니다.")

        if n_env > 0:
            with st.expander("📄 환경 데이터 원본 표 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                csv_download(env_df, f"{school}_환경데이터")
        else:
            st.info("표시할 환경 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ── 탭 2: 생육 길이와 생중량의 관계 ─────────────────────────────────────────
with tab2:
    try:
        if growth_df.empty:
            st.info(f"📊 생육 결과 CSV를 사이드바에서 업로드하면 생육 분석이 표시됩니다. (현재 업로드된 생육 데이터: {n_growth}행)")
        else:
            st.subheader(f"📊 {school} 생육 길이와 생중량의 관계")

            st.caption(
                "상관계수 r은 -1 ~ 1 사이의 값입니다. "
                "0에 가까우면 관계가 약하고, 1 또는 -1에 가까우면 강한 선형 관계를 의미합니다. "
                "양의 r은 한 값이 커질 때 다른 값도 커지는 경향, 음의 r은 반대 경향을 나타냅니다."
            )

            has_length = "지상부 길이" in growth_df.columns
            has_biomass = "생중량" in growth_df.columns

            if not has_length or not has_biomass:
                st.info("지상부 길이 또는 생중량 열이 없어 산점도·상관계수 분석을 할 수 없습니다.")
            else:
                fig2 = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("지상부 길이 vs 생중량", "지하부 길이 vs 생중량"),
                    horizontal_spacing=0.12,
                )

                r_above = None
                r_below = None
                above_msg = ""
                below_msg = ""

                # ── 지상부 길이 vs 생중량 ──────────────────────────────────
                sg_above = growth_df[["지상부 길이", "생중량"]].dropna()
                if len(sg_above) >= 2:
                    x_a = sg_above["지상부 길이"]
                    y_a = sg_above["생중량"]
                    r_above, _ = pearsonr_safe(x_a, y_a)
                    a_a, b_a = trendline(x_a, y_a)

                    fig2.add_trace(go.Scatter(
                        x=x_a, y=y_a,
                        mode="markers",
                        marker=dict(color="#3498db", size=9, opacity=0.7,
                                    line=dict(width=1, color="white")),
                        name="지상부 길이",
                        hovertext=[f"지상부 길이={xv:.1f}mm  생중량={yv:.2f}g"
                                   for xv, yv in zip(x_a, y_a)],
                        hoverinfo="text",
                    ), row=1, col=1)

                    if a_a is not None:
                        xs_a = np.linspace(x_a.min(), x_a.max(), 100)
                        fig2.add_trace(go.Scatter(
                            x=xs_a, y=a_a * xs_a + b_a,
                            mode="lines",
                            line=dict(color="#e74c3c", width=2, dash="dash"),
                            name=f"추세선 (r={r_above:.2f})",
                        ), row=1, col=1)

                    fig2.add_annotation(
                        text=f"r = {r_above:.2f}",
                        xref="paper", yref="paper",
                        x=0.95, y=0.05, xanchor="right", yanchor="bottom",
                        showarrow=False, font=dict(size=11, color="#2c3e50"),
                        row=1, col=1,
                    )
                    above_msg = (
                        f"지상부 길이 vs 생중량 상관계수 r = {r_above:.2f} — "
                        f"{'강한 양의 상관' if r_above > 0.5 else '중간 정도의 양의 상관' if r_above > 0.3 else '약한 상관'}"
                    )
                else:
                    above_msg = "지상부 길이 vs 생중량: 데이터 부족으로 상관계수 계산 불가"

                # ── 지하부 길이 vs 생중량 ──────────────────────────────────
                sg_below = growth_df[["지하부 길이", "생중량"]].dropna()
                if len(sg_below) >= 2:
                    x_b = sg_below["지하부 길이"]
                    y_b = sg_below["생중량"]
                    r_below, _ = pearsonr_safe(x_b, y_b)
                    a_b, b_b = trendline(x_b, y_b)

                    fig2.add_trace(go.Scatter(
                        x=x_b, y=y_b,
                        mode="markers",
                        marker=dict(color="#2ecc71", size=9, opacity=0.7,
                                    line=dict(width=1, color="white")),
                        name="지하부 길이",
                        hovertext=[f"지하부 길이={xv:.1f}mm  생중량={yv:.2f}g"
                                   for xv, yv in zip(x_b, y_b)],
                        hoverinfo="text",
                    ), row=1, col=2)

                    if a_b is not None:
                        xs_b = np.linspace(x_b.min(), x_b.max(), 100)
                        fig2.add_trace(go.Scatter(
                            x=xs_b, y=a_b * xs_b + b_b,
                            mode="lines",
                            line=dict(color="#27ae60", width=2, dash="dash"),
                            name=f"추세선 (r={r_below:.2f})",
                        ), row=1, col=2)

                    fig2.add_annotation(
                        text=f"r = {r_below:.2f}",
                        xref="paper", yref="paper",
                        x=0.95, y=0.05, xanchor="right", yanchor="bottom",
                        showarrow=False, font=dict(size=11, color="#2c3e50"),
                        row=1, col=2,
                    )
                    below_msg = (
                        f"지하부 길이 vs 생중량 상관계수 r = {r_below:.2f} — "
                        f"{'강한 양의 상관' if r_below > 0.5 else '중간 정도의 양의 상관' if r_below > 0.3 else '약한 상관'}"
                    )
                else:
                    below_msg = "지하부 길이 vs 생중량: 데이터 부족으로 상관계수 계산 불가"

                fig2.update_layout(height=400, showlegend=False)
                plotly_font(fig2)
                st.plotly_chart(fig2, width="stretch")

                if r_above is not None and r_below is not None:
                    c1, c2 = st.columns(2)
                    c1.info(above_msg)
                    c2.info(below_msg)
                    if abs(r_above) > abs(r_below):
                        st.info(
                            f"📌 두 요인 중 **지상부 길이(r={r_above:.2f})**가 "
                            f"**지하부 길이(r={r_below:.2f})**보다 생중량과 더 큰 관련성을 보입니다."
                        )
                    elif abs(r_below) > abs(r_above):
                        st.info(
                            f"📌 두 요인 중 **지하부 길이(r={r_below:.2f})**가 "
                            f"**지상부 길이(r={r_above:.2f})**보다 생중량과 더 큰 관련성을 보입니다."
                        )
                    else:
                        st.info(
                            f"두 요인의 상관계수가 비슷합니다 "
                            f"(지상부 r={r_above:.2f}, 지하부 r={r_below:.2f}). "
                            f"둘 다 생중량에 비슷한 정도로 관련되어 있습니다."
                        )

            with st.expander("📄 생육 결과 데이터 원본 표 + CSV 다운로드"):
                if n_growth > 0:
                    st.dataframe(growth_df, width="stretch")
                    csv_download(growth_df, f"{school}_생육결과데이터")
                else:
                    st.info("표시할 생육 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ── 탭 3: 생중량 상·하위 개체 비교 및 결론 ─────────────────────────────────
with tab3:
    try:
        if growth_df.empty:
            st.info(f"📊 생육 결과 CSV를 사이드바에서 업로드하면 상·하위 비교가 표시됩니다. (현재 업로드된 생육 데이터: {n_growth}행)")
        else:
            st.subheader(f"🏅 {school} 생중량 상위 25% vs 하위 25% 비교")

            if "생중량" not in growth_df.columns:
                st.info("생중량 열이 없어 상·하위 개체 비교를 할 수 없습니다.")
            else:
                sg = growth_df[["생중량"]].copy()
                sg["생중량"] = pd.to_numeric(sg["생중량"], errors="coerce")
                sg = sg.dropna(subset=["생중량"])

                if len(sg) < 4:
                    st.info("생중량 데이터가 4개 미만이라 상위·하위 25%를 나눌 수 없습니다.")
                else:
                    q1  = sg["생중량"].quantile(0.25)
                    q3  = sg["생중량"].quantile(0.75)
                    top    = sg[sg["생중량"] >= q3]
                    bottom = sg[sg["생중량"] <= q1]

                    if top.empty or bottom.empty:
                        st.info("상위 또는 하위 25%에 해당하는 개체가 없습니다.")
                    else:
                        idx_top    = top.index
                        idx_bottom = bottom.index

                        above_avg = {"지상부 길이": 0, "지하부 길이": 0}
                        below_avg = {"지상부 길이": 0, "지하부 길이": 0}
                        for col in ["지상부 길이", "지하부 길이"]:
                            if col in growth_df.columns:
                                above_avg[col] = growth_df.loc[idx_top,    col].mean()
                                below_avg[col] = growth_df.loc[idx_bottom, col].mean()

                        comp_data = {
                            "구분": ["생중량 상위 25%", "생중량 하위 25%"],
                            "생중량 평균(g)": [
                                round(top["생중량"].mean(), 2),
                                round(bottom["생중량"].mean(), 2),
                            ],
                        }
                        for col in ["지상부 길이", "지하부 길이"]:
                            if col in growth_df.columns:
                                comp_data[col] = [
                                    round(above_avg[col], 2),
                                    round(below_avg[col], 2),
                                ]

                        st.dataframe(pd.DataFrame(comp_data), width="stretch", hide_index=True)

                        fig3 = go.Figure()
                        bar_labels = []
                        bar_top_vals    = []
                        bar_bottom_vals = []
                        for col in ["지상부 길이", "지하부 길이"]:
                            if col in comp_data:
                                bar_labels.append(f"{col}\n(상위 25%)")
                                bar_labels.append(f"{col}\n(하위 25%)")
                                bar_top_vals.append(comp_data[col][0])
                                bar_bottom_vals.append(comp_data[col][1])

                        if bar_labels:
                            fig3.add_trace(go.Bar(
                                x=bar_labels[:len(bar_labels)//2] if len(bar_labels) >= 2 else bar_labels,
                                y=bar_top_vals,
                                name="상위 25%",
                                marker_color="#e74c3c",
                                offsetgroup=0,
                            ))
                            half = len(bar_labels) // 2
                            fig3.add_trace(go.Bar(
                                x=bar_labels[half:half*2] if half * 2 <= len(bar_labels) else bar_labels[half:],
                                y=bar_bottom_vals,
                                name="하위 25%",
                                marker_color="#3498db",
                                offsetgroup=1,
                            ))
                            fig3.update_layout(
                                barmode="group",
                                height=380,
                                xaxis_title="측정 항목",
                                yaxis_title="평균 값",
                                legend=dict(orientation="h", yanchor="top", y=1.12,
                                            xanchor="center", x=0.5),
                            )
                            plotly_font(fig3)
                            st.plotly_chart(fig3, width="stretch")

                        st.subheader("📌 분석 결론: 생중량 차이에 영향을 미치는 주요 요인")
                        if r_above is not None and r_below is not None:
                            if abs(r_above) > abs(r_below):
                                factor = "지상부 길이"
                                rf     = r_above
                                other  = "지하부 길이"
                                ro     = r_below
                            else:
                                factor = "지하부 길이"
                                rf     = r_below
                                other  = "지상부 길이"
                                ro     = r_above

                            if factor == "지상부 길이":
                                top_val    = round(above_avg.get("지상부 길이", 0), 2)
                                bottom_val = round(below_avg.get("지상부 길이", 0), 2)
                            else:
                                top_val    = round(above_avg.get("지하부 길이", 0), 2)
                                bottom_val = round(below_avg.get("지하부 길이", 0), 2)

                            st.markdown(f"""\
**생중량 차이에 가장 큰 영향을 미치는 요인: {factor} (r = {rf:.2f})**

- 생중량 상위 25% 개체의 {factor} 평균: **{top_val}**
- 생중량 하위 25% 개체의 {factor} 평균: **{bottom_val}**
- {factor}가 길수록 생중량도 커지는 경향(r = {rf:.2f})을 보입니다.
- {other}도 생중량과 관련이 있지만(r = {ro:.2f}), {factor}만큼 강한 관련성은 보이지 않습니다.
- 따라서 {school} 극지식물의 생중량 차이는 **{factor}의 차이**가 주요 원인일 가능성이 높습니다.
""".strip())
                        else:
                            st.info("상관계수 계산이 완료되지 않아 결론을 표시할 수 없습니다.")

            with st.expander("📄 생육 결과 데이터 원본 표 + CSV 다운로드"):
                if n_growth > 0:
                    st.dataframe(growth_df, width="stretch")
                    csv_download(growth_df, f"{school}_생육결과데이터")
                else:
                    st.info("표시할 생육 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
