import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="Streamlit Test",
    page_icon="✅"
)

st.title("✅ Streamlit 연결 테스트")

st.write("이 화면이 보이면 GitHub와 Streamlit이 정상적으로 연결되었습니다.")

st.divider()

st.write("⏰ 현재 시간 (한국 시간):")
st.write(datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"))

st.caption("페이지를 새로고침했을 때 시간이 바뀌면 정상입니다.")

st.success("연결 성공!")
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 한글 폰트 CSS ────────────────────────────────────────────────────────────
st.markdown("""\
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 분석", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

# ── 피어슨 상관계수 헬퍼 (scipy 없으면 numpy로 대체) ────────────────────────
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

# ── CSV 읽기 함수 ────────────────────────────────────────────────────────────
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

# ── 데이터 불러오기 (한 번만, 전역 스코프) ───────────────────────────────────
with st.spinner("데이터를 불러오는 중..."):
    env_df_raw    = read_csv(env_file)
    growth_df_raw = read_csv(growth_file)
    env_df    = std_cols(env_df_raw)    if not env_df_raw.empty    else pd.DataFrame()
    growth_df = std_cols(growth_df_raw) if not growth_df_raw.empty else pd.DataFrame()

n_env    = len(env_df)
n_growth = len(growth_df)

# 학교명 추출
def school_name(file):
    if file is None:
        return None
    name = os.path.basename(file.name)
    return name.split("_")[0].split("-")[0]

sn_env    = school_name(env_file)
sn_growth = school_name(growth_file)
school = sn_env or sn_growth or "우리 학교"

# ── 공통 헬퍼 ────────────────────────────────────────────────────────────────
def safe_mean(df, col):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s.mean() if len(s) >= 2 else None

def safe_max(df, col):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s.max() if len(s) >= 2 else None

def safe_min(df, col):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s.min() if len(s) >= 2 else None

def trendline(x, y):
    mask = pd.notna(x) & pd.notna(y)
    x2, y2 = x[mask], y[mask]
    if len(x2) < 2:
        return None, None
    a, b = np.polyfit(x2, y2, 1)
    return a, b

def csv_download(df, label):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"📥 {label} CSV 다운로드",
        data=csv_bytes,
        file_name=f"{label}.csv",
        mime="text/csv",
    )

def plotly_font(fig):
    fig.update_layout(font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))

def daily_series(df, col):
    sub = df[["time", col]].dropna()
    if sub.empty:
        return None
    sub = sub.copy()
    sub["day"] = sub["time"].dt.date
    return sub.groupby("day")[col].mean()

# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ── 탭 1: 📖 실험 개요 ───────────────────────────────────────────────────────
with tab1:
    try:
        st.markdown(f"""
**연구 배경**
{school} 학교 식물 생육 실험에서 극지식물 개체마다 생육 상태가 다르게 나타납니다.
생육 차이의 원인을 환경 데이터(온도·습도·pH·EC)와 연결하여 분석하고,
개체 간 차이를 통계적으로 살펴봅니다.

**분석 내용**
- 개체별 생중량 분포와 차이 (평균·표준편차·CV)
- 잎 수·지상부 길이와 생중량의 관계 (상관계수·산점도)
- 생중량 상위 25% vs 하위 25% 개체 비교
""".strip())

        # 데이터 요약 표
        st.subheader("📋 데이터 요약")

        if "time" in env_df.columns and n_env > 0:
            tmin = env_df["time"].min()
            tmax = env_df["time"].max()
            period = f"{tmin.strftime('%Y-%m-%d')} ~ {tmax.strftime('%Y-%m-%d')}"
        else:
            period = "정보 없음"

        env_items    = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
        growth_items = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]

        summary_data = {
            "항목": [
                "개체 수",
                "환경 측정 기간",
                "환경 측정 항목 수",
                "생육 측정 항목 수",
                "총 환경 데이터 수",
                "총 생육 데이터 수",
            ],
            "내용": [
                f"{n_growth}개체",
                period,
                f"{len(env_items)}개({', '.join(env_items) if env_items else '없음'})",
                f"{len(growth_items)}개({', '.join(growth_items) if growth_items else '없음'})",
                f"{n_env}행",
                f"{n_growth}행",
            ],
        }
        st.dataframe(pd.DataFrame(summary_data), width="stretch", hide_index=True)

        # 주요 지표 카드 4개
        st.subheader("📊 주요 지표")
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("총 개체 수", f"{n_growth}개체")

        if "temperature" in env_df.columns and n_env > 0:
            avg_t = safe_mean(env_df, "temperature")
            c2.metric("평균 온도", f"{avg_t:.1f}°C" if avg_t is not None else "정보 없음")
        else:
            c2.metric("평균 온도", "데이터 없음")

        if "humidity" in env_df.columns and n_env > 0:
            avg_h = safe_mean(env_df, "humidity")
            c3.metric("평균 습도", f"{avg_h:.1f}%" if avg_h is not None else "정보 없음")
        else:
            c3.metric("평균 습도", "데이터 없음")

        if "생중량" in growth_df.columns and n_growth > 0:
            avg_g = safe_mean(growth_df, "생중량")
            c4.metric("평균 생중량", f"{avg_g:.2f}g" if avg_g is not None else "정보 없음")
        else:
            c4.metric("평균 생중량", "데이터 없음")

        if n_growth == 0:
            st.info("📊 생육 결과 CSV를 업로드하면 지표와 그래프가 표시됩니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ── 탭 2: 🌡️ 환경 데이터 ────────────────────────────────────────────────────
with tab2:
    try:
        if env_df.empty:
            st.info("🌡️ 환경 데이터 CSV를 사이드바에서 업로드하면 환경 분석 내용이 표시됩니다.")
        else:
            st.subheader(f"🌡️ {school} 환경 데이터 분석")

            # 환경 요약 표
            env_num_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_num_cols:
                rows = []
                for col in env_num_cols:
                    s = pd.to_numeric(env_df[col], errors="coerce").dropna()
                    if len(s) >= 2:
                        rows.append({
                            "측정 항목": col,
                            "평균": round(s.mean(), 2),
                            "최소": round(s.min(), 2),
                            "최대": round(s.max(), 2),
                            "표준편차": round(s.std(), 2),
                            "데이터 수": len(s),
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                else:
                    st.info("환경 데이터가 충분하지 않아 요약 통계를 계산할 수 없습니다.")
            else:
                st.info("환경 데이터에 인식할 수 있는 측정 열이 없습니다.")

            # 일별 시계열
            if "time" in env_df.columns and n_env > 0:
                st.subheader("📈 일별 환경 변화 (일 평균)")

                t_daily  = daily_series(env_df, "temperature") if "temperature" in env_df.columns else None
                h_daily  = daily_series(env_df, "humidity")    if "humidity"    in env_df.columns else None
                ph_daily = daily_series(env_df, "ph")          if "ph"          in env_df.columns else None
                ec_daily = daily_series(env_df, "ec")          if "ec"          in env_df.columns else None

                fig_ts = go.Figure()
                has_any = False
                if t_daily is not None and len(t_daily) >= 2:
                    fig_ts.add_trace(go.Scatter(
                        x=list(t_daily.index), y=list(t_daily.values),
                        mode="lines", name="온도 (°C)",
                        line=dict(color="#e74c3c", width=2),
                    ))
                    has_any = True
                if h_daily is not None and len(h_daily) >= 2:
                    fig_ts.add_trace(go.Scatter(
                        x=list(h_daily.index), y=list(h_daily.values),
                        mode="lines", name="습도 (%)",
                        line=dict(color="#3498db", width=2),
                    ))
                    has_any = True
                if ph_daily is not None and len(ph_daily) >= 2:
                    fig_ts.add_trace(go.Scatter(
                        x=list(ph_daily.index), y=list(ph_daily.values),
                        mode="lines", name="pH",
                        line=dict(color="#27ae60", width=2),
                    ))
                    has_any = True
                if ec_daily is not None and len(ec_daily) >= 2:
                    fig_ts.add_trace(go.Scatter(
                        x=list(ec_daily.index), y=list(ec_daily.values),
                        mode="lines", name="EC (mS/cm)",
                        line=dict(color="#8e44ad", width=2),
                    ))
                    has_any = True

                if has_any:
                    fig_ts.update_layout(
                        height=400,
                        xaxis_title="날짜",
                        yaxis_title="값",
                        legend=dict(orientation="h", yanchor="top", y=1.12,
                                    xanchor="center", x=0.5),
                        hovermode="x unified",
                    )
                    plotly_font(fig_ts)
                    st.plotly_chart(fig_ts, width="stretch")
                else:
                    st.info("일별 평균 그래프를 그릴 수 있는 환경 데이터가 충분하지 않습니다.")
            else:
                st.info("환경 데이터에 시간 정보가 없어 시계열 그래프를 그릴 수 없습니다.")

            # 원본 테이블 + 다운로드
            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                if n_env > 0:
                    st.dataframe(env_df, width="stretch")
                    csv_download(env_df, f"{school}_환경데이터")
                else:
                    st.info("표시할 환경 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ── 탭 3: 📊 생육 결과 ───────────────────────────────────────────────────────
with tab3:
    try:
        if growth_df.empty:
            st.info("📊 생육 결과 CSV를 사이드바에서 업로드하면 생육 분석 내용이 표시됩니다.")
        else:
            st.subheader(f"📊 {school} 극지식물 생육 결과 분석")
            st.caption(f"⚠️ 개체번호는 분석 항목에서 제외되며, {school} 개체끼리만 비교합니다.")

            # ── 🥇 핵심 결과 카드 ────────────────────────────────────────────
            st.subheader("🏆 핵심 결과")
            if "생중량" in growth_df.columns and n_growth >= 2:
                s_full = pd.to_numeric(growth_df["생중량"], errors="coerce")
                g = s_full.dropna()
                if len(g) >= 2:
                    avg_g  = g.mean()
                    std_g  = g.std()
                    cv_g   = (std_g / avg_g * 100) if avg_g > 0 else 0.0
                    idx_max = s_full.idxmax()
                    idx_min = s_full.idxmin()
                    max_id  = growth_df.loc[idx_max, "개체번호"] if "개체번호" in growth_df.columns else f"#{idx_max}"
                    min_id  = growth_df.loc[idx_min, "개체번호"] if "개체번호" in growth_df.columns else f"#{idx_min}"

                    kg1, kg2, kg3, kg4 = st.columns(4)
                    kg1.metric(
                        "평균 생중량",
                        f"{avg_g:.2f}g  (σ={std_g:.2f}, CV={cv_g:.1f}%)",
                        help="평균=모든 개체 생중량 합 ÷ 개체 수. 표준편차(σ)=개체들이 평균에서 얼마나 퍼져있는지 나타내는 값. CV(변동계수)=표준편차÷평균×100, 크기가 다른 집단 비교에 유용.",
                    )
                    kg2.metric(
                        "최대 생중량 개체",
                        f"개체 {max_id}  {g.max():.2f}g",
                        help="전체에서 생중량이 가장 큰 개체.",
                    )
                    kg3.metric(
                        "최소 생중량 개체",
                        f"개체 {min_id}  {g.min():.2f}g",
                        help="전체에서 생중량이 가장 작은 개체.",
                    )
                    kg4.metric(
                        "생중량 범위",
                        f"{g.min():.2f} ~ {g.max():.2f}g  (차이 {g.max()-g.min():.2f}g)",
                        help="최대값과 최소값의 차이. 값이 클수록 개체 간 생육 차이가 큼.",
                    )

            # ── 개체별 생육 비교 (막대 그래프) ───────────────────────────────
            st.subheader("📊 개체별 생육 비교")
            bar_cols = [c for c in measure_cols if c in growth_df.columns]
            if bar_cols:
                for col in bar_cols:
                    fig_bar = go.Figure()
                    vals = pd.to_numeric(growth_df[col], errors="coerce")
                    if "개체번호" in growth_df.columns:
                        ids = growth_df["개체번호"]
                    else:
                        ids = [f"#{i}" for i in range(len(growth_df))]

                    avg_val = vals.dropna().mean() if vals.dropna().shape[0] >= 2 else None

                    if col == "생중량" and "생중량" in growth_df.columns:
                        q1 = vals.dropna().quantile(0.25)
                        q3 = vals.dropna().quantile(0.75)
                        bar_colors = [
                            "#e74c3c" if (v is not None and v >= q3) else
                            "#2ecc71" if (v is not None and v <= q1) else
                            "#95a5a6"
                            for v in vals
                        ]
                    else:
                        bar_colors = ["#3498db"] * len(vals)

                    fig_bar.add_trace(go.Bar(
                        x=[str(i) for i in ids],
                        y=vals,
                        marker_color=bar_colors,
                        marker_line_width=0.5,
                        marker_line_color="rgba(0,0,0,0.2)",
                        name=col,
                    ))
                    if avg_val is not None:
                        fig_bar.add_hline(
                            y=avg_val,
                            line_dash="dash",
                            line_color="#e67e22",
                            annotation_text=f"평균 {avg_val:.1f}",
                            annotation_position="top right",
                        )
                    fig_bar.update_layout(
                        height=350,
                        xaxis_title="개체번호",
                        yaxis_title=col,
                        showlegend=False,
                    )
                    plotly_font(fig_bar)
                    st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info("선택한 측정 항목이 생육 데이터에 없습니다.")

            # ── 생중량 분포 (박스플롯) ───────────────────────────────────────
            st.subheader("📦 생중량 분포")
            if "생중량" in growth_df.columns and n_growth >= 2:
                s_full2 = pd.to_numeric(growth_df["생중량"], errors="coerce")
                g2 = s_full2.dropna()
                if "개체번호" in growth_df.columns:
                    ids2 = growth_df.loc[g2.index, "개체번호"]
                else:
                    ids2 = [f"#{i}" for i in g2.index]

                if len(g2) >= 2:
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=g2.values,
                        name="생중량",
                        fillcolor="#3498db",
                        line_color="#2c3e50",
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=-1.5,
                        marker=dict(color="#3498db", size=5),
                        hovertext=[f"개체 {i}  {v:.2f}g" for i, v in zip(ids2, g2.values)],
                        hoverinfo="text",
                    ))
                    q1b = g2.quantile(0.25)
                    q3b = g2.quantile(0.75)
                    fig_box.add_hline(
                        y=q3b, line_dash="dash", line_color="#e74c3c",
                        annotation_text="상위 25% 기준", annotation_position="top right",
                    )
                    fig_box.add_hline(
                        y=q1b, line_dash="dash", line_color="#2ecc71",
                        annotation_text="하위 25% 기준", annotation_position="bottom right",
                    )
                    fig_box.update_layout(
                        height=350,
                        yaxis_title="생중량 (g)",
                        showlegend=False,
                    )
                    plotly_font(fig_box)
                    st.plotly_chart(fig_box, width="stretch")

                    st_cols_d = st.columns(4)
                    st_cols_d[0].metric(
                        "중앙값(median)",
                        f"{g2.median():.2f}g",
                        help="데이터를 크기순으로 나열했을 때 정 가운데 값. 이상치 영향이 적음.",
                    )
                    st_cols_d[1].metric(
                        "Q1 (하위 25%)",
                        f"{q1b:.2f}g",
                        help="하위 25% 경계값. 이 값보다 작은 개체는 하위 그룹.",
                    )
                    st_cols_d[2].metric(
                        "Q3 (상위 25%)",
                        f"{q3b:.2f}g",
                        help="상위 25% 경계값. 이 값보다 큰 개체는 상위 그룹.",
                    )
                    st_cols_d[3].metric(
                        "IQR(사분위 범위)",
                        f"{q3b - q1b:.2f}g",
                        help="Q3 - Q1. 중간 50% 데이터의 퍼짐 정도.",
                    )
                else:
                    st.info("생중량 데이터가 충분하지 않습니다.")
            else:
                st.info("생중량 열이 없어 분포 그래프를 표시할 수 없습니다.")

            # ── 상관관계 분석 (산점도 + 추세선) ─────────────────────────────
            st.subheader("🔗 상관관계 분석")
            st.caption("상관계수(r)는 -1~1 사이 값. 0에 가까우면 관계 약함, ±1에 가까우면 강한 선형 관계.")
            corr_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이"] if c in growth_df.columns]
            if "생중량" in growth_df.columns and corr_cols:
                sg = growth_df[["생중량"] + corr_cols].copy()
                sg["생중량"] = pd.to_numeric(sg["생중량"], errors="coerce")
                for c in corr_cols:
                    sg[c] = pd.to_numeric(sg[c], errors="coerce")
                sg = sg.dropna()
                if len(sg) >= 2:
                    n_corr = len(corr_cols)
                    fig_corr = make_subplots(
                        rows=1, cols=n_corr,
                        subplot_titles=[f"{c} vs 생중량" for c in corr_cols],
                        horizontal_spacing=0.12,
                    )
                    for i, col in enumerate(corr_cols, 1):
                        r_val, _ = pearsonr_safe(sg[col], sg["생중량"])
                        a, b = trendline(sg[col], sg["생중량"])
                        fig_corr.add_trace(go.Scatter(
                            x=sg[col], y=sg["생중량"],
                            mode="markers",
                            marker=dict(color="#3498db", size=8, opacity=0.7,
                                        line=dict(width=1, color="white")),
                            name=col,
                            hovertext=[f"{c}={x:.1f}  생중량={y:.2f}g"
                                       for x, y in zip(sg[col], sg["생중량"])],
                            hoverinfo="text",
                        ), row=1, col=i)
                        if a is not None:
                            xs = np.linspace(sg[col].min(), sg[col].max(), 100)
                            fig_corr.add_trace(go.Scatter(
                                x=xs, y=a*xs + b,
                                mode="lines",
                                line=dict(color="#e74c3c", width=2, dash="dash"),
                                name=f"추세선 (r={r_val:.2f})",
                            ), row=1, col=i)
                        fig_corr.add_annotation(
                            text=f"r = {r_val:.2f}",
                            xref="paper", yref="paper",
                            x=0.95, y=0.05, xanchor="right", yanchor="bottom",
                            showarrow=False, font=dict(size=11, color="#2c3e50"),
                            row=1, col=i,
                        )
                    fig_corr.update_layout(
                        height=400, showlegend=False,
                    )
                    plotly_font(fig_corr)
                    st.plotly_chart(fig_corr, width="stretch")
                else:
                    st.info("상관계수·산점도 계산에 필요한 데이터가 충분하지 않습니다.")
            else:
                st.info("생중량 또는 비교할 측정 항목 열이 없어 상관관계 분석을 할 수 없습니다.")

            # ── 생중량 상위 25% vs 하위 25% 비교 표 ────────────────────────
            st.subheader("🏅 생중량 상위 25% vs 하위 25% 비교")
            if "생중량" in growth_df.columns and n_growth >= 4:
                sg2 = growth_df[["생중량"] + [c for c in measure_cols if c in growth_df.columns]].copy()
                sg2["생중량"] = pd.to_numeric(sg2["생중량"], errors="coerce")
                for c in [c for c in measure_cols if c in sg2.columns]:
                    sg2[c] = pd.to_numeric(sg2[c], errors="coerce")
                sg2 = sg2.dropna(subset=["생중량"])
                if len(sg2) >= 4:
                    q1t  = sg2["생중량"].quantile(0.25)
                    q3t  = sg2["생중량"].quantile(0.75)
                    top_df    = sg2[sg2["생중량"] >= q3t]
                    bottom_df = sg2[sg2["생중량"] <= q1t]
                    if len(top_df) > 0 and len(bottom_df) > 0:
                        comp_data = {
                            "구분":     ["상위 25%", "하위 25%"],
                            "개체 수":  [len(top_df), len(bottom_df)],
                        }
                        for c in [c for c in measure_cols if c in sg2.columns]:
                            comp_data[c] = [
                                round(top_df[c].mean(), 2),
                                round(bottom_df[c].mean(), 2),
                            ]
                        st.dataframe(pd.DataFrame(comp_data), width="stretch", hide_index=True)
                    else:
                        st.info("상위 또는 하위 25%에 해당하는 개체가 없습니다.")
                else:
                    st.info("비교할 데이터가 충분하지 않습니다.")
            else:
                st.info("생중량 열이 없어 상위·하위 비교를 할 수 없습니다.")

            # ── 원본 테이블 + 다운로드 ───────────────────────────────────────
            with st.expander("📄 생육 결과 데이터 원본 테이블 + CSV 다운로드"):
                if n_growth > 0:
                    st.dataframe(growth_df, width="stretch")
                    csv_download(growth_df, f"{school}_생육결과데이터")
                else:
                    st.info("표시할 생육 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
