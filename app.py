import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
# ============================================================
# 0. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="혈액 배송 의사결정 대시보드",
    layout="wide"
)

# ============================================================
# 0-1. 색상 설정
# ============================================================

COLOR_PRIMARY = "#7F1D1D"      # 진한 혈액색
COLOR_SECONDARY = "#991B1B"    # 메인 레드
COLOR_ACCENT = "#DC2626"       # 강조 레드
COLOR_VEHICLE = "#7F1D1D"      # 차량 경로
COLOR_DRONE = "#B45309"        # 드론 경로
COLOR_CURRENT = "#F97316"      # 현재 위치
COLOR_HOSPITAL = "rgba(127,29,29,0.65)"
COLOR_BG = "#F8FAFC"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {COLOR_BG};
    }}

    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }}

    .title-box {{
        background: linear-gradient(120deg, {COLOR_PRIMARY} 0%, {COLOR_SECONDARY} 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 14px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}

    .title-box h1 {{
    font-size: 1.85rem;
    margin: 0;
    font-weight: 800;
}}
    .title-box p {{
    font-size: 1.05rem;
    margin: 8px 0 0 0;
    color: #FEE2E2;
}}

    .kpi-box {{
    background: #FFFFFF;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.07);
    border-left: 4px solid #991B1B;
}}

    .kpi-label {{
    font-size: 0.95rem;
    color: #6B7280;
    font-weight: 700;
    margin-bottom: 6px;
}}

    .kpi-value {{
    font-size: 2.0rem;
    color: #111827;
    font-weight: 800;
    line-height: 1.1;
}}

    .panel-card {{
        background: #FFFFFF;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.07);
        margin-bottom: 14px;
    }}

    .panel-title {{
    font-size: 1.1rem;
    color: #374151;
    font-weight: 800;
    margin-bottom: 14px;
    line-height: 1.35;
}}
    button[data-baseweb="tab"] p {{
    font-size: 1.05rem;
    font-weight: 700;
}}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 1. 파일 경로
# ============================================================

PRED_PATH = "link_hour_congestion_summary.csv"
HOSPITAL_PATH = "병원노드(160).csv"
ROUTE_PATH = "sensitivity_vehicle.xlsx"
DRONE_SENS_PATH = "sensitivity1_combined.xlsx"

# ============================================================
# 2. 데이터 로드 함수
# ============================================================

@st.cache_data
def load_csv_auto(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    for enc in ["utf-8-sig", "cp949", "utf-8", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path)


@st.cache_data
def load_prediction(path):
    df = load_csv_auto(path)

    required_cols = ["link_id", "node_st", "node_ed", "hour", "main_pred_congestion"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"혼잡도 예측 파일에 필요한 컬럼이 없습니다: {missing}")

    df["link_id"] = df["link_id"].astype(str)
    df["node_st"] = df["node_st"].astype(str)
    df["node_ed"] = df["node_ed"].astype(str)
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")

    if "pred_travel_time_min" not in df.columns:
        df["pred_travel_time_min"] = np.nan

    return df


@st.cache_data
def load_hospital(path):
    df = load_csv_auto(path)

    df = df.rename(columns={
        "x": "lon",
        "y": "lat",
        "node_name": "hospital_node",
        "node": "hospital_node",
        "latitude": "lat",
        "longitude": "lon"
    })

    required_cols = ["병원명", "lat", "lon"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"병원 파일에 필요한 컬럼이 없습니다: {missing}")

    if "hospital_node" not in df.columns:
        df["hospital_node"] = ""

    if "일반병상수" not in df.columns:
        df["일반병상수"] = 1

    df["병원명"] = df["병원명"].astype(str)
    df["hospital_node"] = df["hospital_node"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["일반병상수"] = pd.to_numeric(df["일반병상수"], errors="coerce").fillna(1)

    df = df.dropna(subset=["lat", "lon"]).copy()
    df["병원명_clean"] = df["병원명"].str.replace(" ", "", regex=False)

    return df


@st.cache_data
def load_route_excel(path):
    route_detail = pd.read_excel(path, sheet_name="route_detail")
    summary = pd.read_excel(path, sheet_name="summary")

    required_cols = [
        "urgency_scenario", "depot", "mode", "hospital",
        "vehicle_no", "tour_no", "visit_order",
        "arrival_min", "tour_elapsed_min", "seed", "drone_count"
    ]

    missing = [c for c in required_cols if c not in route_detail.columns]

    if missing:
        raise ValueError(f"route_detail 시트에 필요한 컬럼이 없습니다: {missing}")

    for c in ["urgency_scenario", "depot", "mode", "hospital"]:
        route_detail[c] = route_detail[c].astype(str)

    route_detail["hospital_clean"] = route_detail["hospital"].str.replace(" ", "", regex=False)

    if "urgency_scenario" in summary.columns:
        summary["urgency_scenario"] = summary["urgency_scenario"].astype(str)

    return route_detail, summary


@st.cache_data
def load_drone_sensitivity_optional(path):
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_excel(path, sheet_name="summary")

        if "urgency_scenario" in df.columns:
            df["urgency_scenario"] = df["urgency_scenario"].astype(str)

        return df

    except Exception:
        return None


pred_df = load_prediction(PRED_PATH)
hosp_df = load_hospital(HOSPITAL_PATH)
route_all, summary_vehicle = load_route_excel(ROUTE_PATH)
summary_drone = load_drone_sensitivity_optional(DRONE_SENS_PATH)

# ============================================================
# 3. 병원 좌표 매칭
# ============================================================

hosp_coord = {
    row["병원명_clean"]: (float(row["lat"]), float(row["lon"]))
    for _, row in hosp_df.iterrows()
}


def get_hosp_coord(name):
    key = str(name).replace(" ", "")

    if key in hosp_coord:
        return hosp_coord[key]

    for k, v in hosp_coord.items():
        if key in k or k in key:
            return v

    return None, None


def get_hosp_display_name(name):
    if name in ["혈액원", "-"]:
        return name

    key = str(name).replace(" ", "")

    exact = hosp_df[hosp_df["병원명_clean"] == key]

    if not exact.empty:
        return exact.iloc[0]["병원명"]

    for _, row in hosp_df.iterrows():
        k = row["병원명_clean"]

        if key in k or k in key:
            return row["병원명"]

    return str(name)

# ============================================================
# 4. 경로 관련 함수
# ============================================================

def make_straight_path(lat1, lon1, lat2, lon2):
    return [float(lat1), float(lat2)], [float(lon1), float(lon2)]


@st.cache_data(show_spinner=False)
def get_osrm_tour_route(stops_tuple):
    """
    차량 1개 tour 전체를 OSRM에 한 번에 요청.
    stops_tuple = ((lat, lon), (lat, lon), ...)
    """
    stops = list(stops_tuple)

    if len(stops) < 2:
        return [], []

    coord_str = ";".join([f"{lon},{lat}" for lat, lon in stops])
    url = f"https://router.project-osrm.org/route/v1/driving/{coord_str}"

    params = {
        "overview": "full",
        "geometries": "geojson"
    }

    try:
        r = requests.get(url, params=params, timeout=15)

        if r.status_code != 200:
            lats = [p[0] for p in stops]
            lons = [p[1] for p in stops]
            return lats, lons

        data = r.json()

        if "routes" not in data or len(data["routes"]) == 0:
            lats = [p[0] for p in stops]
            lons = [p[1] for p in stops]
            return lats, lons

        coords = data["routes"][0]["geometry"]["coordinates"]

        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]

        return lats, lons

    except Exception:
        lats = [p[0] for p in stops]
        lons = [p[1] for p in stops]
        return lats, lons


@st.cache_data(show_spinner=False)
def get_osrm_segment_route(lat1, lon1, lat2, lon2):
    stops = ((float(lat1), float(lon1)), (float(lat2), float(lon2)))
    return get_osrm_tour_route(stops)


def point_along_polyline(lats, lons, ratio):
    ratio = min(max(float(ratio), 0.0), 1.0)

    if len(lats) == 0:
        return None, None

    if len(lats) == 1:
        return lats[0], lons[0]

    seg_lengths = []

    for i in range(len(lats) - 1):
        d = np.sqrt((lats[i + 1] - lats[i]) ** 2 + (lons[i + 1] - lons[i]) ** 2)
        seg_lengths.append(d)

    total = sum(seg_lengths)

    if total <= 0:
        return lats[0], lons[0]

    target = total * ratio
    cum = 0

    for i, d in enumerate(seg_lengths):
        if cum + d >= target:
            local_ratio = (target - cum) / max(d, 1e-9)

            lat = (1 - local_ratio) * lats[i] + local_ratio * lats[i + 1]
            lon = (1 - local_ratio) * lons[i] + local_ratio * lons[i + 1]

            return lat, lon

        cum += d

    return lats[-1], lons[-1]


def get_point(label, depot_lat, depot_lon):
    if label == "혈액원":
        return depot_lat, depot_lon

    return get_hosp_coord(label)


def get_tour_window(g):
    g = g.sort_values("visit_order").copy()

    if "depart_min" in g.columns and pd.notna(g["depart_min"].iloc[0]):
        start_t = float(g["depart_min"].iloc[0])
    else:
        start_t = 0.0

    if "tour_elapsed_min" in g.columns and pd.notna(g["tour_elapsed_min"].iloc[0]):
        end_t = start_t + float(g["tour_elapsed_min"].iloc[0])
    else:
        end_t = float(g["arrival_min"].max())

    return start_t, end_t


def get_route_status(group_df, current_time_min):
    g = group_df.sort_values("visit_order").copy()

    if g.empty:
        return None

    visits = g.to_dict("records")
    depart_min, end_time = get_tour_window(g)

    if current_time_min < depart_min:
        return {
            "status": "출발 전",
            "from_label": "혈액원",
            "to_label": visits[0]["hospital"],
            "progress": 0.0,
            "completed": [],
            "future": visits
        }

    completed = [v for v in visits if float(v["arrival_min"]) <= current_time_min]
    future = [v for v in visits if float(v["arrival_min"]) > current_time_min]

    if future:
        next_visit = future[0]

        if completed:
            from_label = completed[-1]["hospital"]
            prev_time = float(completed[-1]["arrival_min"])
        else:
            from_label = "혈액원"
            prev_time = depart_min

        to_label = next_visit["hospital"]
        next_time = float(next_visit["arrival_min"])
        progress = (current_time_min - prev_time) / max(next_time - prev_time, 1e-6)

        return {
            "status": "이동 중",
            "from_label": from_label,
            "to_label": to_label,
            "progress": progress,
            "completed": completed,
            "future": future
        }

    if current_time_min <= end_time:
        last_visit = visits[-1]
        prev_time = float(last_visit["arrival_min"])
        progress = (current_time_min - prev_time) / max(end_time - prev_time, 1e-6)

        return {
            "status": "복귀 중",
            "from_label": last_visit["hospital"],
            "to_label": "혈액원",
            "progress": progress,
            "completed": visits,
            "future": []
        }

    return {
        "status": "완료",
        "from_label": "혈액원",
        "to_label": "-",
        "progress": 1.0,
        "completed": visits,
        "future": []
    }


def select_active_tours(route_df, current_time_min):
    active = []

    for (mode, vehicle_no), vdf in route_df.groupby(["mode", "vehicle_no"], sort=True):
        for tour_no, g in vdf.groupby("tour_no", sort=True):
            start_t, end_t = get_tour_window(g)

            if start_t <= current_time_min <= end_t:
                active.append((mode, vehicle_no, tour_no, g.copy()))
                break

    return active


def build_tour_stops(group_df, depot_lat, depot_lon):
    stops = [(float(depot_lat), float(depot_lon))]

    g = group_df.sort_values("visit_order").copy()

    for _, row in g.iterrows():
        lat, lon = get_hosp_coord(row["hospital"])

        if lat is not None:
            stops.append((float(lat), float(lon)))

    stops.append((float(depot_lat), float(depot_lon)))

    return tuple(stops)


def add_route_line(fig, lats, lons, color, width, opacity, hover_text):
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="lines",
        line=dict(width=width + 4, color="rgba(255,255,255,0.82)"),
        opacity=1.0,
        hoverinfo="skip",
        showlegend=False
    ))

    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="lines",
        line=dict(width=width, color=color),
        opacity=opacity,
        hoverinfo="text",
        text=hover_text,
        showlegend=False
    ))

# ============================================================
# 5. 사이드바: 시나리오 설정
# ============================================================

# with st.sidebar:
#     st.header("시나리오 설정")

#     scenario_options = sorted(route_all["urgency_scenario"].dropna().unique())
#     selected_scenario = st.selectbox(
#         "긴급도",
#         scenario_options,
#         format_func=lambda x: {"high": "높음", "medium": "중간", "low": "낮음"}.get(x, x)
#     )

#     depot_options = sorted(route_all["depot"].dropna().unique())
#     selected_depot = st.selectbox("혈액원", depot_options)

#     if "vehicle_count" in route_all.columns:
#         vehicle_options = sorted(route_all["vehicle_count"].dropna().unique())
#         selected_vehicle_count = st.selectbox("차량 수", vehicle_options)
#     else:
#         selected_vehicle_count = None

#     drone_options = sorted(route_all["drone_count"].dropna().unique())
#     selected_drone_count = st.selectbox("드론 수", drone_options)

#     st.divider()

#     hour_options = sorted([int(h) for h in pred_df["hour"].dropna().unique()])
#     start_hour = st.selectbox(
#         "배송 시작 시간",
#         hour_options,
#         index=min(8, len(hour_options) - 1)
#     )

#     st.divider()

#     show_hospital_nodes = st.checkbox("병원 노드 표시", value=False)
#     show_route = st.checkbox("배송 경로 표시", value=True)
#     show_full_tour = st.checkbox("현재 투어 전체 경로 표시", value=True)

# ============================================================
# 5. 사이드바: 시나리오 설정
# ============================================================

with st.sidebar:
    st.header("시나리오 설정")

    scenario_options = sorted(route_all["urgency_scenario"].dropna().unique())
    selected_scenario = st.selectbox(
        "긴급도",
        scenario_options,
        format_func=lambda x: {"high": "높음", "medium": "중간", "low": "낮음"}.get(x, x)
    )

    depot_options = sorted(route_all["depot"].dropna().unique())
    selected_depot = st.selectbox("혈액원", depot_options)

    if "vehicle_count" in route_all.columns:
        vehicle_options = sorted(route_all["vehicle_count"].dropna().unique())
        selected_vehicle_count = st.selectbox("차량 수", vehicle_options)
    else:
        selected_vehicle_count = None

    drone_options = sorted(route_all["drone_count"].dropna().unique())
    selected_drone_count = st.selectbox("드론 수", drone_options)

    # st.divider()

    start_hour = 10
    # st.caption("배송 시작 시간: 10시 기준")

    st.divider()

    show_hospital_nodes = st.checkbox("병원 노드 표시", value=False)
    show_route = st.checkbox("배송 경로 표시", value=True)
    show_full_tour = st.checkbox("현재 투어 전체 경로 표시", value=True)

# ============================================================
# 6. Seed 자동 선택 및 선택 route 생성
# ============================================================

seed_base = route_all[
    (route_all["urgency_scenario"] == selected_scenario) &
    (route_all["drone_count"] == selected_drone_count) &
    (route_all["depot"] == selected_depot)
].copy()

if selected_vehicle_count is not None and "vehicle_count" in seed_base.columns:
    seed_base = seed_base[seed_base["vehicle_count"] == selected_vehicle_count].copy()

if seed_base.empty:
    auto_seed = None
else:
    auto_seed = seed_base["seed"].dropna().min()

selected_route = seed_base.copy()

if auto_seed is not None:
    selected_route = selected_route[selected_route["seed"] == auto_seed].copy()

if selected_drone_count == 0:
    selected_route = selected_route[selected_route["mode"] != "drone"].copy()

selected_route = selected_route.sort_values(
    ["mode", "vehicle_no", "tour_no", "visit_order"]
)

# ============================================================
# 7. 배송 진행 시간 및 현재 혼잡 시간 계산
# ============================================================

if not selected_route.empty:
    max_time = int(np.ceil(selected_route["tour_elapsed_min"].max()))
else:
    max_time = 1

with st.sidebar:
    playback_speed = st.selectbox(
        "영상 속도",
        options=[0.5, 1, 2, 4, 8],
        index=1,
        format_func=lambda x: f"{x}배속"
    )

current_time_min = 0
current_hour = int((int(start_hour) + int(current_time_min // 60)) % 24)
hour_df = pred_df[pred_df["hour"] == current_hour].copy()

# Plotly animation frames가 지도 내부에서 시간을 움직이므로,
# Streamlit 화면 전체를 자동 새로고침하지 않습니다.
current_time_min = 0

current_hour = int((int(start_hour) + int(current_time_min // 60)) % 24)
hour_df = pred_df[pred_df["hour"] == current_hour].copy()

# ============================================================
# 8. 혈액원 좌표 자동 설정
# ============================================================

depot_defaults = {
    "중앙": (37.5636, 126.9786),
    "남부": (37.5172, 126.9368),
    "동부": (37.5657, 127.0398)
}

depot_lat, depot_lon = depot_defaults.get(selected_depot, (37.5665, 126.9780))

# ============================================================
# 9. Summary 필터링
# ============================================================

summary_filter = summary_vehicle[
    (summary_vehicle["urgency_scenario"] == selected_scenario) &
    (summary_vehicle["drone_count"] == selected_drone_count)
].copy()

if selected_vehicle_count is not None and "vehicle_count" in summary_filter.columns:
    summary_filter = summary_filter[summary_filter["vehicle_count"] == selected_vehicle_count].copy()

# ============================================================
# 10. KPI 계산
# ============================================================

n_congested = int((hour_df["main_pred_congestion"] == "정체").sum()) if not hour_df.empty else 0
n_slow = int((hour_df["main_pred_congestion"] == "서행").sum()) if not hour_df.empty else 0
n_smooth = int((hour_df["main_pred_congestion"] == "원활").sum()) if not hour_df.empty else 0

late_count = int(selected_route["is_late"].fillna(0).sum()) if not selected_route.empty else 0
vehicle_used = selected_route[selected_route["mode"] == "vehicle"]["vehicle_no"].nunique() if not selected_route.empty else 0
drone_used = selected_route[selected_route["mode"] == "drone"]["vehicle_no"].nunique() if not selected_route.empty else 0

if not summary_filter.empty and "T_max_mean" in summary_filter.columns:
    tmax_value = float(summary_filter["T_max_mean"].iloc[0])
else:
    tmax_value = np.nan

scenario_label = {
    "high": "높음",
    "medium": "중간",
    "low": "낮음"
}.get(selected_scenario, selected_scenario)

vehicle_label = selected_vehicle_count if selected_vehicle_count is not None else "-"

# ============================================================
# 11. Header
# ============================================================

st.markdown(
    f"""
    <div class="title-box">
        <h1>혈액 배송 의사결정 대시보드</h1>
        <p>{selected_depot} 혈액원 | 차량 {vehicle_label}대 · 드론 {selected_drone_count}대</p>
    </div>
    """,
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-label">총 배송 시뮬레이션 시간</div>
            <div class="kpi-value">{max_time}분</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-label">지연 병원 수</div>
            <div class="kpi-value">{late_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-label">운행 자원</div>
            <div class="kpi-value">{vehicle_used + drone_used}대</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-label">평균 지연</div>
            <div class="kpi-value">{"-" if pd.isna(tmax_value) else f"{tmax_value:.1f}분"}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
# ============================================================
# 12. 탭
# ============================================================

tab_map, tab_detail, tab_sens = st.tabs([
    "배송 진행 지도",
    "경로 상세",
    "민감도 분석"
])

# ============================================================
# Tab 1. 배송 진행 지도 - 지도 전체너비 + 아래 카드 요약
# ============================================================

with tab_map:
    frame_step = 5
    frame_times = list(range(0, max(max_time, 1) + 1, frame_step))
    if len(frame_times) == 0:
        frame_times = [0]

    def make_map_traces_for_time(t):
        traces = []
        traces.append(go.Scattermapbox(lat=[None], lon=[None], mode="markers",
            marker=dict(size=9, color=COLOR_HOSPITAL), name="병원"))
        traces.append(go.Scattermapbox(lat=[None], lon=[None], mode="markers",
            marker=dict(size=14, color=COLOR_ACCENT), name="혈액원"))
        traces.append(go.Scattermapbox(lat=[None], lon=[None], mode="lines",
            line=dict(width=5, color=COLOR_VEHICLE), name="차량 경로"))
        if selected_drone_count > 0:
            traces.append(go.Scattermapbox(lat=[None], lon=[None], mode="lines",
                line=dict(width=4, color=COLOR_DRONE), name="드론 경로"))
        traces.append(go.Scattermapbox(lat=[None], lon=[None], mode="markers",
            marker=dict(size=14, color=COLOR_CURRENT), name="현재 위치"))

        if show_hospital_nodes:
            max_beds = hosp_df["일반병상수"].max()
            marker_size = 5 + (hosp_df["일반병상수"] / max_beds) * 10 if max_beds > 0 else 6
            traces.append(go.Scattermapbox(
                lat=hosp_df["lat"], lon=hosp_df["lon"], mode="markers",
                marker=dict(size=marker_size, color=COLOR_HOSPITAL, opacity=0.75),
                text="병원명: " + hosp_df["병원명"].astype(str) + "<br>병상수: " + hosp_df["일반병상수"].astype(str),
                hoverinfo="text", showlegend=False))

        traces.append(go.Scattermapbox(
            lat=[depot_lat], lon=[depot_lon], mode="markers+text",
            marker=dict(size=24, color=COLOR_ACCENT, opacity=0.95),
            text=[f"{selected_depot} 혈액원"], textposition="top right",
            hoverinfo="text", hovertext=f"{selected_depot} 혈액원<br>차량·드론 출발 및 복귀 거점",
            showlegend=False))

        active_tours = select_active_tours(selected_route, t)
        if show_route and active_tours:
            for mode, vehicle_no, tour_no, group in active_tours:
                group = group.sort_values("visit_order")
                status = get_route_status(group, t)
                if status is None:
                    continue
                color = COLOR_VEHICLE if mode == "vehicle" else COLOR_DRONE
                width = 5 if mode == "vehicle" else 4
                label = f"차량 {vehicle_no}" if mode == "vehicle" else f"드론 {vehicle_no}"

                if mode == "vehicle" and show_full_tour:
                    stops = build_tour_stops(group, depot_lat, depot_lon)
                    if len(stops) >= 2:
                        tour_lats, tour_lons = get_osrm_tour_route(stops)
                        traces.append(go.Scattermapbox(lat=tour_lats, lon=tour_lons, mode="lines",
                            line=dict(width=7, color="rgba(255,255,255,0.82)"), opacity=1.0,
                            hoverinfo="skip", showlegend=False))
                        traces.append(go.Scattermapbox(lat=tour_lats, lon=tour_lons, mode="lines",
                            line=dict(width=3, color=color), opacity=0.28,
                            hoverinfo="text", text=f"{label} / 투어 {tour_no} 전체 경로",
                            showlegend=False))

                if status["status"] in ["이동 중", "복귀 중"]:
                    from_label = status["from_label"]
                    to_label   = status["to_label"]
                    lat1, lon1 = get_point(from_label, depot_lat, depot_lon)
                    lat2, lon2 = get_point(to_label,   depot_lat, depot_lon)
                    if lat1 is None or lat2 is None:
                        continue
                    if mode == "vehicle":
                        seg_lats, seg_lons = get_osrm_segment_route(lat1, lon1, lat2, lon2)
                    else:
                        seg_lats, seg_lons = make_straight_path(lat1, lon1, lat2, lon2)

                    traces.append(go.Scattermapbox(lat=seg_lats, lon=seg_lons, mode="lines",
                        line=dict(width=width+4, color="rgba(255,255,255,0.88)"), opacity=1.0,
                        hoverinfo="skip", showlegend=False))
                    traces.append(go.Scattermapbox(lat=seg_lats, lon=seg_lons, mode="lines",
                        line=dict(width=width, color=color), opacity=0.98,
                        hoverinfo="text",
                        text=(f"{label} / 투어 {tour_no}<br>"
                              f"{get_hosp_display_name(from_label)} → {get_hosp_display_name(to_label)}<br>"
                              f"상태: {status['status']}<br>"
                              f"진행률: {status['progress']*100:.1f}%"),
                        showlegend=False))

                    pos_lat, pos_lon = point_along_polyline(seg_lats, seg_lons, status["progress"])
                    if pos_lat is not None:
                        traces.append(go.Scattermapbox(lat=[pos_lat], lon=[pos_lon], mode="markers",
                            marker=dict(size=23, color="#FFFFFF", opacity=1),
                            hoverinfo="skip", showlegend=False))
                        traces.append(go.Scattermapbox(lat=[pos_lat], lon=[pos_lon], mode="markers",
                            marker=dict(size=15, color=COLOR_CURRENT, opacity=0.98),
                            hoverinfo="text",
                            hovertext=(f"{label} 현재 위치<br>투어 {tour_no}<br>"
                                       f"{get_hosp_display_name(from_label)} → {get_hosp_display_name(to_label)}<br>"
                                       f"배송 진행 시간: {t}분"),
                            showlegend=False))
        return traces

    center_points = [(depot_lat, depot_lon)]
    if not selected_route.empty:
        for h in selected_route["hospital"].dropna().unique():
            lat, lon = get_hosp_coord(h)
            if lat is not None:
                center_points.append((lat, lon))
    center_lat = np.mean([p[0] for p in center_points])
    center_lon = np.mean([p[1] for p in center_points])

    # ── 지도 (전체 너비) ──────────────────────────────────────────
    base_traces = make_map_traces_for_time(0)
    fig = go.Figure(data=base_traces)
    fig.frames = [
        go.Frame(data=make_map_traces_for_time(t), name=str(t))
        for t in frame_times
    ]
    frame_duration = int(800 / playback_speed)
    fig.update_layout(
        mapbox=dict(style="carto-positron",
                    center=dict(lat=center_lat, lon=center_lon), zoom=11.0),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.88)", bordercolor="#E5E7EB", borderwidth=1),
        updatemenus=[{
            "type": "buttons", "showactive": False,
            "x": 0.02, "y": 0.09, "xanchor": "left", "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.92)", "bordercolor": "#E5E7EB", "borderwidth": 1,
            "buttons": [
                {"label": "▶ 재생", "method": "animate",
                 "args": [None, {"frame": {"duration": frame_duration, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]},
                {"label": "⏸ 정지", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
            ]
        }],
        sliders=[{
            "active": 0, "x": 0.12, "y": 0.04, "len": 0.78,
            "xanchor": "left", "yanchor": "bottom",
            "currentvalue": {"prefix": "배송 진행 시간: ", "suffix": "분", "font": {"size": 15}},
            "pad": {"t": 35, "b": 10},
            "steps": [{"label": str(t), "method": "animate",
                        "args": [[str(t)], {"frame": {"duration": 0, "redraw": True},
                                            "transition": {"duration": 0}, "mode": "immediate"}]}
                       for t in frame_times]
        }]
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 지도 아래: 배송 요약 카드 ────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="font-size:1.05rem;font-weight:800;color:#374151;margin-bottom:12px;">전체 배송 스케줄 요약</p>',
        unsafe_allow_html=True
    )

    if selected_route.empty:
        st.info("선택한 조건에 해당하는 스케줄이 없습니다.")
    else:
        timeline_df = selected_route.copy()
        timeline_df["수단"] = timeline_df["mode"].map({"vehicle": "차량", "drone": "드론"}).fillna(timeline_df["mode"])
        timeline_df["병원명표시"] = timeline_df["hospital"].apply(get_hosp_display_name)
        timeline_df["지연여부"] = "정상"
        if "tardiness_min" in timeline_df.columns:
            timeline_df.loc[timeline_df["tardiness_min"].fillna(0) > 0, "지연여부"] = "지연"

        # 차량/드론 수 파악해서 컬럼 수 결정
        groups = list(timeline_df.groupby(["수단", "vehicle_no"], sort=True))
        n_cols = min(len(groups), 3)
        cols = st.columns(n_cols)

        for col_idx, ((수단, vno), vdf) in enumerate(groups):
            vdf = vdf.sort_values(["tour_no", "visit_order"])
            총방문 = len(vdf)
            지연수 = (vdf["지연여부"] == "지연").sum()
            마지막도착 = vdf["arrival_min"].max()
            투어수 = vdf["tour_no"].nunique()

            # 투어별 방문 배지 HTML
            tour_html = ""
            for tour_no, tdf in vdf.groupby("tour_no", sort=True):
                tdf = tdf.sort_values("visit_order")
                badges = ""
                for _, row in tdf.iterrows():
                    is_late = row["지연여부"] == "지연"
                    bg = "#DC2626" if is_late else "#7F1D1D"
                    hosp_name = str(row["병원명표시"])
                    # 이름이 길면 잘라서 표시
                    short_name = hosp_name[:6] + "…" if len(hosp_name) > 7 else hosp_name
                    arr = f"{float(row['arrival_min']):.0f}분"
                    badges += (
                        f'<span title="{hosp_name} ({arr})" style="'
                        f'display:inline-block;background:{bg};color:#fff;'
                        f'border-radius:6px;padding:3px 7px;margin:2px 2px;'
                        f'font-size:0.72rem;font-weight:700;cursor:default;">'
                        f'{short_name}<br><span style="font-weight:400;font-size:0.68rem;">{arr}</span>'
                        f'</span>'
                    )
                tour_html += (
                    f'<div style="margin-bottom:6px;">'
                    f'<span style="font-size:0.75rem;color:#6B7280;font-weight:700;">투어 {tour_no}</span><br>'
                    f'{badges}</div>'
                )

            icon = "🚗" if 수단 == "차량" else "🚁"
            late_badge = (
                f'<span style="background:#FEE2E2;color:#DC2626;border-radius:5px;'
                f'padding:2px 7px;font-size:0.75rem;font-weight:700;">지연 {지연수}건</span>'
                if 지연수 > 0 else
                '<span style="background:#D1FAE5;color:#065F46;border-radius:5px;'
                'padding:2px 7px;font-size:0.75rem;font-weight:700;">모두 정상</span>'
            )

            card = (
                f'<div style="background:#fff;border:1px solid #E5E7EB;border-radius:14px;'
                f'padding:14px 16px;height:100%;box-shadow:0 1px 4px rgba(0,0,0,0.06);">'
                f'<div style="font-size:1rem;font-weight:800;color:#111827;margin-bottom:4px;">'
                f'{icon} {수단} {vno}</div>'
                f'<div style="font-size:0.8rem;color:#6B7280;margin-bottom:8px;">'
                f'투어 {투어수}개 · 방문 {총방문}곳 '
                f'&nbsp;&nbsp;{late_badge}</div>'
                f'<div style="border-top:1px solid #F3F4F6;padding-top:8px;">'
                f'{tour_html}</div>'
                f'</div>'
            )

            with cols[col_idx % n_cols]:
                st.markdown(card, unsafe_allow_html=True)

# ============================================================
# Tab 2. 경로 상세
# ============================================================

with tab_detail:
    st.subheader("선택 시나리오 경로 상세")

    if selected_route.empty:
        st.warning("선택한 조건에 해당하는 경로 데이터가 없습니다.")
    else:
        display_cols = [
            "depot",
            "mode",
            "vehicle_no",
            "tour_no",
            "visit_order",
            "hospital",
            "arrival_min",
            "due_min",
            "tardiness_min",
            "is_late",
            "depart_min",
            "tour_elapsed_min"
        ]

        display_cols = [c for c in display_cols if c in selected_route.columns]

        st.dataframe(
            selected_route[display_cols],
            width="stretch",
            hide_index=True
        )

        st.markdown("#### 차량·드론별 요약")

        group_summary = (
            selected_route
            .groupby(["mode", "vehicle_no"], as_index=False)
            .agg(
                방문수=("hospital", "count"),
                지연수=("is_late", "sum"),
                최대도착=("arrival_min", "max"),
                투어수=("tour_no", "nunique")
            )
        )

        st.dataframe(
            group_summary,
            width="stretch",
            hide_index=True
        )

# ============================================================
# Tab 3. 민감도 분석
# ============================================================

with tab_sens:
    st.subheader("민감도 분석")

    sens_tab1, sens_tab2 = st.tabs([
        "드론 수 민감도",
        "차량 수 민감도"
    ])

    with sens_tab1:
        if summary_drone is None:
            st.warning("sensitivity1_combined.xlsx 파일이 없어 드론 수 민감도 분석을 표시하지 않습니다.")
        else:
            drone_scenarios = sorted(summary_drone["urgency_scenario"].dropna().unique())

            selected_drone_scenario = st.selectbox(
                "긴급도 선택",
                drone_scenarios,
                key="drone_sens_scenario"
            )

            df_d = summary_drone[
                summary_drone["urgency_scenario"] == selected_drone_scenario
            ].copy()

            df_d = df_d.sort_values("drone_count")

            left, right = st.columns([1.25, 1])

            with left:
                fig_d = px.line(
                    df_d,
                    x="drone_count",
                    y="T_max_mean",
                    markers=True,
                    labels={
                        "drone_count": "드론 수",
                        "T_max_mean": "평균 T_max"
                    },
                    title=f"드론 수에 따른 평균 완료시간 변화 ({selected_drone_scenario})"
                )

                fig_d.update_traces(line=dict(color=COLOR_SECONDARY, width=3))
                fig_d.update_layout(height=420)
                st.plotly_chart(fig_d, width="stretch")

            with right:
                show_cols = [
                    "urgency_scenario",
                    "drone_count",
                    "T_max_mean",
                    "T_max_std",
                    "time_mean_sec",
                    "n_ok"
                ]

                show_cols = [c for c in show_cols if c in df_d.columns]

                st.dataframe(
                    df_d[show_cols],
                    width="stretch",
                    hide_index=True
                )

    with sens_tab2:
        vehicle_scenarios = sorted(summary_vehicle["urgency_scenario"].dropna().unique())

        selected_vehicle_scenario = st.selectbox(
            "긴급도 선택",
            vehicle_scenarios,
            key="vehicle_sens_scenario"
        )

        fixed_drone_options = sorted(summary_vehicle["drone_count"].dropna().unique())

        fixed_drone_count = st.selectbox(
            "드론 수 고정",
            fixed_drone_options,
            key="fixed_drone_count"
        )

        df_v = summary_vehicle[
            (summary_vehicle["urgency_scenario"] == selected_vehicle_scenario) &
            (summary_vehicle["drone_count"] == fixed_drone_count)
        ].copy()

        df_v = df_v.sort_values("vehicle_count")

        left, right = st.columns([1.25, 1])

        with left:
            fig_v = px.line(
                df_v,
                x="vehicle_count",
                y="T_max_mean",
                markers=True,
                labels={
                    "vehicle_count": "차량 수",
                    "T_max_mean": "평균 T_max"
                },
                title=f"차량 수에 따른 평균 완료시간 변화 ({selected_vehicle_scenario}, 드론 {fixed_drone_count}대)"
            )

            fig_v.update_traces(line=dict(color=COLOR_SECONDARY, width=3))
            fig_v.update_layout(height=420)
            st.plotly_chart(fig_v, width="stretch")

        with right:
            show_cols = [
                "urgency_scenario",
                "drone_count",
                "vehicle_count",
                "T_max_mean",
                "T_max_std",
                "time_mean_sec",
                "late_mean",
                "n_ok"
            ]

            show_cols = [c for c in df_v.columns if c in show_cols]

            st.dataframe(
                df_v[show_cols],
                width="stretch",
                hide_index=True
            )
