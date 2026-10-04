# 离线过境预报后端

基于 Python 3.10 + FastAPI 0.115.12 + sgp4 2.26 的离线卫星过境预报服务。
地面站提前安排接收：输入卫星两行 TLE、站点（含遮挡）与 UTC 窗口，
输出天线/接收机跟踪区间与每秒跟踪表（方位、仰角、斜距、距离变化率、多普勒）。

## 运行

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

自测：`.venv/bin/python -m pytest tests -q`

## API

### POST /api/passes
返回 JSON 摘要：各可见区间（按起点、卫星 ID 排序）、持续秒数、
按秒采样的最高仰角及其时刻、窗口边界截断标记。

### POST /api/passes/download
同上计算，返回 ZIP：
- `summary.json`：与上面一致的摘要及单位说明；
- `csv/<卫星>_<站点>_<序号>.csv`：区间内每秒一行，列为
  `time_utc, azimuth_deg, elevation_deg, range_km, range_rate_km_s, doppler_shift_hz`。
  多普勒偏移 = `-f_downlink * range_rate / c`，**距离增加对应负偏移**。

### GET /api/health

## 请求格式

见 `examples/request.json`（可复现的 ISS TLE + 含遮挡的北京站）：

```json
{
  "window": {"start": "2024-01-01T02:00:00Z", "end": "2024-01-01T04:30:00Z"},
  "satellites": [
    {"id": "ISS",
     "tle_line1": "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9009",
     "tle_line2": "2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49560532    19",
     "downlink_frequency_hz": 145800000.0}
  ],
  "stations": [
    {"id": "BEIJING", "lat_deg": 39.9042, "lon_deg": 116.4074, "alt_m": 50.0,
     "mask": [[0.0, 10.0], [90.0, 25.0], [180.0, 5.0], [270.0, 15.0], [359.0, 10.0]]}
  ]
}
```

约束与校验：
- 最多 4 颗卫星、4 个站点；窗口 ≤ 24 小时且须带 UTC 时区；
- TLE：69 列宽、逐行校验和、两行卫星号一致；查询窗口离历元超过 7 天拒绝；
- 拒绝重复 ID、纬度超 [-90,90]、经度超 [-180,180]、NaN/Inf 等非有限数；
- `mask` 为 `[方位deg, 最低仰角deg]` 节点（方位 [0,360)，正北顺时针），
  排序后按方位线性插值并跨 0° 环绕；缺省为 0° 地平。

## 算法与近似范围

- 轨道：SGP4/SDP4（sgp4 2.26，WGS72 引力模型），输出 TEME；
- 坐标：UTC 近似 UT1（差 < 0.9 s）计算 GMST（IAU 1982），仅绕 z 轴
  旋转 GMST 将 TEME 转地固；**不计**极移、章动、大气折射与光行时；
- 站址：WGS84 经纬高转 ECEF；方位为正北顺时针，仰角、斜距由 ENU 矢量得到；
- 距离变化率：ECEF 速度（扣除地球自转 ω×r）在视线方向投影；
- 搜索：1 秒网格判定“仰角严格高于遮挡”，交叉时刻二分至 0.1 秒；
  遮挡可将一次过境分成多段；窗口边界截断以
  `truncated_at_start/end` 标记；相切（仰角恰好等于遮挡）不算有效区间；
  **不足约 1 秒的短窗口可能漏检**；
- 任一时刻 SGP4 传播失败，整份请求报错（HTTP 400）。

典型精度：位置百米~公里级（随 TLE 龄期增长），方向角约 0.1° 量级，
适用于接收计划编排，不适用于精密定轨。

## 双轴转台跟踪规划与回放

在原预报之上增加机械规划（`app/tracker.py`）、rotctld 协议客户端
（`app/rotctl.py`）、独占回放控制器（`app/playback.py`）与本机
转台模拟器（`app/rotctld_sim.py`），跨文件复用同一套传播与站点几何。

### POST /api/track/plan

请求体（见 `examples/track_request.json`，为跨北区间示例）：

- `forecast`：原 /api/passes 请求体；`interval_index`：按其返回顺序的区间编号；
- `az_min_deg`/`az_max_deg`：机械方位限位，跨度 ≤ 720°；
- `el_min_deg`/`el_max_deg`：仰角限位，须在 [0, 90]° 内；
- `max_az_rate_dps`/`max_el_rate_dps`：两轴最大角速度（°/s，正有限数）；
- `current_position`/`home_position`：当前与归位位置（须在限位内）；
- `preset_seconds`/`homing_seconds`：预置与归位秒数（正有限数）。

行为：

- 目标按 1 秒采样并包含区间两端点，`t_rel_s` 为相对首个跟踪点的秒数；
- 方位允许 +360·k 展开（不做过顶翻转），单段时长不得超过 30 分钟；
- 完整路径 当前→预置→跟踪→归位 的每段都须满足限位与速度约束；
  选取总方位转动最小者，同值取机械方位序列字典序最小者；
- 任一点不可行即整段拒绝（HTTP 422），不截角、不跳点。

### POST /api/playback · GET /api/playback · POST /api/playback/cancel

提交体：`{"plan": <上一步响应>, "host": "127.0.0.1", "port": 4533,
"position_tolerance_deg": 2.0, "response_timeout_s": 5.0}`。
控制器独占运行（重复提交返回 409），按单调时钟的相对时间向本机
rotctld TCP 端点逐点下发；启动时先用 `p` 核对实际位置。协议为换行
分帧：`P <az> <el>` 设位、`p` 读位、`S` 停止，非零 `RPRT` 视为错误。
超时、断连或取消会停止后续指令、尽力发 `S` 并释放占用，查询接口保留
真实终态（`completed`/`failed`/`cancelled` 及进度、最后位置）。

### 本机联调演示

\`\`\`bash
# 终端 1：转台模拟器（5°/s 恒速 slew，初始位置 0,0）
.venv/bin/python -m app.rotctld_sim --port 4533
# 终端 2：预报服务
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# 终端 3
curl -X POST localhost:8000/api/track/plan -d @examples/track_request.json \
     -H 'Content-Type: application/json' > plan.json
# 把模拟器先移到计划起点（示例为 0,5），再提交回放
printf 'P 0.000 5.000\n' | nc 127.0.0.1 4533
curl -X POST localhost:8000/api/playback -H 'Content-Type: application/json' \
     -d "{\"plan\": $(cat plan.json), \"port\": 4533}"
curl localhost:8000/api/playback          # 查询状态
curl -X POST localhost:8000/api/playback/cancel   # 取消
\`\`\`

单位：角度为度（机械方位可超出 [0,360)），角速度为度/秒，时间为秒；
预置/归位段按 1 秒线性斜坡下发。回放仅按相对时间发设位指令，
不闭环校正转台实际跟踪误差。
