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
空目标或畸形计划（`targets` 为空）在提交时即以 422 拒绝，时间线在提交前
同步构建，回放线程内任何未预期异常都会落为 `failed` 并释放控制器，
不会停留在 `running` 占用独占权。

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

## IQ 录波多普勒校正

在原预报之上增加录波校正（`app/sigmf.py` 校验、`app/doppler.py` 频移/相位/诊断、
`app/delivery.py` 打包、`app/main.py` 端点），跨文件复用同一套 SGP4 传播与站点几何。

### POST /api/doppler/correct

multipart/form-data 提交：

- `forecast`：原 /api/passes 请求体 JSON 字符串（含 TLE 与站点）；
- `satellite_id` / `station_id`：所选星、站 ID；
- `tx_frequency_hz`：正有限发射频率（Hz）；
- `meta` / `data`：SigMF 元数据（JSON）与样本文件。

录波约束（任一不满足即整份拒绝，HTTP 422；传播失败 400）：

- 仅单通道 `cf32_le`；恰好一个 `core:sample_start = 0` 的 capture，无头尾附加字节
  （数据文件长度必须恰好为样本数 × 8 字节）；
- 元数据须含 UTC 开录时间（`core:datetime`，带时区）、中心频率
  （`core:frequency`，正有限）与采样率（`core:sample_rate`，1 kHz ~ 200 kHz）；
- 拒绝空录波、截断（长度非 8 的倍数）、非有限样本与非法元数据；
  最多 2^20 复样本、时长 ≤ 60 s；
- 录波全段 `[t0, t0 + N/fs]` 须落在所选站星的同一可见区间内
  （用原 `find_passes` 判定）；TLE 历元距录波超过 7 天拒绝。

校正模型：

- 径向速度复用预报 CSV 的 ECEF 距离变化率，**远离为正**；
- 基带频移 `shift(t) = (f_tx - f_center) - f_tx · range_rate(t) / c`，
  即发射载波在录波基带中的位置；
- 节点按 1 秒间隔并含录波末端计算，节点间线性插值；
  任一节点 |shift| ≥ fs/2 即整份拒绝；
- 相位 `phi(0) = 0`、`phi' = shift`，逐样本梯形积分（分块处理、块间相位连续），
  样本乘 `exp(-j·phi)` 校正；**不重采样、不归一化**，样本数、采样率与幅度保持不变。

响应为 ZIP（`doppler_corrected.zip`）：

- `corrected.sigmf-meta`：原元数据副本，仅 `core:frequency` 改为发射频率，
  UTC 开录时间不变；上传的原件不修改；
- `corrected.sigmf-data`：校正后的 cf32_le 样本；
- `diagnostics.json`：频移节点表 + 逐窗诊断。诊断为不重叠 1024 点 Hann 窗，
  每窗给出校正前后峰频（Hz，带符号）与均方功率（线性，|x|² 均值）；
  不足一窗则 `windows` 为空。

### 可复现示例

```bash
# 生成含变频载波的示例录波（确定性合成，载波跟随预报多普勒 + 250 Hz 固定偏移）
.venv/bin/python examples/make_iq_recording.py
# 提交并下载
curl -X POST localhost:8000/api/doppler/correct \
  -F 'forecast=<examples/request.json' \
  -F satellite_id=ISS -F station_id=BEIJING \
  -F tx_frequency_hz=145800000.0 \
  -F meta=@examples/iq/iss_beijing.sigmf-meta \
  -F data=@examples/iq/iss_beijing.sigmf-data \
  -o corrected.zip
```

校正后载波应落在 250 Hz 所在 FFT bin（48 kHz / 1024 ≈ 46.9 Hz 分辨率），
各窗均方功率校正前后一致。

单位与近似：频率 Hz、采样率 Hz、时间秒（UTC）、功率为线性均方；
相位积分用梯形近似，节点间频移线性插值；几何与轨道近似同预报部分
（UTC≈UT1、GMST 旋转、不计光行时与折射）。诊断峰频分辨率为 fs/1024，
未做抛物线插值。

### SigMF 频率与通道约定

- 读写中心频率一律使用 capture 的 `core:frequency`；capture 缺省时
  回退到 global 的 `core:frequency`（两者均无则拒绝）。校正输出同时
  写入 capture 与 global。
- 拒绝多通道录波（`core:num_channels` 非 1）与任何非零附加字节
  （global/capture 的 `core:offset`、`core:header_bytes`、
  `core:trailing_bytes`）；数据文件长度必须恰好为样本数 × 8 字节。

## 未知窄带信标识别

在原预报之上增加未知载波识别（`app/beacon.py` 测量与判定、
`app/delivery.py` 打包、`app/main.py` 端点），复用同一套 SGP4 传播、
站点几何与录波校验。仅处理**单个稳定未调制载波**；不读取 SigMF
annotations 中的身份标签，识别完全基于频率轨迹匹配。

### POST /api/beacon/identify · POST /api/beacon/identify/download

multipart/form-data 提交：

- `forecast`：原 /api/passes 请求体 JSON 字符串（最多 4 颗候选卫星，
  各自 `downlink_frequency_hz` 即其发射频率）；
- `station_id`：站点 ID；
- `lo_offset_limit_hz` / `rms_limit_hz` / `separation_hz`：
  正有限数，单位 Hz（本振偏差上限、残差 RMS 上限、分离量）；
- `meta` / `data`：SigMF 元数据与样本文件（约束同录波校正部分）。

处理流程：

1. 测频：不重叠 1024 点 Hann 窗，谱峰用对数幅度三点抛物线细化
   （跨 FFT 环边界取模）；峰功率不超过该窗谱功率中位数 10 倍的窗剔除；
   有效窗 UTC 中心时刻逐窗保留，跨度不足 5 秒整份拒绝（422）。
2. 逐候选在窗中心预测基带频移
   `shift(t) = (f_tx - f_center) - f_tx·range_rate(t)/c`；
   全段不在同一可见区间、预测达到奈奎斯特界（fs/2）、TLE 过期或
   估计本振偏差超限的候选排除并给出原因。
3. 本振偏差 = mean(观测 − 预测)，去偏残差 RMS 排名。
4. 判定：无可比较候选或最低 RMS 超限为 `no-match`；次低与最低之差
   不大于分离量为 `ambiguous`（同分候选按 ID 列出，不冒称唯一识别）；
   其余为 `identified`。

JSON 端点返回状态、排名（含排除原因、本振偏差、残差 RMS）与窗口摘要；
download 端点返回 ZIP（`beacon_identification.zip`）：

- `summary.json`：与 JSON 端点一致的摘要；
- `observations.csv`：逐窗 `index,time_utc,measured_freq_hz,peak_power,median_power,accepted`；
- `candidates/<卫星ID>.csv`：每个可比较候选的逐窗
  `time_utc,observed_hz,predicted_hz,offset_hz,residual_hz`。

### 可复现识别与歧义示例

```bash
# 生成 12 s 确定性录波：载波跟随 ISS 预报多普勒 + 250 Hz 固定本振偏移；
# 元数据使用 capture 级 core:frequency
.venv/bin/python examples/make_beacon_recording.py

# 歧义：候选 ISS_ALT 与 ISS 同 TLE、下行高 300 Hz，残差差 < 0.5 Hz 分离量
curl -X POST localhost:8000/api/beacon/identify \
  -F 'forecast=<examples/beacon_request.json' \
  -F station_id=BEIJING -F lo_offset_limit_hz=500 \
  -F rms_limit_hz=1.0 -F separation_hz=0.5 \
  -F meta=@examples/iq/beacon.sigmf-meta \
  -F data=@examples/iq/beacon.sigmf-data
# -> {"status":"ambiguous","ambiguous_ids":["ISS","ISS_ALT"],...}

# 唯一识别：改用单候选 examples/request.json（同一录波）
curl -X POST localhost:8000/api/beacon/identify \
  -F 'forecast=<examples/request.json' \
  -F station_id=BEIJING -F lo_offset_limit_hz=500 \
  -F rms_limit_hz=1.0 -F separation_hz=0.5 \
  -F meta=@examples/iq/beacon.sigmf-meta \
  -F data=@examples/iq/beacon.sigmf-data
# -> {"status":"identified","identified":"ISS",...}

# 下载 ZIP（摘要 JSON + 观测 CSV + 候选残差 CSV）
curl -X POST localhost:8000/api/beacon/identify/download \
  -F 'forecast=<examples/beacon_request.json' \
  -F station_id=BEIJING -F lo_offset_limit_hz=500 \
  -F rms_limit_hz=1.0 -F separation_hz=0.5 \
  -F meta=@examples/iq/beacon.sigmf-meta \
  -F data=@examples/iq/beacon.sigmf-data \
  -o beacon_identification.zip
```

范围与近似：测频偏差约 ±0.7 Hz（对数抛物线、Hann 主瓣），
RMS 上限不宜低于约 1 Hz；同 TLE 近频候选在短录波上不可区分，
应通过分离量参数声明为歧义而非唯一识别；几何与轨道近似同预报部分。
