# NAND Flash MoE Simulator

NAND Flash MoE 仿真器 - 模拟 Mixture-of-Experts (MoE) 模型中专家参数存储在 NAND Flash 上的读取行为。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 项目简介

本项目是一个用于研究和优化 MoE 模型在 NAND Flash 存储系统上推理性能的仿真工具。它精确模拟了：

- NAND Flash 的硬件特性（多通道、多 Plane、Page 结构）
- tR（读延迟）和 tX（传输延迟）的并行/串行关系
- Intra-expert 和 Inter-expert 预取机制
- 带宽利用率分析和瓶颈识别

### 核心特性

| 特性 | 说明 |
|------|------|
| **双布局模式** | CH-first（基线） vs PL-first（预取优化） |
| **精确时序** | tR-TX 重叠掩盖（Hid）计算 |
| **预取仿真** | 支持 intra/inter-expert 预取策略 |
| **带宽分析** | 详细的带宽损失分解和优化建议 |
| **可视化** | 布局可视化支持 |

---

## 快速开始

### 方式一：图形界面 (GUI) ⭐推荐

**Windows 用户：** 直接双击 `NAND_MoE_Simulator.exe` 即可启动图形界面

**Python 用户：**
```bash
python nand.py          # 无参数自动启动 GUI
python nand.py --gui    # 显式启动 GUI
```

GUI 功能特点：
- 可视化参数输入，无需记忆命令行
- 预设配置快速加载（低配/中配/高配）
- 实时显示仿真结果
- 一键导出 CSV 和布局图

### 方式二：命令行模式

**Python 源码：**
```bash
# 安装依赖
pip install matplotlib numpy

# 运行仿真
python nand.py -c 8 -p 8 --bw 30e9 --tr 22e-6 -e 0,1,2
```

**Windows 可执行文件：**
```cmd
NAND_MoE_Simulator.exe -c 8 -p 8 --bw 30e9 --tr 22e-6 -e 0,1,2
```

---

## 命令行参数

### 硬件几何参数 (NAND Geometry)

| 参数 | 短选项 | 默认值 | 说明 |
|------|--------|--------|------|
| `--channels` | `-c` | 8 | NAND 通道数 (C) |
| `--planes` | `-p` | 8 | 每通道 Plane 数 (P) |
| `--page-size` | `-s` | 16384 | 页大小（字节） |

### 性能参数 (Performance)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bw` | 30e9 | 总带宽（字节/秒），如 30e9 = 30GB/s |
| `--tr` | 22e-6 | 读延迟 tR（秒），如 22e-6 = 22us |

### 专家参数 (Expert)

| 参数 | 短选项 | 默认值 | 说明 |
|------|--------|--------|------|
| `--experts` | `-e` | 0,1,2 | 要模拟的专家 ID，支持逗号或范围格式 |
| `--num-experts` | | 10 | 总专家数量（用于布局） |
| `--gate-bytes` | | 294912 | gate 部分字节数 |
| `--up-bytes` | | 294912 | up 部分字节数 |
| `--down-bytes` | | 294912 | down 部分字节数 |

### 布局选项 (Layout)

| 参数 | 选项 | 默认 | 说明 |
|------|------|------|------|
| `--layout` | ch-first / pl-first | pl-first | 数据布局方式 |

**布局对比：**
- `ch-first`: gate/up/down 各占独立 Plane，预取不触发
- `pl-first`: gate/up/down 在同一 Plane 的不同 CH，预取全触发 ✅

### 缓存选项 (Cache/Prefetch)

| 参数 | 默认 | 说明 |
|------|------|------|
| `--intra` | ON | 启用 intra-expert 预取（同专家内 part 间） |
| `--no-intra` | | 禁用 intra-expert 预取 |
| `--inter` | ON | 启用 inter-expert 预取（跨专家） |
| `--no-inter` | | 禁用 inter-expert 预取 |

### 输出选项 (Output)

| 参数 | 说明 |
|------|------|
| `--csv` | 输出 CSV 文件路径 |
| `--viz` | 可视化图片保存路径（如 layout.png） |
| `--max-pages` | 可视化最大页数（默认 20） |
| `--no-viz` | 不显示可视化窗口 |
| `--quiet` / `-q` | 静默模式，只输出结果 |

### GUI 选项

| 参数 | 说明 |
|------|------|
| `--gui` | 启动图形界面（忽略其他参数） |

**启动 GUI 的方式：**
1. 直接双击 `NAND_MoE_Simulator.exe`
2. 运行 `python nand.py`（无参数自动启动 GUI）
3. 运行 `python nand.py --gui`

---

## 使用示例

### 基础用法

```bash
# 默认配置，模拟专家 0,1,2
python nand.py

# 指定通道数和平面数
python nand.py -c 4 -p 4

# 自定义带宽和延迟
python nand.py --bw 10e9 --tr 50e-6

# 模拟多个专家
python nand.py -e 0,1,2,3,4

# 使用范围格式
python nand.py -e 0-9

# 混合格式
python nand.py -e 0-3,5,7-9
```

### 布局对比

```bash
# CH-first 布局（预取不触发）
python nand.py --layout ch-first -e 0,1 --no-viz

# PL-first 布局（预取优化）
python nand.py --layout pl-first -e 0,1 --no-viz
```

### 预取开关对比

```bash
# 无预取
python nand.py -e 0,1 --no-intra --no-inter --no-viz

# 只有 intra 预取
python nand.py -e 0,1 --intra --no-inter --no-viz

# 完整预取
python nand.py -e 0,1 --intra --inter --no-viz
```

### 导出结果

```bash
# 导出 CSV
python nand.py -e 0-5 --csv result.csv --no-viz

# 保存布局图
python nand.py -e 0,1,2 --viz layout.png --no-viz

# 静默模式（适合脚本）
python nand.py -e 0-9 --quiet
```

---

## 输出说明

### 标准输出示例

```
============================================================
  Sequential Read : experts=[0, 1, 2]
  tR=22.0us  tX=4.3691us  intra=ON  inter=ON
============================================================

  [Step Detail]
   EID   PART  crit_ch  tR(us)  hid(us)   tX(us)  c_pl saved(us)  time(us)  note
  ------------------------------------------------------------------------------
     0   gate        0   44.00    17.48    26.21     0      0.00     52.74
     0     up        0   22.00    17.48    26.21    12     52.43     30.74  [prefetch(intra):12pl]
     ...

  [Bandwidth Analysis]
  ==================================================
        Theoretical BW :   30.000 GB/s (额定带宽)
          Effective BW :   20.923 GB/s (实际带宽)
           Utilization :    69.74 %
  --------------------------------------------------

  [Bandwidth Loss Breakdown]
  ==================================================
  1. tR Latency Overhead
          - Blocked tR time :   110.00 us (40.7%)
         - Hid (overlapped) :   110.00 us (saved)

  2. Prefetch Benefits
                 - TX saved :   446.43 us (82.5%)
                 - tR saved :   110.00 us (20.3%)

  4. Channel Parallelism
            - Avg planes/CH :      6.0 / 8
            - Parallel util :     75.0% (per-page)

  [Recommendations]
  ==================================================
  [OK] 带宽利用率良好 (69.7%)
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `tR(us)` | 该 step 产生的读延迟 |
| `hid(us)` | tR 被上一页 TX 掩盖的时间 |
| `tX(us)` | 关键路径传输时间 |
| `c_pl` | 命中缓存的 Plane 数 |
| `saved(us)` | 预取节省的总时间 |
| `prefetch(intra)` | 同专家预取命中 |
| `prefetch(inter)` | 跨专家预取命中 |

---

## 带宽损失分析

仿真器提供详细的带宽损失分解：

### 1. tR 延迟开销

NAND 读取需要等待 tR 时间，这是最主要的带宽损失来源。
- **Blocked tR**: 未被掩盖的 tR 时间
- **Hid (overlapped)**: 被 TX 掩盖的 tR 时间（收益）

### 2. 预取收益

- **TX saved**: 预取避免的数据传输时间
- **tR saved**: 预取避免的 page 读取时间

### 3. 通道并行度

- **Avg planes/CH**: 平均每通道使用的 Plane 数
- **Parallel util**: 通道并行利用率

### 优化建议

程序根据带宽利用率自动给出建议：
- `[!]` 带宽利用率低：建议增大 page_size 或启用预取
- `[OK]` 带宽利用率良好
- `[TIP]` 建议启用特定预取选项

---

## 项目结构

```
nand/
├── nand.py                      # 主程序源码
├── rules.md                     # 仿真规则手册
├── README.md                    # 本文档
├── NAND_MoE_Simulator.spec      # PyInstaller 配置
├── build/                       # 构建中间文件
└── dist/
    └── NAND_MoE_Simulator.exe   # Windows 可执行文件
```

---

## 硬件参数参考

| 参数 | 低配 NAND | 高配 NAND | 本例默认值 |
|------|-----------|-----------|-----------|
| Channels | 4-8 | 8-16 | 8 |
| Planes/CH | 2-4 | 4-8 | 8 |
| Page Size | 16KB | 16KB | 16KB |
| Bandwidth | 3-7 GB/s | 10-14 GB/s | 30 GB/s |
| tR | 50-100 us | 20-50 us | 22 us |

---

## 数学模型

### 总时间计算

```
T_total = N_rows × tR + ΣtX_i - Hid_total

where:
  N_rows = 访问的不同 page 数量
  tR = 每页读延迟
  tX_i = 第 i 页的传输时间
  Hid_total = Σmin(tR, tX_{i-1})  # tR-TX 重叠掩盖
```

### 有效带宽

```
Eff_BW = Total_Bytes / T_total
```

### 带宽利用率

```
Utilization = Eff_BW / Theoretical_BW × 100%
```

---

## 许可证

MIT License

---

## 作者

threethones

---

## 更新日志

### v1.1 (2026-03-23)
- 添加命令行参数支持
- 添加带宽损失分析
- 添加性能优化建议
- 支持静默模式

### v1.0 (2026-03-23)
- 初始版本
- 基础 NAND 仿真
- 双布局模式支持
- 预取机制仿真
