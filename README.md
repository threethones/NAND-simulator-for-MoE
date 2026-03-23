# NAND Flash MoE Simulator

NAND Flash MoE 仿真器 - 模拟 Mixture-of-Experts (MoE) 模型中专家参数存储在 NAND Flash 上的读取行为。

---

## 项目简介

本项目是一个用于研究和优化 MoE 模型在 NAND Flash 存储系统上推理性能的仿真工具。

### 核心特性

| 特性 | 说明 |
|------|------|
| **双布局模式** | CH-first（基线） vs PL-first（预取优化） |
| **精确时序** | tR-TX 重叠掩盖（Hid）计算 |
| **预取仿真** | 支持 intra/inter-expert 预取策略 |
| **TopK 分析** | Monte Carlo 模拟三种访问模式 × 四种预取组合 |
| **可视化** | 布局可视化与性能对比图表 |

---

## 快速开始

### 方式一：图形界面 (GUI) ⭐推荐

**Windows 用户：** 直接双击 `NAND_MoE_Simulator.exe` 即可启动图形界面

**Python 用户：**
```bash
python nand.py    # 启动 GUI
```

GUI 功能特点：
- 可视化参数输入
- 实时显示仿真结果
- TopK 分析（顺序/局部随机/全随机访问模式对比）
- 一键导出图表

---

## 项目结构

```
nand/
├── nand.py                 # 核心仿真模块
├── layout_analyse.py       # TopK 分析模块
├── gui.py                  # 图形界面
├── rules.md                # 仿真规则手册
├── README.md               # 本文档
└── dist/
    └── NAND_MoE_Simulator.exe   # Windows 可执行文件
```

---

## 硬件参数参考

| 参数 | 低配 NAND | 高配 NAND |
|------|-----------|-----------|
| Channels | 4-8 | 8-16 |
| Planes/CH | 2-4 | 4-8 |
| Page Size | 16KB | 16KB |
| Bandwidth/CH | 1-2 GB/s | 3-4 GB/s |
| tR | 50-100 us | 20-50 us |

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

---

## 许可证

MIT License

---

## 作者

threethones

---

## 更新日志

### v2.0 (2026-03-23)
- 添加 TopK 分析功能（三种访问模式 × 四种预取组合）
- 代码结构优化，移除冗余功能
- 统一 GUI 为唯一入口

### v1.0 (2026-03-23)
- 初始版本
- 基础 NAND 仿真
- 双布局模式支持
- 预取机制仿真
