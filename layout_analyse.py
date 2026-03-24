# ================================================================== #
#  layout_analyse.py  —— MoE NAND TopK 分析框架                       #
#                                                                      #
#  核心功能：                                                           #
#  1. 三种访问模式 × 四种预取组合的 Monte Carlo 模拟                   #
#  2. 预取对比表格生成与可视化                                          #
# ================================================================== #

from __future__ import annotations

import random
import numpy as np
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib

from nand import (
    NandSimulator,
    estimate_sequential_latency,
)

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False


# ================================================================== #
#  常量                                                                #
# ================================================================== #

PART_ORDER = ["gate", "up", "down"]

# 四种预取组合
PREFETCH_MODES = {
    "none":       dict(intra_expert_cache=False, inter_expert_cache=False),
    "intra_only": dict(intra_expert_cache=True,  inter_expert_cache=False),
    "inter_only": dict(intra_expert_cache=False, inter_expert_cache=True),
    "full":       dict(intra_expert_cache=True,  inter_expert_cache=True),
}

PREFETCH_LABELS = {
    "none":       "\u65e0\u9884\u53d6",
    "intra_only": "\u4e13\u5bb6\u5185\u9884\u53d6",
    "inter_only": "\u4e13\u5bb6\u95f4\u9884\u53d6",
    "full":       "\u5168\u9884\u53d6",
}

_MODE_COLORS = {
    "sequential": "#2196F3",
    "local":      "#FF9800",
    "random":     "#9C27B0",
}
_MODE_LABELS = {
    "sequential": "\u987a\u5e8f\u8bfb\u53d6",
    "local":      "\u5c40\u90e8\u968f\u673a\u8bfb\u53d6",
    "random":     "\u5168\u968f\u673a\u8bfb\u53d6",
}


# ================================================================== #
#  核心函数：simulate_topk_batch                                       #
# ================================================================== #

def simulate_topk_batch(
    sim: NandSimulator,
    expert_ids: List[int],
    bw_total_Bps: float,
    tR_sec: float,
    intra_expert_cache: bool = True,
    inter_expert_cache: bool = True,
) -> dict:
    """对一个 top-k batch 做完整延迟分析。"""

    res = estimate_sequential_latency(
        sim,
        expert_ids,
        bw_total_Bps=bw_total_Bps,
        tR_sec=tR_sec,
        intra_expert_cache=intra_expert_cache,
        inter_expert_cache=inter_expert_cache,
    )

    step_stats  = res["step_stats"]
    total_bytes = res["total_bytes"]
    total_time  = res["total_time_sec"]
    eff_bw_GBps = (total_bytes / total_time / 1e9) if total_time > 0 else 0.0

    intra_saved_tr_sec = sum(
        st.get("saved_tr_sec", 0.)
        for st in step_stats.values()
        if st.get("prefetch_src") == "intra"
    )
    intra_saved_tx_sec = sum(
        st.get("saved_tx_sec", 0.)
        for st in step_stats.values()
        if st.get("prefetch_src") == "intra"
    )
    inter_saved_tr_sec = sum(
        st.get("saved_tr_sec", 0.)
        for st in step_stats.values()
        if st.get("prefetch_src") == "inter"
    )
    inter_saved_tx_sec = sum(
        st.get("saved_tx_sec", 0.)
        for st in step_stats.values()
        if st.get("prefetch_src") == "inter"
    )

    intra_saved_sec = intra_saved_tr_sec + intra_saved_tx_sec
    inter_saved_sec = inter_saved_tr_sec + inter_saved_tx_sec
    full_saved_sec  = intra_saved_sec + inter_saved_sec

    sum_tr_sec  = sum(step_stats[(eid, p)]["tr_sec"]          for eid in expert_ids for p in PART_ORDER)
    sum_tx_sec  = sum(step_stats[(eid, p)]["tx_sec"]          for eid in expert_ids for p in PART_ORDER)
    sum_hid_sec = sum(step_stats[(eid, p)].get("hid_sec", 0.) for eid in expert_ids for p in PART_ORDER)

    tr_exposed_sec  = max(0., sum_tr_sec - sum_hid_sec)
    hid_ratio       = (sum_hid_sec / sum_tr_sec) if sum_tr_sec > 0 else 0.0
    tr_overhead_pct = (tr_exposed_sec / total_time * 100) if total_time > 0 else 0.0

    per_exp_breakdown = _build_per_exp_breakdown(step_stats, expert_ids)

    return {
        "total_time_sec":     total_time,
        "total_bytes":        total_bytes,
        "eff_bw_GBps":        eff_bw_GBps,
        "tr_total_us":        sum_tr_sec  * 1e6,
        "tx_total_us":        sum_tx_sec  * 1e6,
        "hid_total_us":       sum_hid_sec * 1e6,
        "tr_exposed_us":      tr_exposed_sec  * 1e6,
        "hid_ratio":          hid_ratio,
        "tr_overhead_pct":    tr_overhead_pct,
        "intra_saved_us":     intra_saved_sec    * 1e6,
        "intra_saved_tr_us":  intra_saved_tr_sec * 1e6,
        "intra_saved_tx_us":  intra_saved_tx_sec * 1e6,
        "inter_saved_us":     inter_saved_sec    * 1e6,
        "inter_saved_tr_us":  inter_saved_tr_sec * 1e6,
        "inter_saved_tx_us":  inter_saved_tx_sec * 1e6,
        "full_saved_us":      full_saved_sec     * 1e6,
        "per_exp_breakdown":  per_exp_breakdown,
        "_v6_full":           res,
    }


def _build_per_exp_breakdown(
    step_stats: dict,
    expert_ids: List[int],
) -> List[dict]:
    result = []
    for eid in expert_ids:
        tr_us  = sum(step_stats[(eid, p)]["tr_sec"]           for p in PART_ORDER) * 1e6
        tx_us  = sum(step_stats[(eid, p)]["tx_sec"]           for p in PART_ORDER) * 1e6
        hid_us = sum(step_stats[(eid, p)].get("hid_sec", 0.)  for p in PART_ORDER) * 1e6
        tr_saved_us = sum(step_stats[(eid, p)].get("saved_tr_sec", 0.) for p in PART_ORDER) * 1e6
        tx_saved_us = sum(step_stats[(eid, p)].get("saved_tx_sec", 0.) for p in PART_ORDER) * 1e6
        result.append({
            "eid":          eid,
            "tr_us":        tr_us,
            "tx_us":        tx_us,
            "hid_us":       hid_us,
            "tr_saved_us":  tr_saved_us,
            "tx_saved_us":  tx_saved_us,
            "time_us":      tr_us + tx_us - hid_us,
        })
    return result


# ================================================================== #
#  Monte Carlo 采样：三种访问模式 × 四种预取组合                        #
# ================================================================== #

def run_monte_carlo_prefetch_compare(
    sim: NandSimulator,
    num_experts: int,
    bw_total_Bps: float,
    tR_sec: float,
    topk: int = 10,
    n_trials: int = 200,
    local_window_factor: int = 2,
    seed: int = 42,
) -> Dict[str, Dict[str, List[dict]]]:
    """
    三种访问模式 × 四种预取组合的 Monte Carlo 对比。

    返回结构：
      results[access_mode][prefetch_mode] = List[dict]

    access_mode : "sequential" | "local" | "random"
    prefetch_mode: "none" | "intra_only" | "inter_only" | "full"

    同一 trial 的 expert_ids 在四种预取模式下完全相同，
    确保对比的公平性（控制变量）。
    """
    rng = random.Random(seed)

    def gen_sequential():
        start = rng.randint(0, num_experts - topk)
        return list(range(start, start + topk))

    def gen_local():
        window = topk * local_window_factor
        start  = rng.randint(0, num_experts - window)
        pool   = list(range(start, start + window))
        return sorted(rng.sample(pool, topk))

    def gen_random():
        return sorted(rng.sample(range(num_experts), topk))

    generators = {
        "sequential": gen_sequential,
        "local":      gen_local,
        "random":     gen_random,
    }

    # results[access_mode][prefetch_mode] = List[dict]
    results: Dict[str, Dict[str, List[dict]]] = {
        amode: {pmode: [] for pmode in PREFETCH_MODES}
        for amode in generators
    }

    for amode, gen in generators.items():
        for _ in range(n_trials):
            eids = gen()   # 同一组 expert_ids，跑四种预取
            for pmode, pkwargs in PREFETCH_MODES.items():
                r = simulate_topk_batch(
                    sim, eids, bw_total_Bps, tR_sec, **pkwargs
                )
                r["expert_ids"] = eids
                results[amode][pmode].append(r)

    return results


# ================================================================== #
#  统计汇总                                                            #
# ================================================================== #

def summarize(results: List[dict]) -> dict:
    def _stats(vals):
        return {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "p50":  float(np.percentile(vals, 50)),
            "p95":  float(np.percentile(vals, 95)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
        }

    return {
        "total_time_us":     _stats([r["total_time_sec"] * 1e6 for r in results]),
        "eff_bw_GBps":       _stats([r["eff_bw_GBps"]          for r in results]),
        "tr_total_us":       _stats([r["tr_total_us"]          for r in results]),
        "tx_total_us":       _stats([r["tx_total_us"]          for r in results]),
        "hid_total_us":      _stats([r["hid_total_us"]         for r in results]),
        "tr_exposed_us":     _stats([r["tr_exposed_us"]        for r in results]),
        "hid_ratio":         _stats([r["hid_ratio"]            for r in results]),
        "tr_overhead_pct":   _stats([r["tr_overhead_pct"]      for r in results]),
        "intra_saved_us":    _stats([r["intra_saved_us"]       for r in results]),
        "intra_saved_tr_us": _stats([r["intra_saved_tr_us"]    for r in results]),
        "intra_saved_tx_us": _stats([r["intra_saved_tx_us"]    for r in results]),
        "inter_saved_us":    _stats([r["inter_saved_us"]       for r in results]),
        "inter_saved_tr_us": _stats([r["inter_saved_tr_us"]    for r in results]),
        "inter_saved_tx_us": _stats([r["inter_saved_tx_us"]    for r in results]),
        "full_saved_us":     _stats([r["full_saved_us"]        for r in results]),
    }


# ================================================================== #
#  预取对比打印（字符串返回，便于GUI显示）                               #
# ================================================================== #

def format_prefetch_comparison(
    compare_results: Dict[str, Dict[str, List[dict]]],
    bw_total_Bps: float,
    channels: int = 8,
) -> str:
    """
    返回三种访问模式 × 四种预取组合的汇总对比表字符串。
    用于GUI显示，不直接打印。
    """
    peak_bw_per_ch = bw_total_Bps / 1e9  # 单通道峰值
    peak_bw_total = peak_bw_per_ch * channels  # 总峰值带宽
    lines = []

    lines.append("=" * 95)
    lines.append("  预取模式对比（三种访问模式 × 四种预取组合）")
    lines.append(f"  峰值带宽 = {peak_bw_total:.1f} GB/s ({peak_bw_per_ch:.2f} GB/s × {channels} CH)")
    lines.append("=" * 95)

    # 固定列宽定义（基于等宽字体字符数）
    # 中文字符占2宽度，英文/数字占1宽度
    COL_WIDTHS = [12, 10, 8, 8, 8, 8, 10, 10, 8]  # 各列宽度
    
    def make_separator():
        """生成分隔线，确保 | 位置对齐"""
        parts = []
        for w in COL_WIDTHS:
            parts.append("-" * w)
        return "  " + "|".join(parts)
    
    def make_row(items):
        """生成一行，确保 | 位置对齐"""
        # items 长度应与 COL_WIDTHS 一致
        cells = []
        for i, (item, width) in enumerate(zip(items, COL_WIDTHS)):
            text = str(item)
            # 计算实际显示宽度（中文算2）
            display_width = sum(2 if ord(c) > 127 else 1 for c in text)
            padding = width - display_width
            if padding < 0:
                padding = 0
            cells.append(text + " " * padding)
        return "  " + "|".join(cells)

    for amode in ["sequential", "local", "random"]:
        label = _MODE_LABELS[amode]
        lines.append(f"\n【{label}】")
        
        # 表头
        lines.append(make_row(["预取模式", "时间(us)", "带宽", "利用率", "tR(us)", "TX(us)", "掩盖(us)", "暴露(us)", "加速比"]))
        lines.append(make_separator())

        # 取 none 的 mean 作为加速比基准
        s_none = summarize(compare_results[amode]["none"])
        base_time = s_none["total_time_us"]["mean"]

        for pmode in ["none", "intra_only", "inter_only", "full"]:
            s = summarize(compare_results[amode][pmode])
            plabel = PREFETCH_LABELS[pmode]

            total_mean = s["total_time_us"]["mean"]
            eff_bw     = s["eff_bw_GBps"]["mean"]
            util_pct   = eff_bw / peak_bw_total * 100
            tr_us      = s["tr_total_us"]["mean"]
            tx_us      = s["tx_total_us"]["mean"]
            hid_us     = s["hid_total_us"]["mean"]
            exposed_us = s["tr_exposed_us"]["mean"]
            speedup    = base_time / total_mean

            lines.append(make_row([
                plabel,
                f"{total_mean:>8.1f}",
                f"{eff_bw:>6.2f}",
                f"{util_pct:>5.1f}%",
                f"{tr_us:>6.1f}",
                f"{tx_us:>6.1f}",
                f"{hid_us:>8.1f}",
                f"{exposed_us:>8.1f}",
                f"{speedup:>5.2f}x"
            ]))

    return "\n".join(lines)


# ================================================================== #
#  带宽分析                                                             #
# ================================================================== #

def format_bw_analysis(
    mc_results: Dict[str, List[dict]],
    bw_total_Bps: float,
    tR_sec: float,
    page_size_bytes: int,
    channels: int,
) -> str:
    """返回各访问模式的带宽分析，不含标题和物理参数。"""
    peak_bw_total = bw_total_Bps * channels / 1e9
    lines = []

    for mode, results in mc_results.items():
        s     = summarize(results)
        label = _MODE_LABELS[mode]

        total_us    = s["total_time_us"]["mean"]
        tx_us       = s["tx_total_us"]["mean"]
        tr_us       = s["tr_total_us"]["mean"]
        hid_us      = s["hid_total_us"]["mean"]
        exposed_us  = s["tr_exposed_us"]["mean"]
        hid_ratio   = s["hid_ratio"]["mean"]
        eff_bw      = s["eff_bw_GBps"]["mean"]
        util_pct    = eff_bw / peak_bw_total * 100

        lines.append(f"\n【{label}】")
        lines.append(f"  有效带宽 : {eff_bw:6.3f} GB/s")
        lines.append(f"  带宽利用率    : {min(util_pct, 100.0):5.1f}%")
        lines.append(f"  时间分解：")
        lines.append(f"    总延迟    : {total_us:8.1f} us")
        lines.append(f"    TX 时间   : {tx_us:8.1f} us")
        lines.append(f"    tR 总量   : {tr_us:8.1f} us")
        lines.append(f"      ├─ tR掩盖 : {hid_us:8.1f} us (掩盖率 {hid_ratio*100:.1f}%)")
        lines.append(f"      └─ tR暴露 : {exposed_us:8.1f} us")
    
    return "\n".join(lines)


# ================================================================== #
#  可视化                                                               #
# ================================================================== #

def plot_prefetch_comparison(
    compare_results: Dict[str, Dict[str, List[dict]]],
    bw_total_Bps: float,
    channels: int = 8,
    save_path: str = None,
) -> None:
    """
    绘制三种访问模式 × 四种预取组合的对比图，共 4 张子图：

      图1（左上）：总延迟 mean              —— 分组柱状图
      图2（右上）：有效带宽 mean             —— 分组柱状图（附峰值带宽参考线）
      图3（左下）：时间分解堆叠柱状图        —— tR暴露 / tR掩盖 / TX
      图4（右下）：预取节省量               —— 专家内tR / 专家内TX / 专家间tR / 专家间TX 堆叠
    """
    peak_bw = bw_total_Bps * channels / 1e9  # 总峰值带宽

    access_modes   = ["sequential", "local", "random"]
    prefetch_modes = ["none", "intra_only", "inter_only", "full"]

    # ── 提取统计数据 ─────────────────────────────────────────────────
    data = {
        amode: {
            pmode: summarize(compare_results[amode][pmode])
            for pmode in prefetch_modes
        }
        for amode in access_modes
    }

    # ── 颜色方案 ─────────────────────────────────────────────────────
    p_colors = {
        "none":       "#B0BEC5",
        "intra_only": "#42A5F5",
        "inter_only": "#FFA726",
        "full":       "#66BB6A",
    }
    p_labels = {
        "none":       "无预取",
        "intra_only": "专家内预取",
        "inter_only": "专家间预取",
        "full":       "完全预取",
    }
    c_exposed  = "#EF5350"
    c_hid      = "#FFA726"
    c_tx       = "#42A5F5"
    c_intra_tr = "#1565C0"
    c_intra_tx = "#90CAF9"
    c_inter_tr = "#E65100"
    c_inter_tx = "#FFCC80"

    a_labels = [_MODE_LABELS[m] for m in access_modes]

    # ── 布局 ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("预取模式对比：三种访问模式 × 四种预取组合",
                 fontsize=14, fontweight="bold", y=0.98)

    n_access   = len(access_modes)
    x          = np.arange(n_access)
    bar_w      = 0.18
    offsets    = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w

    # ================================================================
    # 图1：总延迟 mean（无误差棒）
    # ================================================================
    ax1 = axes[0, 0]
    for i, pmode in enumerate(prefetch_modes):
        means = [data[am][pmode]["total_time_us"]["mean"] for am in access_modes]
        bars  = ax1.bar(
            x + offsets[i], means, bar_w,
            label=p_labels[pmode],
            color=p_colors[pmode],
            edgecolor="white", linewidth=0.5,
        )
        # 柱顶标注数值
        for bar, m in zip(bars, means):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, m + 2,
                f"{m:.0f}", ha="center", va="bottom",
                fontsize=6.5, rotation=90,
            )

    ax1.set_title("总延迟 (mean)", fontsize=11)
    ax1.set_ylabel("延迟 (us)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(a_labels)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(bottom=0)

    # ================================================================
    # 图2：有效带宽 mean
    # ================================================================
    ax2 = axes[0, 1]
    for i, pmode in enumerate(prefetch_modes):
        means = [data[am][pmode]["eff_bw_GBps"]["mean"] for am in access_modes]
        bars  = ax2.bar(
            x + offsets[i], means, bar_w,
            label=p_labels[pmode],
            color=p_colors[pmode],
            edgecolor="white", linewidth=0.5,
        )
        for bar, m in zip(bars, means):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, m + 0.2,
                f"{m:.1f}", ha="center", va="bottom",
                fontsize=6.5, rotation=90,
            )

    ax2.axhline(peak_bw, color="red", linestyle="--", linewidth=1.2,
                label=f"峰值 {peak_bw:.0f} GB/s")
    ax2.set_title("有效带宽 (mean)", fontsize=11)
    ax2.set_ylabel("带宽 (GB/s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(a_labels)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, peak_bw * 1.15)

    # ================================================================
    # 图3：时间分解堆叠柱状图（空转 / hid / TX）
    # ================================================================
    ax3 = axes[1, 0]

    group_gap = 0.35
    bar_w3    = 0.17
    xs3       = []
    tick_pos, tick_labels = [], []
    cur_x = 0.0
    for ai, amode in enumerate(access_modes):
        group_xs = []
        for pi in range(len(prefetch_modes)):
            group_xs.append(cur_x)
            cur_x += bar_w3
        xs3.append(group_xs)
        tick_pos.append(np.mean(group_xs))
        tick_labels.append(a_labels[ai])
        cur_x += group_gap

    for ai, amode in enumerate(access_modes):
        for pi, pmode in enumerate(prefetch_modes):
            s       = data[amode][pmode]
            exposed = s["tr_exposed_us"]["mean"]
            hid_val = s["hid_total_us"]["mean"]
            tx_val  = s["tx_total_us"]["mean"]
            xi      = xs3[ai][pi]

            ax3.bar(xi, exposed, bar_w3, color=c_exposed,
                    label="tR暴露" if (ai == 0 and pi == 0) else "")
            ax3.bar(xi, hid_val, bar_w3, bottom=exposed, color=c_hid,
                    label="tR掩盖" if (ai == 0 and pi == 0) else "")
            ax3.bar(xi, tx_val,  bar_w3, bottom=exposed + hid_val, color=c_tx,
                    label="TX" if (ai == 0 and pi == 0) else "")

    short_labels = ["N", "内预", "间预", "完全"]
    for ai in range(n_access):
        for pi, sl in enumerate(short_labels):
            ax3.text(xs3[ai][pi], -18, sl, ha="center", va="top",
                     fontsize=6.5, rotation=45, color="#555555")

    ax3.set_title("时间分解：tR暴露 / tR掩盖 / TX", fontsize=11)
    ax3.set_ylabel("时间 (us)")
    ax3.set_xticks(tick_pos)
    ax3.set_xticklabels(tick_labels)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_ylim(bottom=-30)

    # ================================================================
    # 图4：预取节省量堆叠（intra_only / inter_only / full）
    # ================================================================
    ax4 = axes[1, 1]
    show_pmodes = ["intra_only", "inter_only", "full"]
    bar_w4      = 0.22
    offsets4    = np.array([-1, 0, 1]) * bar_w4

    for pi, pmode in enumerate(show_pmodes):
        intra_tr = np.array([data[am][pmode]["intra_saved_tr_us"]["mean"] for am in access_modes])
        intra_tx = np.array([data[am][pmode]["intra_saved_tx_us"]["mean"] for am in access_modes])
        inter_tr = np.array([data[am][pmode]["inter_saved_tr_us"]["mean"] for am in access_modes])
        inter_tx = np.array([data[am][pmode]["inter_saved_tx_us"]["mean"] for am in access_modes])
        xi = x + offsets4[pi]

        ax4.bar(xi, intra_tr, bar_w4, color=c_intra_tr,
                label="专家内 tR" if pi == 0 else "")
        ax4.bar(xi, intra_tx, bar_w4, bottom=intra_tr, color=c_intra_tx,
                label="专家内 TX" if pi == 0 else "")
        ax4.bar(xi, inter_tr, bar_w4, bottom=intra_tr + intra_tx, color=c_inter_tr,
                label="专家间 tR" if pi == 0 else "")
        ax4.bar(xi, inter_tx, bar_w4, bottom=intra_tr + intra_tx + inter_tr, color=c_inter_tx,
                label="专家间 TX" if pi == 0 else "")

        totals = intra_tr + intra_tx + inter_tr + inter_tx
        for xii, tot in zip(xi, totals):
            ax4.text(xii, tot + 3, f"{tot:.0f}", ha="center", va="bottom",
                     fontsize=7, fontweight="bold")

    for pi, pmode in enumerate(show_pmodes):
        for ai in range(n_access):
            ax4.text(
                x[ai] + offsets4[pi], -18,
                p_labels[pmode].replace(" ", "\n"),
                ha="center", va="top", fontsize=6, color="#555555",
            )

    ax4.set_title("预取节省量分解（专家内/专家间 × tR/TX）", fontsize=11)
    ax4.set_ylabel("节省时间 (us)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(a_labels)
    ax4.legend(
        fontsize=8, loc="upper right",
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=c_intra_tr, label="专家内 tR"),
            plt.Rectangle((0, 0), 1, 1, color=c_intra_tx, label="专家内 TX"),
            plt.Rectangle((0, 0), 1, 1, color=c_inter_tr, label="专家间 tR"),
            plt.Rectangle((0, 0), 1, 1, color=c_inter_tx, label="专家间 TX"),
        ],
    )
    ax4.grid(axis="y", alpha=0.3)
    ax4.set_ylim(bottom=-30)

    # ── 底部说明 ─────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        "图3/图4 柱组内顺序：N=无预取  内预=专家内预取  间预=专家间预取  完全=完全预取(内+间)",
        ha="center", fontsize=8, color="#666666",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[图表已保存] {save_path}")
    plt.show(block=False)


# ================================================================== #
#  便捷入口函数（供 GUI 调用）                                           #
# ================================================================== #

def run_topk_analysis(
    sim: NandSimulator,
    bw_total_Bps: float,
    tR_sec: float,
    num_experts: int = 512,
    topk: int = 10,
    n_trials: int = 200,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    运行完整的 TopK 分析，返回结果字符串和图表。
    
    Returns:
        (comparison_text, bw_analysis_text): 两个分析结果的字符串
    """
    # 运行 3×4 预取对比
    compare = run_monte_carlo_prefetch_compare(
        sim,
        num_experts=num_experts,
        bw_total_Bps=bw_total_Bps,
        tR_sec=tR_sec,
        topk=topk,
        n_trials=n_trials,
        seed=seed,
    )
    
    # 生成对比文本
    comparison_text = format_prefetch_comparison(compare, bw_total_Bps, channels=sim.geo.channels)
    
    # 生成带宽分析（使用 full 预取模式的结果）
    mc_results = {
        mode: compare[mode]["full"] for mode in ["sequential", "local", "random"]
    }
    bw_analysis_text = format_bw_analysis(
        mc_results,
        bw_total_Bps=bw_total_Bps,
        tR_sec=tR_sec,
        page_size_bytes=sim.geo.page_size_bytes,
        channels=sim.geo.channels,
    )
    
    return comparison_text, bw_analysis_text, compare


# 此模块通过 GUI 调用，无命令行入口
