from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ================================================================== #
#  数据结构                                                            #
# ================================================================== #

@dataclass
class NandGeometry:
    channels: int
    planes_per_channel: int
    page_size_bytes: int


@dataclass
class PageSlice:
    ch: int
    pl: int
    page: int
    offset: int
    length: int
    part: str
    expert_id: int


@dataclass
class PageWrite:
    expert_id: int
    write_index: int
    ch: int
    pl: int
    size_bytes: int
    slices: List[PageSlice] = field(default_factory=list)


@dataclass
class ExpertPlacement:
    expert_id: int
    gate_bytes: int
    up_bytes: int
    down_bytes: int
    page_size_bytes: int
    writes: List[PageWrite] = field(default_factory=list)

    def all_slices(self) -> List[PageSlice]:
        return [s for w in self.writes for s in w.slices]


# ================================================================== #
#  布局函数：Round-Robin 按页分配                                      #
# ================================================================== #

def place_experts_page_rr(
    geo: NandGeometry,
    num_experts: int,
    gate_bytes: int,
    up_bytes: int,
    down_bytes: int,
) -> "NandSimulator":
    C, P, S = geo.channels, geo.planes_per_channel, geo.page_size_bytes
    L = C * P

    def map_global_to_phy(g: int):
        in_off   = g % S
        k        = g // S
        lane     = k % L
        t        = k // L
        lane_off = t * S + in_off
        ch       = lane % C
        pl       = lane // C
        page     = lane_off // S
        return ch, pl, page, lane_off % S

    placements: List[ExpertPlacement] = []
    global_ptr = 0

    for eid in range(num_experts):
        ep = ExpertPlacement(
            expert_id=eid,
            gate_bytes=gate_bytes, up_bytes=up_bytes, down_bytes=down_bytes,
            page_size_bytes=S,
        )
        write_map: Dict[Tuple[int, int, int], PageWrite] = {}

        for part_name, part_bytes in [("gate", gate_bytes), ("up", up_bytes), ("down", down_bytes)]:
            remaining = part_bytes
            while remaining > 0:
                g = global_ptr
                ch, pl, page, page_off = map_global_to_phy(g)
                chunk = min(S - page_off, remaining)
                key = (ch, pl, page)
                if key not in write_map:
                    pw = PageWrite(expert_id=eid, write_index=len(write_map),
                                   ch=ch, pl=pl, size_bytes=S)
                    write_map[key] = pw
                    ep.writes.append(pw)
                write_map[key].slices.append(PageSlice(
                    ch=ch, pl=pl, page=page,
                    offset=page_off, length=chunk,
                    part=part_name, expert_id=eid,
                ))
                global_ptr += chunk
                remaining  -= chunk

        placements.append(ep)

    return NandSimulator(geo=geo, placements=placements)


# ================================================================== #
#  布局函数：PL-first Round-Robin（预取优先）                          #
# ================================================================== #

def place_experts_page_rr_pl_first(
    geo: NandGeometry,
    num_experts: int,
    gate_bytes: int,
    up_bytes: int,
    down_bytes: int,
) -> "NandSimulator":
    """
    PL-first 布局：lane 顺序为先铺满一个 CH 的所有 PL，再换下一个 CH。
      ch = lane // P
      pl = lane % P
    对比原 CH-first（ch = lane % C, pl = lane // C）：
      原方案：同一 PL 内先铺满所有 CH → gate/up/down 各占独立 PL → 预取不触发
      新方案：同一 CH 内先铺满所有 PL → gate/up/down 在同一 PL 的不同 CH → 预取全触发
    """
    C, P, S = geo.channels, geo.planes_per_channel, geo.page_size_bytes
    L = C * P  # 总 lane 数，不变

    def map_global_to_phy(g: int):
        in_off   = g % S
        k        = g // S
        lane     = k % L
        t        = k // L
        lane_off = t * S + in_off
        # ↓ 核心改动：PL-first
        ch       = lane // P   # 先填满一个 CH 的所有 PL，再换 CH
        pl       = lane % P
        page     = lane_off // S
        return ch, pl, page, lane_off % S

    placements: List[ExpertPlacement] = []
    global_ptr = 0

    for eid in range(num_experts):
        ep = ExpertPlacement(
            expert_id=eid,
            gate_bytes=gate_bytes, up_bytes=up_bytes, down_bytes=down_bytes,
            page_size_bytes=S,
        )
        write_map: Dict[Tuple[int, int, int], PageWrite] = {}

        for part_name, part_bytes in [("gate", gate_bytes), ("up", up_bytes), ("down", down_bytes)]:
            remaining = part_bytes
            while remaining > 0:
                g = global_ptr
                ch, pl, page, page_off = map_global_to_phy(g)
                chunk = min(S - page_off, remaining)
                key = (ch, pl, page)
                if key not in write_map:
                    pw = PageWrite(expert_id=eid, write_index=len(write_map),
                                   ch=ch, pl=pl, size_bytes=S)
                    write_map[key] = pw
                    ep.writes.append(pw)
                write_map[key].slices.append(PageSlice(
                    ch=ch, pl=pl, page=page,
                    offset=page_off, length=chunk,
                    part=part_name, expert_id=eid,
                ))
                global_ptr += chunk
                remaining  -= chunk

        placements.append(ep)

    return NandSimulator(geo=geo, placements=placements)


# ================================================================== #
#  类型别名                                                            #
# ================================================================== #

SliceKey  = Tuple[int, int, int, int]   # (eid, ch, pl, page)
DistRowCh = Dict[int, Dict[int, int]]   # page -> {ch -> plane_count}


# ================================================================== #
#  核心模拟器                                                          #
# ================================================================== #

class NandSimulator:
    def __init__(self, geo: NandGeometry, placements: List[ExpertPlacement]):
        self.geo = geo
        self.placements = placements
        self._ep_map: Dict[int, ExpertPlacement] = {ep.expert_id: ep for ep in placements}
        self._all_slices: List[PageSlice] = [
            slc for ep in placements for slc in ep.all_slices()
        ]

    def _pages_touched(self, eid: int, part: str) -> Set[SliceKey]:
        return {
            (eid, slc.ch, slc.pl, slc.page)
            for slc in self._ep_map[eid].all_slices()
            if slc.part == part
        }

    def _preloaded_pages(
        self,
        cur_eid: int, cur_part: str,
        nxt_eid: int, nxt_part: str,
    ) -> Set[SliceKey]:
        cur_pl_page_chs: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        for slc in self._ep_map[cur_eid].all_slices():
            if slc.part == cur_part:
                cur_pl_page_chs[(slc.pl, slc.page)].add(slc.ch)

        return {
            (nxt_eid, slc.ch, slc.pl, slc.page)
            for slc in self._ep_map[nxt_eid].all_slices()
            if slc.part == nxt_part
            and (slc.pl, slc.page) in cur_pl_page_chs
            and slc.ch not in cur_pl_page_chs[(slc.pl, slc.page)]
        }

    def _build_dist_row_ch(self, p_need: Set[SliceKey], eid: int, part: str) -> DistRowCh:
        dist: DistRowCh = defaultdict(lambda: defaultdict(int))
        for slc in self._ep_map[eid].all_slices():
            if slc.part == part and (eid, slc.ch, slc.pl, slc.page) in p_need:
                dist[slc.page][slc.ch] += 1
        return dist

    def _calc_part_stats(self, dist_row_ch: DistRowCh, n_new_pages: int,
                         tR_sec: float, tX_sec: float) -> dict:
        if not dist_row_ch:
            return dict(time_sec=0.0, tr_sec=0.0, tx_sec=0.0,
                        crit_ch=-1, crit_planes=0, crit_tx_sec=0.0,
                        cached_planes=0, saved_sec=0.0, _dist_row_ch={})

        sorted_pages = sorted(dist_row_ch.keys())
        tx_total = sum(max(dist_row_ch[pg].values()) * tX_sec for pg in sorted_pages)
        tr_total = n_new_pages * tR_sec

        total_planes_per_ch: Dict[int, int] = defaultdict(int)
        for pg in sorted_pages:
            for ch, cnt in dist_row_ch[pg].items():
                total_planes_per_ch[ch] += cnt

        crit_ch     = max(total_planes_per_ch, key=total_planes_per_ch.get)
        crit_planes = total_planes_per_ch[crit_ch]
        crit_tx_sec = crit_planes * tX_sec

        return dict(
            # time_sec 暂时先用 tr+tx，后续在 estimate_* 里会用真实 hid 修正
            time_sec=tr_total + tx_total, tr_sec=tr_total, tx_sec=tx_total,
            crit_ch=crit_ch, crit_planes=crit_planes, crit_tx_sec=crit_tx_sec,
            cached_planes=0, saved_sec=0.0, _dist_row_ch=dist_row_ch,
        )

    # -------------------------------------------------------------- #
    #  可视化1：精确 slice 级别                                        #
    # -------------------------------------------------------------- #
    def visualize_layout_channel_page_plane(
        self,
        max_pages=None,
        show_grid=True,
        title=None,
        figsize=(18, 7),
        annotate_expert_id=True,
        annotate_min_width_frac=0.06,
        annotate_fontsize=7,
    ):
        C = self.geo.channels
        P = self.geo.planes_per_channel
        S = self.geo.page_size_bytes

        max_page_seen = max((sl.page for sl in self._all_slices), default=0)
        need_pages = max_page_seen + 1
        if max_pages is None:
            max_pages = need_pages
        else:
            max_pages = max(1, min(max_pages, need_pages))

        part_colors = {
            "gate": mpl.colors.to_rgba("#1f77b4"),
            "up":   mpl.colors.to_rgba("#ff7f0e"),
            "down": mpl.colors.to_rgba("#2ca02c"),
        }

        fig, axes = plt.subplots(1, C, figsize=figsize, sharey=True, squeeze=False)
        axes = axes[0]

        for ch in range(C):
            ax = axes[ch]
            ax.set_facecolor("white")

            for sl in self._all_slices:
                if sl.ch != ch or sl.page >= max_pages:
                    continue
                x = sl.pl + (sl.offset / S)
                w = sl.length / S
                rect = Rectangle(
                    (x, sl.page + 0.1), w, 0.8,
                    facecolor=part_colors.get(sl.part, (0.5, 0.5, 0.5, 1.0)),
                    edgecolor="black", linewidth=0.25, alpha=0.85,
                )
                ax.add_patch(rect)
                if annotate_expert_id and w >= annotate_min_width_frac:
                    ax.text(
                        x + w / 2, sl.page + 0.5, str(sl.expert_id),
                        ha="center", va="center",
                        fontsize=annotate_fontsize, color="black", clip_on=True,
                    )

            ax.set_title(f"CH{ch}", fontsize=11)
            ax.set_xlim(0, P)
            ax.set_ylim(0, max_pages)
            ax.set_xticks([i + 0.5 for i in range(P)])
            ax.set_xticklabels([f"PL{i}" for i in range(P)], fontsize=9)
            ax.set_xlabel("Plane(col)+fraction", fontsize=9)

            if show_grid:
                for xline in range(P + 1):
                    ax.axvline(xline, color="gray", linewidth=0.4, alpha=0.35)
                for yline in range(max_pages + 1):
                    ax.axhline(yline, color="gray", linewidth=0.4, alpha=0.25)

        axes[0].set_ylabel("Page(row)", fontsize=10)
        fig.suptitle(title or "Layout", fontsize=12)
        handles = [
            mpl.patches.Patch(color=part_colors[p], label=p)
            for p in ("gate", "up", "down")
        ]
        fig.legend(handles=handles, loc="upper right", framealpha=0.9)
        plt.tight_layout()
        return fig, axes


# ================================================================== #
#  可视化2：整格级别                                                   #
# ================================================================== #

def visualize_layout(
    sim: NandSimulator,
    *,
    expert_ids: List[int] = None,
    part_order: List[str] = None,
    title: str = None,
    max_pages: int = None,
    figsize: Tuple[float, float] = None,
    save_path: str = None,
) -> None:
    if part_order is None:
        part_order = ["gate", "up", "down"]
    if expert_ids is None:
        expert_ids = [ep.expert_id for ep in sim.placements]

    C = sim.geo.channels
    P = sim.geo.planes_per_channel

    all_slices = [
        slc for eid in expert_ids
        for slc in sim._ep_map[eid].all_slices()
        if slc.part in part_order
    ]
    if not all_slices:
        print("[visualize_layout] no data.")
        return

    max_page_seen = max(slc.page for slc in all_slices)
    need_pages    = max_page_seen + 1
    max_pages     = max(1, min(max_pages, need_pages)) if max_pages else need_pages

    part_colors = {
        "gate": mpl.colors.to_rgba("#1f77b4"),
        "up":   mpl.colors.to_rgba("#ff7f0e"),
        "down": mpl.colors.to_rgba("#2ca02c"),
    }
    annotate_fontsize = max(5, min(8, int(60 / max(max_pages, P))))
    if figsize is None:
        figsize = (max(10, C * max(2.5, P * 0.5)), max(5, max_pages * 0.5 + 1.5))

    fig, axes = plt.subplots(1, C, figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]

    for ch in range(C):
        ax = axes[ch]
        for pl in range(P):
            for pg in range(max_pages):
                ax.add_patch(Rectangle((pl, pg), 1, 1,
                    linewidth=0.4, edgecolor="#aaaaaa", facecolor="white", zorder=0))

        cell: Dict[Tuple[int, int], List[Tuple[str, int]]] = defaultdict(list)
        for eid in expert_ids:
            for slc in sim._ep_map[eid].all_slices():
                if slc.ch == ch and slc.part in part_order and slc.page < max_pages:
                    entry = (slc.part, eid)
                    if entry not in cell[(slc.pl, slc.page)]:
                        cell[(slc.pl, slc.page)].append(entry)

        for (pl, pg), entries in cell.items():
            n = len(entries)
            w = 1.0 / n
            for i, (part, eid) in enumerate(entries):
                color = list(part_colors.get(part, (0.5, 0.5, 0.5, 1.0)))
                ax.add_patch(Rectangle((pl + i * w, pg), w, 1,
                    linewidth=0, facecolor=color, alpha=0.85, zorder=1))
                ax.annotate(f"E{eid}\n{part[:1].upper()}",
                    xy=(pl + (i + 0.5) * w, pg + 0.5), ha="center", va="center",
                    fontsize=annotate_fontsize, color="black", clip_on=True, zorder=2)

        ax.set_title(f"CH{ch}", fontsize=11)
        ax.set_xlim(0, P)
        ax.set_ylim(0, max_pages)
        ax.set_xticks([i + 0.5 for i in range(P)])
        ax.set_xticklabels([f"PL{i}" for i in range(P)], fontsize=9)
        ax.set_xlabel("Plane", fontsize=8)
        ax.invert_yaxis()
        ax.set_yticks(range(max_pages))
        ax.set_yticklabels([str(pg) for pg in range(max_pages)], fontsize=7)

    axes[0].set_ylabel("Page(row)", fontsize=10)
    handles = [mpl.patches.Patch(color=part_colors[p], label=p) for p in ("gate", "up", "down")]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title or "Layout", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Layout saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ================================================================== #
#  辅助：根据全局 page→TX 表，计算某个 step 的真实 hid                #
# ================================================================== #

def _calc_step_hid(
    step_pages: Set[int],           # 该 step 实际需要做 tR 的 page 集合（new_pages）
    global_row_list: List[int],     # 全局 page 排序列表
    global_row_tx:   Dict[int, float],  # page → 合计 TX
    tR_sec: float,
) -> float:
    """
    对 step 内每个 new_page，查找它在 global_row_list 中的前一个 page 的合计 TX，
    计算 min(tR, prev_tx) 之和，即该 step 的 tR 被掩盖量。
    """
    if not step_pages:
        return 0.0

    # 建立 page → 在 global_row_list 中的位置索引
    page_to_idx: Dict[int, int] = {pg: i for i, pg in enumerate(global_row_list)}

    hid = 0.0
    for pg in step_pages:
        idx = page_to_idx.get(pg, -1)
        if idx <= 0:
            # 第一个全局 page，没有前驱，tR 无法被掩盖
            continue
        prev_pg  = global_row_list[idx - 1]
        prev_tx  = global_row_tx[prev_pg]
        hid     += min(tR_sec, prev_tx)
    return hid


# ================================================================== #
#  延迟估算：单 Expert                                                 #
# ================================================================== #

def estimate_expert_latency(
    sim: NandSimulator,
    expert_id: int,
    *,
    bw_total_Bps: float,
    tR_sec: float,
    part_order: List[str] = None,
    intra_expert_cache: bool = True,
    seen_pages_external: Optional[Set[int]] = None,
) -> dict:
    if part_order is None:
        part_order = ["gate", "up", "down"]

    tX_sec = sim.geo.page_size_bytes / (bw_total_Bps / sim.geo.channels)
    cache_pages:   Set[SliceKey] = set()
    parts_out:     Dict[str, dict] = {}
    global_row_tx: Dict[int, float] = defaultdict(float)
    seen_pages:    Set[int] = set(seen_pages_external) if seen_pages_external else set()

    # ---- Phase 1：收集各 part 的 dist_row_ch 和 new_pages ----
    part_new_pages: Dict[str, Set[int]] = {}

    for idx, part in enumerate(part_order):
        P_all      = sim._pages_touched(expert_id, part)
        cached_set = P_all & cache_pages if intra_expert_cache else set()
        P_need     = P_all - cached_set

        dist_row_ch = sim._build_dist_row_ch(P_need, expert_id, part)
        part_pages  = set(dist_row_ch.keys())
        new_pages   = part_pages - seen_pages
        seen_pages |= part_pages

        stats = sim._calc_part_stats(dist_row_ch, len(new_pages), tR_sec, tX_sec)
        stats["cached_planes"] = len(cached_set)
        stats["saved_sec"]     = len(cached_set) * tX_sec
        parts_out[part] = stats
        part_new_pages[part] = new_pages

        for pg, ch_dict in dist_row_ch.items():
            global_row_tx[pg] += max(ch_dict.values()) * tX_sec

        if intra_expert_cache and idx + 1 < len(part_order):
            cache_pages |= sim._preloaded_pages(
                expert_id, part, expert_id, part_order[idx + 1])

    # ---- Phase 2 & 3：全局 hid ----
    global_row_list = sorted(global_row_tx.keys())
    global_tx_list  = [global_row_tx[k] for k in global_row_list]
    N_rows     = len(global_row_list)
    total_tx   = sum(global_tx_list)
    total_hid  = sum(min(tR_sec, global_tx_list[j - 1]) for j in range(1, N_rows))
    total_time = N_rows * tR_sec + total_tx - total_hid

    # ---- 修正每个 part 的 time_sec（减去真实 hid）----
    for part in part_order:
        st  = parts_out[part]
        hid = _calc_step_hid(part_new_pages[part], global_row_list, global_row_tx, tR_sec)
        st["hid_sec"]  = hid
        st["time_sec"] = st["tr_sec"] + st["tx_sec"] - hid

    ep = sim._ep_map[expert_id]
    total_bytes = ep.gate_bytes + ep.up_bytes + ep.down_bytes

    return dict(
        expert_id=expert_id, tX_sec=tX_sec, tR_sec=tR_sec,
        parts=parts_out,
        global_row_list=global_row_list, global_tx_list=global_tx_list,
        N_rows=N_rows,
        tr_total_sec=N_rows * tR_sec, tx_total_sec=total_tx,
        hid_total_sec=total_hid, total_time_sec=total_time,
        total_bytes=total_bytes,
        effective_bw_Bps=total_bytes / total_time if total_time > 0 else 0.0,
    )


# ================================================================== #
#  延迟估算：顺序多 Expert                                            #
# ================================================================== #

def estimate_sequential_latency(
    sim: NandSimulator,
    expert_ids: List[int],
    *,
    bw_total_Bps: float,
    tR_sec: float,
    part_order: List[str] = None,
    intra_expert_cache: bool = True,
    inter_expert_cache: bool = True,
) -> dict:
    if part_order is None:
        part_order = ["gate", "up", "down"]

    tX_sec = sim.geo.page_size_bytes / (bw_total_Bps / sim.geo.channels)
    steps: List[Tuple[int, str]] = [(eid, part) for eid in expert_ids for part in part_order]

    cache_pages:    Set[SliceKey] = set()
    global_row_tx:  Dict[int, float] = defaultdict(float)
    step_stats:     Dict[Tuple[int, str], dict] = {}
    step_new_pages: Dict[Tuple[int, str], Set[int]] = {}
    seen_pages:     Set[int] = set()

    for i, (eid, part) in enumerate(steps):
        P_all      = sim._pages_touched(eid, part)
        cached_set = P_all & cache_pages
        P_need     = P_all - cached_set

        dist_row_ch = sim._build_dist_row_ch(P_need, eid, part)
        part_pages  = set(dist_row_ch.keys())
        new_pages   = part_pages - seen_pages
        seen_pages |= part_pages

        stats = sim._calc_part_stats(dist_row_ch, len(new_pages), tR_sec, tX_sec)
        stats["cached_planes"] = len(cached_set)

        # ── 直接计算节省（TX + tR）──────────────────────────────────
        saved_tx_sec     = len(cached_set) * tX_sec
        cached_pages_hit = {pg for (_, _, _, pg) in cached_set}
        need_pages_set   = {pg for (_, _, _, pg) in P_need}
        saved_pages      = cached_pages_hit - need_pages_set   # tR 完全被跳过的 page
        saved_tr_sec     = len(saved_pages) * tR_sec
        stats["saved_tx_sec"]  = saved_tx_sec
        stats["saved_tr_sec"]  = saved_tr_sec
        stats["saved_sec"]     = saved_tx_sec + saved_tr_sec
        # ── 标记预取来源 ─────────────────────────────────────────────
        if cached_set:
            prev_eid = steps[i - 1][0] if i > 0 else None
            stats["prefetch_src"] = "intra" if prev_eid == eid else "inter"
        else:
            stats["prefetch_src"] = None
        # ─────────────────────────────────────────────────────────────

        step_stats[(eid, part)]     = stats
        step_new_pages[(eid, part)] = new_pages

        for pg, ch_dict in dist_row_ch.items():
            global_row_tx[pg] += max(ch_dict.values()) * tX_sec

        if i + 1 < len(steps):
            nxt_eid, nxt_part = steps[i + 1]
            is_intra = (eid == nxt_eid)
            if (is_intra and intra_expert_cache) or (not is_intra and inter_expert_cache):
                cache_pages |= sim._preloaded_pages(eid, part, nxt_eid, nxt_part)

    # ---- Phase 2 & 3：全局 hid（不变）----
    global_row_list = sorted(global_row_tx.keys())
    global_tx_list  = [global_row_tx[k] for k in global_row_list]
    N_rows     = len(global_row_list)
    total_tx   = sum(global_tx_list)
    total_hid  = sum(min(tR_sec, global_tx_list[j - 1]) for j in range(1, N_rows))
    total_time = N_rows * tR_sec + total_tx - total_hid

    for (eid, part), st in step_stats.items():
        hid = _calc_step_hid(
            step_new_pages[(eid, part)], global_row_list, global_row_tx, tR_sec)
        st["hid_sec"]  = hid
        st["time_sec"] = st["tr_sec"] + st["tx_sec"] - hid

    # ---- per_expert_summary（同步累加新字段）----
    total_bytes = sum(
        sim._ep_map[eid].gate_bytes + sim._ep_map[eid].up_bytes + sim._ep_map[eid].down_bytes
        for eid in expert_ids)

    per_expert_summary: Dict[int, dict] = {}
    for eid in expert_ids:
        s = dict(crit_planes=0, crit_tx_sec=0.0, tr_sec=0.0,
                 cached_planes=0, saved_sec=0.0, saved_tr_sec=0.0, saved_tx_sec=0.0,
                 bytes=sim._ep_map[eid].gate_bytes + sim._ep_map[eid].up_bytes + sim._ep_map[eid].down_bytes)
        for part in part_order:
            st = step_stats[(eid, part)]
            s["crit_planes"]   += st["crit_planes"]
            s["crit_tx_sec"]   += st["crit_tx_sec"]
            s["tr_sec"]        += st["tr_sec"]
            s["cached_planes"] += st["cached_planes"]
            s["saved_sec"]     += st["saved_sec"]
            s["saved_tr_sec"]  += st["saved_tr_sec"]
            s["saved_tx_sec"]  += st["saved_tx_sec"]
        per_expert_summary[eid] = s

    return dict(
        expert_ids=expert_ids, tX_sec=tX_sec, tR_sec=tR_sec,
        intra_expert_cache=intra_expert_cache, inter_expert_cache=inter_expert_cache,
        step_stats=step_stats, per_expert_summary=per_expert_summary,
        global_row_list=global_row_list, global_tx_list=global_tx_list,
        N_rows=N_rows,
        tr_total_sec=N_rows * tR_sec, tx_total_sec=total_tx,
        hid_total_sec=total_hid, total_time_sec=total_time,
        total_bytes=total_bytes,
        effective_bw_Bps=total_bytes / total_time if total_time > 0 else 0.0,
    )


# ================================================================== #
#  打印表格：多 Expert 独立延迟                                        #
# ================================================================== #

def print_multi_expert_latency_table(
    sim: NandSimulator,
    *,
    num_experts: int,
    bw_total_Bps: float,
    tR_sec: float,
    part_order: List[str] = None,
    intra_expert_cache: bool = True,
    csv_path: Optional[str] = None,
):
    if part_order is None:
        part_order = ["gate", "up", "down"]

    results = [
        estimate_expert_latency(sim, eid,
            bw_total_Bps=bw_total_Bps, tR_sec=tR_sec,
            part_order=part_order, intra_expert_cache=intra_expert_cache)
        for eid in range(num_experts)
    ]

    tX_us = results[0]["tX_sec"] * 1e6
    tR_us = tR_sec * 1e6
    W = 160

    col_header = "  ".join(
        f"{'crit_ch':>7} {'tR(us)':>7} {'tX(us)':>8} {'c_pl':>5} {'saved(us)':>9} {'time(us)':>8}"
        for _ in part_order
    )
    print(f"\n{'='*W}")
    print(f"  Multi-Expert Latency  |  tR={tR_us:.1f}us  tX={tX_us:.4f}us  "
          f"intra_cache={'ON' if intra_expert_cache else 'OFF'}")
    print(f"{'='*W}")
    print(f"  {'EID':>3}  {col_header}  |"
          f"  {'tR_tot':>8} {'tX_tot':>8} {'hid':>8} {'TOTAL':>9} {'BW(GB/s)':>9}")
    print(f"{'-'*W}")

    rows_csv = []
    for r in results:
        eid = r["expert_id"]
        cells = []
        for part in part_order:
            st = r["parts"][part]
            cells.append(
                f"  {str(st['crit_ch']):>7} {st['tr_sec']*1e6:>7.2f} {st['crit_tx_sec']*1e6:>8.2f} "
                f"{st['cached_planes']:>5} {st['saved_sec']*1e6:>9.2f} {st['time_sec']*1e6:>8.2f}"
            )
        print(f"  {eid:>3}  {'  '.join(cells)}  |"
              f"  {r['tr_total_sec']*1e6:>8.2f}  {r['tx_total_sec']*1e6:>8.2f}"
              f"  {r['hid_total_sec']*1e6:>8.2f}  {r['total_time_sec']*1e6:>9.2f}"
              f"  {r['effective_bw_Bps']/1e9:>9.3f}")
        rows_csv.append([eid, r["total_time_sec"] * 1e6, r["effective_bw_Bps"] / 1e9])

    print(f"{'='*W}\n")
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerows([["expert_id", "total_us", "bw_GBps"]] + rows_csv)


# ================================================================== #
#  打印表格：顺序多 Expert 延迟                                        #
# ================================================================== #

def print_sequential_latency_table(
    sim: NandSimulator,
    expert_ids: List[int],
    *,
    bw_total_Bps: float,
    tR_sec: float,
    part_order: List[str] = None,
    intra_expert_cache: bool = True,
    inter_expert_cache: bool = True,
    csv_path: Optional[str] = None,
    quiet: bool = False,
) -> dict:
    if part_order is None:
        part_order = ["gate", "up", "down"]

    r = estimate_sequential_latency(
        sim, expert_ids,
        bw_total_Bps=bw_total_Bps, tR_sec=tR_sec,
        part_order=part_order,
        intra_expert_cache=intra_expert_cache,
        inter_expert_cache=inter_expert_cache,
    )

    tX_us = r["tX_sec"] * 1e6
    tR_us = tR_sec * 1e6
    W = 125
    
    if quiet:
        # Quiet模式：直接返回结果，不打印表格
        return r

    print(f"\n{'='*W}")
    print(f"  Sequential Read : experts={expert_ids}")
    print(f"  tR={tR_us:.1f}us  tX={tX_us:.4f}us  "
          f"intra={'ON' if intra_expert_cache else 'OFF'}  "
          f"inter={'ON' if inter_expert_cache else 'OFF'}")
    print(f"{'='*W}")
    print(f"\n  [Step Detail]")
    # ↓ 新增 hid(us) 列，放在 tR 和 tX 之间
    print(f"  {'EID':>4} {'PART':>6}  "
          f"{'crit_ch':>7} {'tR(us)':>7} {'hid(us)':>8} {'tX(us)':>8} "
          f"{'c_pl':>5} {'saved(us)':>9} {'time(us)':>9}  note")
    print(f"  {'-'*110}")

    steps    = [(eid, part) for eid in expert_ids for part in part_order]
    prev_eid = None
    rows_csv = []

    for i, (eid, part) in enumerate(steps):
        st    = r["step_stats"][(eid, part)]
        notes = []
        if prev_eid is not None and eid != prev_eid:
            notes.append(f"<-- inter(E{prev_eid}→E{eid})")
        if st["cached_planes"] > 0:
            src = "intra" if (prev_eid == eid) else "inter"
            notes.append(f"[prefetch({src}):{st['cached_planes']}pl]")
        if st["tr_sec"] == 0.0 and st["crit_planes"] > 0:
            notes.append("[tR saved]")

        # ↓ 新增 hid_sec 列打印
        print(f"  {eid:>4} {part:>6}  "
              f"{str(st['crit_ch']):>7} {st['tr_sec']*1e6:>7.2f} {st['hid_sec']*1e6:>8.2f} "
              f"{st['crit_tx_sec']*1e6:>8.2f} "
              f"{st['cached_planes']:>5} {st['saved_sec']*1e6:>9.2f} {st['time_sec']*1e6:>9.2f}  "
              f"{'  '.join(notes)}")

        rows_csv.append([eid, part,
                         f"{st['tr_sec']*1e6:.2f}", f"{st['hid_sec']*1e6:.2f}",   # ↓ 新增 hid
                         f"{st['crit_tx_sec']*1e6:.2f}",
                         st['cached_planes'], f"{st['saved_sec']*1e6:.2f}",
                         f"{st['time_sec']*1e6:.2f}"])
        prev_eid = eid

    print(f"\n  [Per-Expert Summary]")
    print(f"  {'EID':>4}  {'tR(us)':>8} {'tX(us)':>9} {'c_pl':>6} {'saved(us)':>10} {'bytes':>12}")
    print(f"  {'-'*60}")
    for eid in expert_ids:
        s = r["per_expert_summary"][eid]
        print(f"  {eid:>4}  {s['tr_sec']*1e6:>8.2f} {s['crit_tx_sec']*1e6:>9.2f} "
              f"{s['cached_planes']:>6} {s['saved_sec']*1e6:>10.2f} {s['bytes']:>12,}")

    # ========== 带宽分析 ==========
    # 计算各项时间
    total_time = r['total_time_sec']
    pure_tx_time = r['tx_total_sec']  # 纯数据传输时间
    tr_time = r['tr_total_sec']       # 总tR时间
    hid_time = r['hid_total_sec']     # 被掩盖的tR时间
    effective_tr_time = tr_time - hid_time  # 实际阻塞的tR时间
    
    # 计算节省的时间
    total_saved_tx = sum(r['step_stats'][k]['saved_tx_sec'] for k in r['step_stats'])
    total_saved_tr = sum(r['step_stats'][k]['saved_tr_sec'] for k in r['step_stats'])
    
    # 理论峰值带宽（假设只有TX，无tR）
    theoretical_bw = bw_total_Bps
    
    # 实际有效带宽
    effective_bw = r['effective_bw_Bps']
    
    # 带宽利用率
    bw_utilization = (effective_bw / theoretical_bw) * 100 if theoretical_bw > 0 else 0
    
    # 如果没有tR延迟时的带宽（理想情况）
    ideal_time_if_no_tr = pure_tx_time
    ideal_bw_if_no_tr = r['total_bytes'] / ideal_time_if_no_tr if ideal_time_if_no_tr > 0 else theoretical_bw
    
    # 计算各类开销导致的带宽损失
    tr_overhead_time = effective_tr_time  # tR造成的额外时间
    
    print(f"\n  [Global Summary]")
    print(f"  {'N_rows':>14} : {r['N_rows']}")
    print(f"  {'tR_total':>14} : {r['tr_total_sec']*1e6:.2f} us")
    print(f"  {'tX_total':>14} : {r['tx_total_sec']*1e6:.2f} us")
    print(f"  {'hid_total':>14} : {r['hid_total_sec']*1e6:.2f} us")
    print(f"  {'TOTAL':>14} : {r['total_time_sec']*1e6:.2f} us")
    print(f"  {'total_bytes':>14} : {r['total_bytes']:,} bytes")
    
    print(f"\n  [Bandwidth Analysis]")
    print(f"  {'='*50}")
    print(f"  {'Theoretical BW':>20} : {theoretical_bw/1e9:>8.3f} GB/s (额定带宽)")
    print(f"  {'Effective BW':>20} : {effective_bw/1e9:>8.3f} GB/s (实际带宽)")
    print(f"  {'Utilization':>20} : {bw_utilization:>8.2f} %")
    print(f"  {'-'*50}")
    
    # 带宽损失分解
    print(f"\n  [Bandwidth Loss Breakdown]")
    print(f"  {'='*50}")
    
    # 1. tR延迟导致的损失（未掩盖部分）
    if total_time > 0:
        tr_loss_pct = (effective_tr_time / total_time) * 100
        print(f"  1. tR Latency Overhead")
        print(f"     {'- Blocked tR time':>22} : {effective_tr_time*1e6:>8.2f} us ({tr_loss_pct:.1f}%)")
        print(f"     {'- Hid (overlapped)':>22} : {hid_time*1e6:>8.2f} us (saved)")
    
    # 2. 预取带来的收益
    if total_saved_tx > 0 or total_saved_tr > 0:
        print(f"\n  2. Prefetch Benefits")
        tx_saved_pct = (total_saved_tx / total_time) * 100 if total_time > 0 else 0
        tr_saved_pct = (total_saved_tr / total_time) * 100 if total_time > 0 else 0
        print(f"     {'- TX saved':>22} : {total_saved_tx*1e6:>8.2f} us ({tx_saved_pct:.1f}%)")
        print(f"     {'- tR saved':>22} : {total_saved_tr*1e6:>8.2f} us ({tr_saved_pct:.1f}%)")
    
    # 3. Page粒度损失（如果所有page都未满）
    # 计算每个page的有效数据利用率
    total_pages_accessed = sum(len(r['step_stats'][k].get('_dist_row_ch', {})) for k in r['step_stats'])
    if total_pages_accessed > 0:
        avg_page_util = (r['total_bytes'] / sim.geo.page_size_bytes) / total_pages_accessed * 100
        if avg_page_util < 100:
            page_granularity_loss = 100 - avg_page_util
            print(f"\n  3. Page Granularity Loss")
            print(f"     {'- Page utilization':>22} : {avg_page_util:>8.1f}% (avg)")
            print(f"     {'- Wasted per page':>22} : {page_granularity_loss:>8.1f}% internal fragmentation")
    
    # 4. 关键路径瓶颈分析
    # 找出最关键的channel
    max_planes_per_step = []
    for k, st in r['step_stats'].items():
        dist = st.get('_dist_row_ch', {})
        if dist:
            for pg, ch_dict in dist.items():
                max_planes_per_step.append(max(ch_dict.values()))
    if max_planes_per_step:
        avg_planes = sum(max_planes_per_step) / len(max_planes_per_step)
        max_possible_planes = sim.geo.planes_per_channel
        ch_parallel_util = (avg_planes / max_possible_planes) * 100
        print(f"\n  4. Channel Parallelism")
        print(f"     {'- Avg planes/CH':>22} : {avg_planes:>8.1f} / {max_possible_planes}")
        print(f"     {'- Parallel util':>22} : {ch_parallel_util:>8.1f}% (per-page)")
    
    print(f"\n  [Recommendations]")
    print(f"  {'='*50}")
    if bw_utilization < 50:
        print(f"  [!] 带宽利用率较低 ({bw_utilization:.1f}%)，建议：")
        if effective_tr_time > pure_tx_time * 0.5:
            print(f"     - tR延迟占比较高，考虑增大page_size或增加预取")
        if intra_expert_cache and total_saved_tx < pure_tx_time * 0.1:
            print(f"     - intra预取效果有限，确认布局是否为pl-first")
    else:
        print(f"  [OK] 带宽利用率良好 ({bw_utilization:.1f}%)")
    
    if not intra_expert_cache and not inter_expert_cache:
        print(f"  [TIP] 建议启用--intra和--inter预取以提升性能")
    elif not inter_expert_cache and len(expert_ids) > 1:
        print(f"  [TIP] 建议启用--inter预取以提升多专家场景性能")
    
    print(f"{'='*W}\n")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerows(
                [["eid", "part", "tr_us", "hid_us", "tx_us", "c_pl", "saved_us", "time_us"]]  # ↓ 新增 hid_us
                + rows_csv)

    return r


# ================================================================== #
#  入口示例                                                            #
# ================================================================== #

def parse_expert_ids(value: str) -> List[int]:
    """解析 expert_ids，支持 '0,1,2,3' 或 '0-5' 或 '0-3,5,7-9' 格式"""
    result = []
    for part in value.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='NAND Flash MoE Simulator - 模拟MoE专家存储在NAND上的读取行为',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  nand.exe -c 8 -p 8 -s 16384 --bw 30e9 --tr 22e-6 -e 0,1,2,3
  nand.exe -c 8 -p 8 --layout pl-first -e 0-9 --csv result.csv
  nand.exe --experts 0,1,5 --intra --inter --viz layout.png
        """
    )
    
    # 硬件几何参数
    hw_group = parser.add_argument_group('硬件几何参数 (NAND Geometry)')
    hw_group.add_argument('-c', '--channels', type=int, default=8,
                          help='NAND通道数 (默认: 8)')
    hw_group.add_argument('-p', '--planes', type=int, default=8,
                          help='每通道平面数 (默认: 8)')
    hw_group.add_argument('-s', '--page-size', type=int, default=16384,
                          help='每页字节数 (默认: 16384 = 16KB)')
    
    # 性能参数
    perf_group = parser.add_argument_group('性能参数 (Performance)')
    perf_group.add_argument('--bw', type=float, default=30e9,
                            help='总带宽 (字节/秒，默认: 30e9 = 30GB/s)')
    perf_group.add_argument('--tr', type=float, default=22e-6,
                            help='读延迟tR (秒，默认: 22e-6 = 22us)')
    
    # 专家参数
    expert_group = parser.add_argument_group('专家参数 (Expert)')
    expert_group.add_argument('-e', '--experts', type=str, default='0,1,2',
                              help='要模拟的专家ID列表，支持逗号或范围格式，如 "0,1,2" 或 "0-9" 或 "0-3,5,7-9" (默认: 0,1,2)')
    expert_group.add_argument('--num-experts', type=int, default=10,
                              help='总专家数量，用于布局 (默认: 10)')
    expert_group.add_argument('--gate-bytes', type=int, default=294912,
                              help='gate部分字节数 (默认: 294912 = 288KB)')
    expert_group.add_argument('--up-bytes', type=int, default=294912,
                              help='up部分字节数 (默认: 294912 = 288KB)')
    expert_group.add_argument('--down-bytes', type=int, default=294912,
                              help='down部分字节数 (默认: 294912 = 288KB)')
    
    # 布局选项
    layout_group = parser.add_argument_group('布局选项 (Layout)')
    layout_group.add_argument('--layout', choices=['ch-first', 'pl-first'], default='pl-first',
                              help='布局方式: ch-first (默认，预取不触发) 或 pl-first (预取优化) (默认: pl-first)')
    
    # 缓存选项
    cache_group = parser.add_argument_group('缓存选项 (Cache/Prefetch)')
    cache_group.add_argument('--intra', action='store_true', default=True,
                             help='启用intra-expert预取 (同专家gate->up->down预取) (默认: True)')
    cache_group.add_argument('--no-intra', dest='intra', action='store_false',
                             help='禁用intra-expert预取')
    cache_group.add_argument('--inter', action='store_true', default=True,
                             help='启用inter-expert预取 (跨专家预取) (默认: True)')
    cache_group.add_argument('--no-inter', dest='inter', action='store_false',
                             help='禁用inter-expert预取')
    
    # 输出选项
    output_group = parser.add_argument_group('输出选项 (Output)')
    output_group.add_argument('--csv', type=str,
                              help='输出CSV文件路径')
    output_group.add_argument('--viz', type=str,
                              help='可视化输出图片路径 (如 layout.png)')
    output_group.add_argument('--max-pages', type=int, default=20,
                              help='可视化最大页数 (默认: 20)')
    output_group.add_argument('--no-viz', action='store_true',
                              help='不显示可视化窗口')
    output_group.add_argument('--quiet', '-q', action='store_true',
                              help='静默模式，只输出结果')
    
    # GUI 选项
    gui_group = parser.add_argument_group('GUI 选项')
    gui_group.add_argument('--gui', action='store_true',
                           help='启动图形界面（忽略其他参数）')
    
    args = parser.parse_args()
    
    # 如果指定了 --gui，启动图形界面
    if args.gui:
        try:
            from gui import main_gui
            main_gui()
            return 0
        except ImportError as e:
            print(f"错误: 无法启动 GUI: {e}")
            return 1
    
    # 解析 expert_ids
    try:
        expert_ids = parse_expert_ids(args.experts)
    except ValueError as e:
        print(f"错误: expert_ids格式无效 '{args.experts}': {e}")
        return 1
    
    # 验证参数
    if args.channels <= 0 or args.planes <= 0 or args.page_size <= 0:
        print("错误: channels, planes, page-size 必须大于0")
        return 1
    if args.bw <= 0 or args.tr <= 0:
        print("错误: bw 和 tr 必须大于0")
        return 1
    if not expert_ids:
        print("错误: 至少需要指定一个expert")
        return 1
    
    # 创建几何配置
    geo = NandGeometry(
        channels=args.channels,
        planes_per_channel=args.planes,
        page_size_bytes=args.page_size
    )
    
    if not args.quiet:
        print(f"\n{'='*60}")
        print("NAND Flash MoE Simulator")
        print(f"{'='*60}")
        print(f"硬件配置: {args.channels}通道 x {args.planes}平面, 页大小={args.page_size}字节")
        print(f"性能参数: 带宽={args.bw/1e9:.1f}GB/s, tR={args.tr*1e6:.1f}us")
        print(f"布局方式: {args.layout}")
        print(f"缓存选项: intra={'ON' if args.intra else 'OFF'}, inter={'ON' if args.inter else 'OFF'}")
        print(f"模拟专家: {expert_ids}")
        print(f"{'='*60}\n")
    
    # 选择布局函数
    if args.layout == 'pl-first':
        sim = place_experts_page_rr_pl_first(
            geo, args.num_experts,
            args.gate_bytes, args.up_bytes, args.down_bytes
        )
    else:
        sim = place_experts_page_rr(
            geo, args.num_experts,
            args.gate_bytes, args.up_bytes, args.down_bytes
        )
    
    # 可视化
    if args.viz:
        visualize_layout(sim, expert_ids=expert_ids, max_pages=args.max_pages,
                         title=f"Expert Layout ({args.layout})", save_path=args.viz)
        if not args.quiet:
            print(f"[布局图已保存] {args.viz}")
    elif not args.no_viz:
        try:
            visualize_layout(sim, expert_ids=expert_ids, max_pages=args.max_pages,
                             title=f"Expert Layout ({args.layout})")
        except Exception as e:
            if not args.quiet:
                print(f"[可视化失败: {e}]")
    
    # 顺序延迟仿真
    result = print_sequential_latency_table(
        sim, expert_ids,
        bw_total_Bps=args.bw, tR_sec=args.tr,
        intra_expert_cache=args.intra,
        inter_expert_cache=args.inter,
        csv_path=args.csv,
        quiet=args.quiet
    )
    
    if args.csv and not args.quiet:
        print(f"\n[CSV已保存] {args.csv}")
    
    # 输出有效带宽和简要分析
    eff_bw = result['effective_bw_Bps']
    theoretical_bw = args.bw
    utilization = (eff_bw / theoretical_bw) * 100 if theoretical_bw > 0 else 0
    
    if args.quiet:
        # Quiet模式：只输出关键结果
        print(f"\n有效带宽: {eff_bw/1e9:.3f} GB/s (利用率: {utilization:.1f}%)")
    else:
        print(f"\n有效带宽: {eff_bw/1e9:.3f} GB/s")
    
    return 0


def main_with_gui():
    """支持 GUI 的入口"""
    import sys
    
    # 检查是否有命令行参数（除了脚本名）
    if len(sys.argv) == 1:
        # 无参数，启动 GUI
        try:
            from gui import main_gui
            main_gui()
            return 0
        except ImportError:
            # GUI 模块不可用，显示帮助
            print("NAND Flash MoE Simulator")
            print("=" * 60)
            print("运行方式:")
            print("  1. 图形界面: 直接双击运行，或使用 --gui 参数")
            print("  2. 命令行: 使用参数运行，如: python nand.py -c 8 -p 8")
            print("\n使用 --help 查看所有命令行参数")
            return 1
    
    # 检查是否有 --gui 参数
    if "--gui" in sys.argv:
        sys.argv.remove("--gui")
        try:
            from gui import main_gui
            main_gui()
            return 0
        except ImportError as e:
            print(f"错误: 无法启动 GUI: {e}")
            return 1
    
    # 命令行模式
    return main()


if __name__ == "__main__":
    import sys
    sys.exit(main_with_gui())