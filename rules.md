# NAND Flash MoE 仿真器完整规则手册

> 版本：2026-03-23
> 基准代码：MoE NAND仿真器 V6（含 hid 修正，最终确认版）

---

## 一、硬件几何参数

| 参数 | 含义 | 示例 |
|---|---|---|
| `C`（channels） | NAND 通道数 | 8 |
| `P`（planes_per_channel） | 每通道平面数 | 8 |
| `S`（page_size_bytes） | 每页字节数 | 16384 B |
| `L = C × P` | 总 lane 数 | 64 |

---

## 二、布局方式（Layout）

### 2.1 全局地址 → 物理地址映射（两方案共用）

```python
in_off   = g % S               # 页内字节偏移
k        = g // S              # 第几个页槽
lane     = k % L               # 落在哪个 lane
t        = k // L              # 该 lane 内第几轮
lane_off = t * S + in_off
page     = lane_off // S       # 物理页号
```

差异仅在 **lane → (ch, pl)** 的映射方式。

---

### 2.2 方案一：CH-first Round-Robin

```python
ch = lane % C      # 先铺满所有 CH，再换 PL
pl = lane // C
```

**效果**：gate / up / down 各自占据独立的 PL → **intra 预取不触发** ❌

```
Lane:  0  1  2  3  4  5  6  7  8  9 ...
CH:    0  1  2  3  4  5  6  7  0  1 ...
PL:    0  0  0  0  0  0  0  0  1  1 ...
```

---

### 2.3 方案二：PL-first Round-Robin

```python
ch = lane // P     # 先铺满一个 CH 的所有 PL，再换 CH
pl = lane % P
```

**效果**：gate / up / down 分布在同一 PL 的不同 CH 上 → **intra 预取全触发** ✅

```
Lane:  0  1  2  3  4  5  6  7  8  9 ...
CH:    0  0  0  0  0  0  0  0  1  1 ...
PL:    0  1  2  3  4  5  6  7  0  1 ...
```

---

### 2.4 两种布局对比

| 特性 | CH-first | PL-first |
|---|---|---|
| lane→ch | `lane % C` | `lane // P` |
| lane→pl | `lane // C` | `lane % P` |
| gate/up/down 分布 | 各占独立 PL | 同 PL 不同 CH |
| intra 预取触发 | ❌ | ✅ |
| 适用场景 | 基线对比 | 预取优化 |

---

## 三、tR 计算规则

### 3.1 基本原则

> tR 以 **page 编号** 为粒度，与 plane 数无关。  
> 同一 page 上无论多少 PL，只做一次 tR，所有 CH/PL 同时 sense。

### 3.2 何时产生 tR

每遇到一个**首次出现的 page**，产生一次 tR：

$$t_{R,\text{step}} = |\text{new\_pages}| \times t_R$$

```python
new_pages = part_pages - seen_pages
seen_pages |= part_pages
tr_total = len(new_pages) * tR_sec
```

### 3.3 何时 tR = 0（tR saved）

| 情况 | 原因 |
|---|---|
| intra 预取命中 | 同 expert 前一 part 已读过该 page，`new_pages` 为空 |
| inter 预取命中 | 前一 expert 已读过该 page，`seen_pages` 中已有 |
| cache 命中 | `(eid, ch, pl, page)` 全在 `cache_pages`，`P_need` 为空 |

---

## 四、TX 计算规则

### 4.1 基本参数

$$t_X = \frac{S}{bw\_total / C}$$

### 4.2 并行结构

```
同一 page 的 TX 顺序：
  第1轮：所有 CH 并行发 PL0 的数据
  第2轮：所有 CH 并行发 PL1 的数据
  ...
→ 不同 CH 并行（取 max），不同 PL 串行（累加）
```

> **关键原则：TX 瓶颈 = critic channel（plane 数最多的那个 CH）**

### 4.3 per-step per-page 的 TX

$$t_{X,\text{row}}[\text{pg}] = \max_{\text{ch}} \left(\text{planes}_{ch,pg}\right) \times t_X$$

### 4.4 global_row_tx 累加

不同 step 在同一 page 上的 TX **串行执行**，用 `+=` 累加：

$$t_{X,\text{row}}^{\text{global}}[\text{pg}] = \sum_{\text{step}} \max_{\text{ch}} (\text{planes}_{ch,pg}) \times t_X$$

---

## 五、hid（掩盖）规则

### 5.1 物理原理

```
Row i:   |<-- tR_i -->|<-- TX_i -->|
Row i+1:              |<-- tR_{i+1} -->|<-- TX_{i+1} -->|
                      ↑
         tR_{i+1} 与 TX_i 并行 → hid = min(tR, TX_i)
```

**第 j 行的 tR，可以被第 j-1 行的 TX 掩盖**

### 5.2 全局 hid

$$\text{hid\_total} = \sum_{j=1}^{N-1} \min\left(t_R,\ t_{X,j-1}^{\text{global}}\right)$$

### 5.3 总时间公式

$$T_{\text{total}} = N_{\text{rows}} \cdot t_R + \sum_i t_{X,i}^{\text{global}} - \text{hid\_total}$$

### 5.4 per-step hid

对该 step 的每个 new_page，查找其**前驱 page** 的 TX：

$$\text{hid\_step} = \sum_{\text{pg} \in \text{new\_pages}} \min\left(t_R,\ t_{X,\text{prev\_pg}}^{\text{global}}\right)$$

per-step 最终时间：

$$\text{time}^{(\text{step})} = t_{R,\text{step}} + t_{X,\text{step}} - \text{hid\_step}$$

---

## 六、预取规则（Prefetch）

### 6.1 物理原理

tR 是 page row 级别操作，该 row 上**所有 CH、所有 PL** 同时 sense 好。  
TX 按 CH 并行发出 → 读 cur_part 时，同一 `(pl, page)` 上其他 CH 的 nxt_part 数据可被**免费预取**。

### 6.2 触发条件（intra & inter 代码逻辑相同）

```python
# 条件：
#   1. (pl, page) 相同
#   2. CH 不同（cur_part 没读过的 CH）
cur_pl_page_chs[(pl, page)] = {所有读 cur_part 的 ch}
预取集合 = {nxt_part 中，(pl, page) 相同但 ch 不在 cur_pl_page_chs 中的 slice}
```

### 6.3 intra vs inter 对比

| 类型 | 触发时机 | 节省 tR | 节省 TX | 开关 |
|---|---|---|---|---|
| **intra-expert** | gate→up，up→down（同 expert） | ✅ | ✅ | `intra_expert_cache` |
| **inter-expert** | E_prev 末 part → E_cur 首 part | ✅ | ✅ | `inter_expert_cache` |

### 6.4 预取不触发的情况

| 情况 | 原因 |
|---|---|
| 不同 PL | TX 按 PL 串行，不同 PL 是不同轮次 |
| 不同 page | 不同 page row，tR 独立触发 |
| 同 part 内 | 预取只跨相邻 part |
| CH-first 布局 | gate/up/down 各占独立 PL，条件不满足 |

---

## 七、step_stats 字段说明

| 字段 | 含义 |
|---|---|
| `tr_sec` | 该 step 实际产生的 tR 时间 |
| `tx_sec` | 该 step 的总 TX 时间 |
| `crit_tx_sec` | 关键路径 CH 的 TX 时间 |
| `crit_ch` | 关键路径 CH 编号 |
| `crit_planes` | 关键路径 CH 的总 plane 数 |
| `cached_planes` | 命中缓存的 plane 数 |
| `saved_tx_sec` | TX 节省量（cached_planes × tX） |
| `saved_tr_sec` | tR 节省量（跳过的 page × tR） |
| `hid_sec` | 该 step 的 tR 被掩盖量 |
| `time_sec` | `tr + tx - hid` |
| `prefetch_src` | `"intra"` / `"inter"` / `None` |

---

## 八、有效带宽

$$\text{eff\_BW} = \frac{\text{total\_bytes}}{T_{\text{total}}}$$
