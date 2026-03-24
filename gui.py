#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAND MoE Simulator - GUI 版本
支持硬件参数配置、布局可视化、结果展示和 TopK 分析
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import sys
import io
import numpy as np

# 导入核心模块
from nand import (
    NandGeometry, place_experts_page_rr, place_experts_page_rr_pl_first,
    place_experts_tlc,
    NandSimulator, estimate_sequential_latency,
    parse_expert_ids
)

# 尝试导入 matplotlib
matplotlib_available = False
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib_available = True
except ImportError:
    pass

# 导入专家命中仿真模块
from layout_analyse import run_topk_analysis, plot_prefetch_comparison


class RedirectText(io.StringIO):
    """将 stdout 重定向到 tkinter 文本框"""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        
    def write(self, s):
        self.text_widget.insert(tk.END, s)
        self.text_widget.see(tk.END)
        
    def flush(self):
        pass


class NandSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NAND MoE Simulator")
        self.root.geometry("1200x900")
        self.root.minsize(800, 600)
        
        # 运行状态标志，防止同时运行多个任务
        self.is_running = False
        
        # 预设配置
        self.presets = {
            "\u4f4e\u914dNAND": {
                "channels": 4,
                "planes": 4,
                "page_size": 16,
                "bw": 1.75,
                "tr": 50,
            },
            "\u9ad8\u914dNAND": {
                "channels": 8,
                "planes": 8,
                "page_size": 16,
                "bw": 3.75,
                "tr": 22,
            },
        }
        
        # 界面颜色配置
        self.colors = {
            'bg': '#f5f5f5',
            'frame': '#ffffff',
            'accent': '#2196F3',
            'text': '#333333',
            'success': '#4CAF50',
            'warning': '#FF9800',
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # 创建 Notebook 标签页
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 主模拟标签页
        self.sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sim_frame, text='\u4e3b\u6a21\u62df')
        
        # 专家命中仿真标签页
        self.topk_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.topk_frame, text='\u4e13\u5bb6\u547d\u4e2d\u4eff\u771f')
        
        # 初始化两个标签页
        self._init_sim_tab()
        self._init_topk_tab()
    
    def _create_scrollable_left_panel(self, parent):
        """创建带滚动条的左侧面板"""
        # 创建 Canvas 和滚动条容器（固定宽度240像素，容纳长标签）
        container = tk.Frame(parent, bg=self.colors['frame'], bd=2, relief=tk.GROOVE, width=240)
        container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        container.pack_propagate(False)  # 禁止子控件改变容器大小
        
        # 创建 Canvas
        canvas = tk.Canvas(container, bg=self.colors['frame'], highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 创建滚动条
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 创建内部框架存放控件
        left_panel = tk.Frame(canvas, bg=self.colors['frame'])
        canvas.create_window((0, 0), window=left_panel, anchor="nw")
        
        # 绑定事件更新滚动区域
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        left_panel.bind("<Configure>", on_frame_configure)
        
        # 绑定鼠标滚轮（只在 Canvas 区域内有效）
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)
        # 也绑定到内部框架
        left_panel.bind("<MouseWheel>", on_mousewheel)
        
        return left_panel, canvas
    
    def _init_sim_tab(self):
        """初始化主模拟标签页"""
        # 主容器
        main_container = tk.Frame(self.sim_frame, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：带滚动条的参数输入面板
        left_panel, _ = self._create_scrollable_left_panel(main_container)
        
        # 硬件配置参数
        hw_frame = tk.LabelFrame(left_panel, text="\u786c\u4ef6\u914d\u7f6e\u53c2\u6570", 
                                  bg=self.colors['frame'], padx=8, pady=8)
        hw_frame.pack(fill=tk.X, padx=5, pady=(5, 3))
        
        # 加载预设按钮
        tk.Button(hw_frame, text="\u52a0\u8f7d\u9884\u8bbe\u914d\u7f6e", 
                  command=self._show_preset_menu,
                  bg=self.colors['accent'], fg='white',
                  font=('Microsoft YaHei', 9)).pack(fill=tk.X, pady=(0, 8))
        
        # 通道数
        tk.Label(hw_frame, text="\u901a\u9053\u6570 (Channels):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.channels_var = tk.IntVar(value=8)
        tk.Spinbox(hw_frame, from_=1, to=64, textvariable=self.channels_var, 
                   width=15).pack(fill=tk.X, pady=(0, 5))
        
        # Plane 数
        tk.Label(hw_frame, text="\u6bcf\u901a\u9053 Plane \u6570:", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.planes_var = tk.IntVar(value=8)
        tk.Spinbox(hw_frame, from_=1, to=16, textvariable=self.planes_var, 
                   width=15).pack(fill=tk.X, pady=(0, 5))
        
        # Page Size
        tk.Label(hw_frame, text="Page Size (KB):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.page_size_var = tk.IntVar(value=16)
        tk.Spinbox(hw_frame, from_=1, to=64, textvariable=self.page_size_var, 
                   width=15).pack(fill=tk.X, pady=(0, 5))
        
        # 硬件性能参数
        perf_frame = tk.LabelFrame(left_panel, text="\u786c\u4ef6\u6027\u80fd\u53c2\u6570", 
                                   bg=self.colors['frame'], padx=8, pady=8)
        perf_frame.pack(fill=tk.X, padx=5, pady=3)
        
        # 单通道带宽
        tk.Label(perf_frame, text="\u5355\u901a\u9053\u5e26\u5bbd (GB/s):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.bw_var = tk.DoubleVar(value=3.75)
        tk.Entry(perf_frame, textvariable=self.bw_var, width=17).pack(fill=tk.X, pady=(0, 5))
        
        # tR 延迟
        tk.Label(perf_frame, text="tR \u8bfb\u53d6\u5ef6\u8fdf (\u03bcs):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.tr_var = tk.DoubleVar(value=22.0)
        tk.Entry(perf_frame, textvariable=self.tr_var, width=17).pack(fill=tk.X, pady=(0, 5))
        
        # MoE 模型参数
        moe_frame = tk.LabelFrame(left_panel, text="MoE \u6a21\u578b\u53c2\u6570", 
                                  bg=self.colors['frame'], padx=8, pady=8)
        moe_frame.pack(fill=tk.X, padx=5, pady=3)
        
        # Expert 数量
        tk.Label(moe_frame, text="Expert \u603b\u6570:", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.num_experts_var = tk.IntVar(value=512)
        tk.Spinbox(moe_frame, from_=1, to=2048, textvariable=self.num_experts_var, 
                   width=15).pack(fill=tk.X, pady=(0, 5))
        
        # Gate 大小
        tk.Label(moe_frame, text="Gate \u53c2\u6570\u5927\u5c0f (bytes):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.gate_bytes_var = tk.IntVar(value=294912)
        tk.Entry(moe_frame, textvariable=self.gate_bytes_var, width=17).pack(fill=tk.X, pady=(0, 5))
        
        # Up 大小
        tk.Label(moe_frame, text="Up \u53c2\u6570\u5927\u5c0f (bytes):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.up_bytes_var = tk.IntVar(value=294912)
        tk.Entry(moe_frame, textvariable=self.up_bytes_var, width=17).pack(fill=tk.X, pady=(0, 5))
        
        # Down 大小
        tk.Label(moe_frame, text="Down \u53c2\u6570\u5927\u5c0f (bytes):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.down_bytes_var = tk.IntVar(value=294912)
        tk.Entry(moe_frame, textvariable=self.down_bytes_var, width=17).pack(fill=tk.X, pady=(0, 5))
        
        # TopK 和 Expert 选择
        topk_frame = tk.LabelFrame(left_panel, text="\u4e13\u5bb6\u547d\u4e2d\u914d\u7f6e", 
                                   bg=self.colors['frame'], padx=8, pady=8)
        topk_frame.pack(fill=tk.X, padx=5, pady=3)
        
        tk.Label(topk_frame, text="TopK (\u9009\u62e9 Expert \u6570):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.topk_var = tk.IntVar(value=10)
        tk.Spinbox(topk_frame, from_=1, to=50, textvariable=self.topk_var, 
                   width=15).pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(topk_frame, text="Expert IDs (\u7528\u9017\u53f7\u6216\u77ed\u6a2a\u7ebf\u5206\u9694):",
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.expert_ids_var = tk.StringVar(value="0,1,2,3,4,5,6,7,8,9")
        tk.Entry(topk_frame, textvariable=self.expert_ids_var, width=17).pack(fill=tk.X, pady=(0, 5))
        
        # 布局选择
        layout_frame = tk.LabelFrame(left_panel, text="\u5e03\u5c40\u9009\u62e9", 
                                     bg=self.colors['frame'], padx=8, pady=8)
        layout_frame.pack(fill=tk.X, padx=5, pady=3)
        
        self.layout_var = tk.StringVar(value="ch_first")
        tk.Radiobutton(layout_frame, text="\u8df3\u8dc3SLC\uff08CH-first\uff09", 
                       variable=self.layout_var, value="ch_first",
                       bg=self.colors['frame']).pack(anchor=tk.W)
        tk.Radiobutton(layout_frame, text="\u9ed8\u8ba4SLC\uff08PL-first\uff09", 
                       variable=self.layout_var, value="pl_first",
                       bg=self.colors['frame']).pack(anchor=tk.W)
        tk.Radiobutton(layout_frame, text="TLC", 
                       variable=self.layout_var, value="tlc",
                       bg=self.colors['frame']).pack(anchor=tk.W)
        
        # 预取选项
        prefetch_frame = tk.LabelFrame(left_panel, text="\u9884\u53d6\u9009\u9879", 
                                       bg=self.colors['frame'], padx=8, pady=8)
        prefetch_frame.pack(fill=tk.X, padx=5, pady=3)
        
        self.intra_var = tk.BooleanVar(value=True)
        self.inter_var = tk.BooleanVar(value=True)
        tk.Checkbutton(prefetch_frame, text="\u542f\u7528\u4e13\u5bb6\u5185\u9884\u53d6", 
                       variable=self.intra_var, bg=self.colors['frame']).pack(anchor=tk.W)
        tk.Checkbutton(prefetch_frame, text="\u542f\u7528\u4e13\u5bb6\u95f4\u9884\u53d6", 
                       variable=self.inter_var, bg=self.colors['frame']).pack(anchor=tk.W)
        
        # 运行按钮
        self.run_btn = tk.Button(left_panel, text="\u8fd0\u884c\u6a21\u62df", 
                                  command=self.run_simulation,
                                  bg=self.colors['success'], fg='white',
                                  font=('Microsoft YaHei', 11, 'bold'),
                                  height=1)
        self.run_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 显示布局图选项
        self.show_plot_var = tk.BooleanVar(value=False)
        if matplotlib_available:
            tk.Checkbutton(left_panel, text="\u8fd0\u884c\u540e\u663e\u793a\u5e03\u5c40\u56fe", 
                          variable=self.show_plot_var, 
                          bg=self.colors['frame']).pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # 右侧：输出显示面板
        right_panel = tk.Frame(main_container, bg=self.colors['frame'], bd=2, relief=tk.GROOVE)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 输出文本区域
        tk.Label(right_panel, text="\u6a21\u62df\u7ed3\u679c", 
                 bg=self.colors['frame'], font=('Microsoft YaHei', 10, 'bold')).pack(pady=5)
        
        self.output_text = scrolledtext.ScrolledText(
            right_panel, wrap=tk.NONE, 
            font=('Consolas', 10),
            width=100, height=35
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def _init_topk_tab(self):
        """初始化专家命中仿真标签页"""
        # 主容器
        main_container = tk.Frame(self.topk_frame, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：带滚动条的参数输入面板
        left_panel, _ = self._create_scrollable_left_panel(main_container)
        
        # 硬件配置参数（可编辑，与主标签页共享变量）
        hw_config_frame = tk.LabelFrame(left_panel, text="\u786c\u4ef6\u914d\u7f6e\u53c2\u6570", 
                                         bg=self.colors['frame'], padx=8, pady=8)
        hw_config_frame.pack(fill=tk.X, padx=5, pady=(5, 3))
        
        # 加载预设按钮
        tk.Button(hw_config_frame, text="\u52a0\u8f7d\u9884\u8bbe\u914d\u7f6e", 
                  command=self._show_preset_menu,
                  bg=self.colors['accent'], fg='white',
                  font=('Microsoft YaHei', 9)).pack(fill=tk.X, pady=(0, 8))
        
        # Channels
        tk.Label(hw_config_frame, text="\u901a\u9053\u6570 (Channels):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        tk.Spinbox(hw_config_frame, from_=1, to=64, textvariable=self.channels_var, 
                   width=15).pack(fill=tk.X, pady=(0, 5))
        
        # Planes
        tk.Label(hw_config_frame, text="\u6bcf\u901a\u9053 Plane \u6570:", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        tk.Spinbox(hw_config_frame, from_=1, to=16, textvariable=self.planes_var, 
                   width=15).pack(fill=tk.X, pady=(0, 5))
        
        # Page Size
        tk.Label(hw_config_frame, text="Page Size (KB):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        tk.Spinbox(hw_config_frame, from_=1, to=64, textvariable=self.page_size_var, 
                   width=15).pack(fill=tk.X, pady=(0, 5))
        
        # 硬件性能参数（可编辑，与主标签页共享变量）
        hw_perf_frame = tk.LabelFrame(left_panel, text="\u786c\u4ef6\u6027\u80fd\u53c2\u6570", 
                                       bg=self.colors['frame'], padx=8, pady=8)
        hw_perf_frame.pack(fill=tk.X, padx=5, pady=3)
        
        # 单通道带宽
        tk.Label(hw_perf_frame, text="\u5355\u901a\u9053\u5e26\u5bbd (GB/s):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        tk.Entry(hw_perf_frame, textvariable=self.bw_var, width=17).pack(fill=tk.X, pady=(0, 5))
        
        # tR
        tk.Label(hw_perf_frame, text="tR \u8bfb\u53d6\u5ef6\u8fdf (\u03bcs):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        tk.Entry(hw_perf_frame, textvariable=self.tr_var, width=17).pack(fill=tk.X, pady=(0, 5))
        
        # MoE 模型参数
        moe_frame = tk.LabelFrame(left_panel, text="MoE \u6a21\u578b\u53c2\u6570", 
                                  bg=self.colors['frame'], padx=8, pady=8)
        moe_frame.pack(fill=tk.X, padx=5, pady=3)
        
        # Expert 总数
        tk.Label(moe_frame, text="Expert \u603b\u6570:", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.topk_num_experts_var = tk.IntVar(value=512)
        tk.Spinbox(moe_frame, from_=16, to=2048, textvariable=self.topk_num_experts_var, 
                   width=15).pack(fill=tk.X, pady=(0, 5))
        
        # Gate 参数大小
        tk.Label(moe_frame, text="Gate \u53c2\u6570\u5927\u5c0f (bytes):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.topk_gate_bytes_var = tk.IntVar(value=294912)
        tk.Entry(moe_frame, textvariable=self.topk_gate_bytes_var, width=17).pack(fill=tk.X, pady=(0, 5))
        
        # Up 参数大小
        tk.Label(moe_frame, text="Up \u53c2\u6570\u5927\u5c0f (bytes):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.topk_up_bytes_var = tk.IntVar(value=294912)
        tk.Entry(moe_frame, textvariable=self.topk_up_bytes_var, width=17).pack(fill=tk.X, pady=(0, 5))
        
        # Down 参数大小
        tk.Label(moe_frame, text="Down \u53c2\u6570\u5927\u5c0f (bytes):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.topk_down_bytes_var = tk.IntVar(value=294912)
        tk.Entry(moe_frame, textvariable=self.topk_down_bytes_var, width=17).pack(fill=tk.X, pady=(0, 5))
        
        # 分析参数
        params_frame = tk.LabelFrame(left_panel, text="\u5206\u6790\u53c2\u6570", 
                                     bg=self.colors['frame'], padx=8, pady=8)
        params_frame.pack(fill=tk.X, padx=5, pady=3)
        
        # TopK
        tk.Label(params_frame, text="TopK (\u6bcf\u6b21\u9009\u62e9 Expert \u6570):", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.topk_k_var = tk.IntVar(value=10)
        tk.Spinbox(params_frame, from_=1, to=50, textvariable=self.topk_k_var, 
                   width=15).pack(fill=tk.X, pady=(0, 5))
        
        # 实验次数
        tk.Label(params_frame, text="\u5b9e\u9a8c\u6b21\u6570:", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.n_trials_var = tk.IntVar(value=50)
        tk.Spinbox(params_frame, from_=10, to=1000, textvariable=self.n_trials_var, 
                   width=15).pack(fill=tk.X, pady=(0, 5))
        
        # 布局选择
        tk.Label(params_frame, text="\u5e03\u5c40\u9009\u62e9:", 
                 bg=self.colors['frame']).pack(anchor=tk.W)
        self.topk_layout_var = tk.StringVar(value="ch_first")
        tk.Radiobutton(params_frame, text="\u8df3\u8dc3SLC\uff08CH-first\uff09", 
                       variable=self.topk_layout_var, value="ch_first",
                       bg=self.colors['frame']).pack(anchor=tk.W)
        tk.Radiobutton(params_frame, text="\u9ed8\u8ba4SLC\uff08PL-first\uff09", 
                       variable=self.topk_layout_var, value="pl_first",
                       bg=self.colors['frame']).pack(anchor=tk.W)
        tk.Radiobutton(params_frame, text="TLC", 
                       variable=self.topk_layout_var, value="tlc",
                       bg=self.colors['frame']).pack(anchor=tk.W)
        
        # 运行按钮
        self.topk_run_btn = tk.Button(left_panel, text="\u8fd0\u884c\u4e13\u5bb6\u547d\u4e2d\u4eff\u771f", 
                                       command=self.run_topk_analysis,
                                       bg=self.colors['accent'], fg='white',
                                       font=('Microsoft YaHei', 11, 'bold'),
                                       height=1)
        self.topk_run_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 显示图表选项
        self.topk_show_plot_var = tk.BooleanVar(value=True)
        if matplotlib_available:
            tk.Checkbutton(left_panel, text="\u663e\u793a\u5206\u6790\u56fe\u8868", 
                          variable=self.topk_show_plot_var, 
                          bg=self.colors['frame']).pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # 右侧：结果显示面板
        right_panel = tk.Frame(main_container, bg=self.colors['frame'], bd=2, relief=tk.GROOVE)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 输出文本区域
        tk.Label(right_panel, text="\u4e13\u5bb6\u547d\u4e2d\u8bfbNAND\u884c\u4e3a\u5206\u6790", 
                 bg=self.colors['frame'], font=('Microsoft YaHei', 10, 'bold')).pack(pady=5)
        
        self.topk_output_text = scrolledtext.ScrolledText(
            right_panel, wrap=tk.NONE, 
            font=('Consolas', 10),
            width=120, height=40
        )
        self.topk_output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def _show_preset_menu(self):
        """显示预设菜单"""
        menu = tk.Menu(self.root, tearoff=0)
        for name in self.presets:
            menu.add_command(label=name, command=lambda n=name: self._load_preset(n))
        menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())
    
    def _load_preset(self, preset_name):
        """加载预设配置"""
        preset = self.presets[preset_name]
        self.channels_var.set(preset['channels'])
        self.planes_var.set(preset['planes'])
        self.page_size_var.set(preset['page_size'])
        self.bw_var.set(preset['bw'])
        self.tr_var.set(preset['tr'])
        
        # 更新输出
        self.output_text.insert(tk.END, f"\n[\u9884\u8bbe] \u5df2\u52a0\u8f7d: {preset_name}\n")
        self.output_text.insert(tk.END, f"  Channels: {preset['channels']}, Planes: {preset['planes']}\n")
        self.output_text.insert(tk.END, f"  Page Size: {preset['page_size']} KB\n")
        self.output_text.insert(tk.END, f"  BW: {preset['bw']} GB/s, tR: {preset['tr']} us\n")
        self.output_text.see(tk.END)
    
    def _create_geometry(self):
        """创建 NAND 几何结构"""
        return NandGeometry(
            channels=self.channels_var.get(),
            planes_per_channel=self.planes_var.get(),
            page_size_bytes=self.page_size_var.get() * 1024
        )
    
    def _place_experts(self, geo):
        """根据布局选择放置 experts"""
        layout = self.layout_var.get()
        
        if layout == "ch_first":
            return place_experts_page_rr(
                geo,
                num_experts=self.num_experts_var.get(),
                gate_bytes=self.gate_bytes_var.get(),
                up_bytes=self.up_bytes_var.get(),
                down_bytes=self.down_bytes_var.get()
            )
        elif layout == "pl_first":
            return place_experts_page_rr_pl_first(
                geo,
                num_experts=self.num_experts_var.get(),
                gate_bytes=self.gate_bytes_var.get(),
                up_bytes=self.up_bytes_var.get(),
                down_bytes=self.down_bytes_var.get()
            )
        else:  # tlc
            return place_experts_tlc(
                geo,
                num_experts=self.num_experts_var.get(),
                gate_bytes=self.gate_bytes_var.get(),
                up_bytes=self.up_bytes_var.get(),
                down_bytes=self.down_bytes_var.get()
            )
    
    def run_simulation(self):
        """运行模拟（在后台线程中）"""
        if self.is_running:
            return
        self.is_running = True
        self.run_btn.config(state=tk.DISABLED, text="\u6a21\u62df\u8fd0\u884c\u4e2d...")
        self.output_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self._do_simulation, daemon=True)
        thread.start()
    
    def _do_simulation(self):
        """实际模拟逻辑"""
        try:
            # 捕获输出
            old_stdout = sys.stdout
            sys.stdout = RedirectText(self.output_text)
            
            # 创建几何结构和布局
            geo = self._create_geometry()
            sim = self._place_experts(geo)
            
            # 解析 expert IDs（支持离散和连续格式，如 "0,1,2" 或 "0-9" 或 "0-3,5,7-9"）
            expert_ids_str = self.expert_ids_var.get()
            expert_ids = parse_expert_ids(expert_ids_str)
            
            # 获取参数
            bw = self.bw_var.get() * 1e9  # 转换为 B/s
            tR = self.tr_var.get() * 1e-6  # 转换为秒
            intra = self.intra_var.get()
            inter = self.inter_var.get()
            
            # 打印参数信息
            print("=" * 80)
            print("NAND MoE Simulator - \u6a21\u62df\u53c2\u6570")
            print("=" * 80)
            print(f"\u786c\u4ef6\u914d\u7f6e:")
            print(f"  Channels: {geo.channels}, Planes/Channel: {geo.planes_per_channel}")
            print(f"  Page Size: {geo.page_size_bytes / 1024:.0f} KB")
            print(f"  \u603b\u5e26\u5bbd: {bw * geo.channels / 1e9:.2f} GB/s ({bw/1e9:.2f} GB/s \u00d7 {geo.channels} CH)")
            print(f"  tR: {self.tr_var.get()} us")
            print(f"\nMoE \u914d\u7f6e:")
            print(f"  Expert \u603b\u6570: {self.num_experts_var.get()}")
            print(f"  Gate: {self.gate_bytes_var.get()} bytes")
            print(f"  Up/Down: {self.up_bytes_var.get()} bytes")
            print(f"\nTopK \u914d\u7f6e:")
            print(f"  TopK: {self.topk_var.get()}")
            print(f"  Expert IDs: {expert_ids}")
            print(f"\n\u9884\u53d6\u914d\u7f6e:")
            print(f"  Intra-Expert: {intra}, Inter-Expert: {inter}")
            print(f"  \u5e03\u5c40: {self.layout_var.get()}")
            print("=" * 80)
            print()
            
            # 运行模拟
            from nand import print_sequential_latency_table
            print_sequential_latency_table(
                sim,
                expert_ids,
                bw_total_Bps=bw,
                tR_sec=tR,
                intra_expert_cache=intra,
                inter_expert_cache=inter
            )
            
            # 计算利用率
            result = estimate_sequential_latency(
                sim, expert_ids, bw_total_Bps=bw, tR_sec=tR,
                intra_expert_cache=intra, inter_expert_cache=inter
            )
            eff_bw = result['total_bytes'] / result['total_time_sec'] / 1e9
            total_bw = bw * geo.channels / 1e9
            utilization = (eff_bw / total_bw) * 100
            
            print(f"\n[\u5e26\u5bbd\u5229\u7528\u7387]")
            print(f"  \u603b\u7406\u8bba\u5e26\u5bbd: {total_bw:.3f} GB/s")
            print(f"  \u5b9e\u9645\u6709\u6548\u5e26\u5bbd: {eff_bw:.3f} GB/s")
            print(f"  \u5229\u7528\u7387: {utilization:.2f}%")
            
            # 如果需要显示布局图
            if self.show_plot_var.get() and matplotlib_available:
                self.root.after(100, self._show_layout_in_main_thread, sim)
            
        except Exception as e:
            print(f"\n\u9519\u8bef: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            self.root.after(0, lambda: self.run_btn.config(
                state=tk.NORMAL, text="\u8fd0\u884c\u6a21\u62df"
            ))
    
    def _show_layout_in_main_thread(self, sim):
        """在主线程中显示布局图"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            from matplotlib.patches import Rectangle
            from collections import defaultdict
            
            # 解析 expert IDs
            expert_ids_str = self.expert_ids_var.get()
            expert_ids = [int(x.strip()) for x in expert_ids_str.split(',') if x.strip()]
            
            part_order = ["gate", "up", "down"]
            title = f"Expert Layout ({self.layout_var.get()})"
            max_pages = 64  # 限制显示页数，避免图表过大
            
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
            need_pages = max_page_seen + 1
            max_pages = max(1, min(max_pages, need_pages)) if max_pages else need_pages
            
            part_colors = {
                "gate": mpl.colors.to_rgba("#1f77b4"),
                "up":   mpl.colors.to_rgba("#ff7f0e"),
                "down": mpl.colors.to_rgba("#2ca02c"),
            }
            annotate_fontsize = max(5, min(8, int(60 / max(max_pages, P))))
            
            # 限制图表默认大小，使其适应屏幕，同时支持缩放
            base_width = min(2.0, max(1.2, P * 0.25))  # 每Channel宽度
            base_height = min(0.25, max(0.15, 8.0 / max_pages)) if max_pages > 0 else 0.2
            figsize = (
                min(16, max(8, C * base_width)),      # 总宽度限制在 8-16 英寸
                min(10, max(6, max_pages * base_height))  # 总高度限制在 6-10 英寸
            )
            
            fig, axes = plt.subplots(1, C, figsize=figsize, sharey=True, squeeze=False)
            fig.canvas.manager.set_window_title('Expert Layout (支持鼠标滚轮缩放)')
            axes = axes[0]
            
            for ch in range(C):
                ax = axes[ch]
                for pl in range(P):
                    for pg in range(max_pages):
                        ax.add_patch(Rectangle((pl, pg), 1, 1,
                            linewidth=0.4, edgecolor="#aaaaaa", facecolor="white", zorder=0))
                
                cell = defaultdict(list)
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
            plt.show(block=False)  # 非阻塞模式
        except Exception as e:
            print(f"\n\u56fe\u8868\u663e\u793a\u9519\u8bef: {e}")
            import traceback
            traceback.print_exc()
    
    def run_topk_analysis(self):
        """运行专家命中仿真"""
        if self.is_running:
            return
        self.is_running = True
        self.topk_run_btn.config(state=tk.DISABLED, text="\u5206\u6790\u4e2d...")
        self.topk_output_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self._do_topk_analysis, daemon=True)
        thread.start()
    
    def _do_topk_analysis(self):
        """实际专家命中仿真逻辑"""
        try:
            # 捕获输出
            old_stdout = sys.stdout
            sys.stdout = RedirectText(self.topk_output_text)
            
            # 创建几何结构和布局
            geo = self._create_geometry()
            
            # 使用 TopK 标签页的参数
            layout = self.topk_layout_var.get()
            num_experts = self.topk_num_experts_var.get()
            topk = self.topk_k_var.get()
            n_trials = self.n_trials_var.get()
            # MoE 模型参数
            gate_bytes = self.topk_gate_bytes_var.get()
            up_bytes = self.topk_up_bytes_var.get()
            down_bytes = self.topk_down_bytes_var.get()
            
            if layout == "ch_first":
                sim = place_experts_page_rr(geo, num_experts, gate_bytes, up_bytes, down_bytes)
            elif layout == "pl_first":
                sim = place_experts_page_rr_pl_first(geo, num_experts, gate_bytes, up_bytes, down_bytes)
            else:  # tlc
                sim = place_experts_tlc(geo, num_experts, gate_bytes, up_bytes, down_bytes)
            
            # 硬件参数
            bw = self.bw_var.get() * 1e9  # 单通道
            tR = self.tr_var.get() * 1e-6
            
            # 打印参数信息
            # 参数标题已删除
            print(f"\u786c\u4ef6\u914d\u7f6e:")
            print(f"  Channels: {geo.channels}, Planes/Channel: {geo.planes_per_channel}")
            print(f"  Page Size: {geo.page_size_bytes / 1024:.0f} KB")
            print(f"  \u5355\u901a\u9053\u5e26\u5bbd: {bw/1e9:.2f} GB/s")
            print(f"  \u603b\u5e26\u5bbd: {bw * geo.channels / 1e9:.2f} GB/s")
            print(f"  tR: {self.tr_var.get()} us")
            print(f"\nMoE \u6a21\u578b\u53c2\u6570:")
            print(f"  Expert \u603b\u6570: {num_experts}")
            print(f"  Gate: {gate_bytes} bytes")
            print(f"  Up: {up_bytes} bytes")
            print(f"  Down: {down_bytes} bytes")
            print(f"\n\u5206\u6790\u53c2\u6570:")
            print(f"  TopK: {topk}")
            print(f"  \u5b9e\u9a8c\u6b21\u6570: {n_trials}")
            print(f"  \u5e03\u5c40: {layout}")
            print("=" * 110)
            print()
            
            # 运行专家命中仿真
            comparison_text, bw_analysis_text, compare_results = run_topk_analysis(
                sim,
                bw_total_Bps=bw,
                tR_sec=tR,
                num_experts=num_experts,
                topk=topk,
                n_trials=n_trials
            )
            
            # 输出结果
            print(comparison_text)
            print("\n")
            print(bw_analysis_text)
            
            # 保存图表结果供后续显示
            self.last_compare_results = compare_results
            
            # 如果需要显示图表
            if self.topk_show_plot_var.get() and matplotlib_available:
                self.root.after(100, self._show_topk_plot_in_main_thread, compare_results, bw, sim.geo.channels)
            
        except Exception as e:
            print(f"\n\u9519\u8bef: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            self.root.after(0, lambda: self.topk_run_btn.config(
                state=tk.NORMAL, text="\u8fd0\u884c\u4e13\u5bb6\u547d\u4e2d\u4eff\u771f"
            ))
    
    def _show_topk_plot_in_main_thread(self, compare_results, bw, channels):
        """在主线程中显示专家命中仿真图表"""
        try:
            plot_prefetch_comparison(compare_results, bw_total_Bps=bw, channels=channels)
        except Exception as e:
            print(f"\n\u56fe\u8868\u663e\u793a\u9519\u8bef: {e}")


def main():
    root = tk.Tk()
    app = NandSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
