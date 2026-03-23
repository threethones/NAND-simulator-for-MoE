"""
NAND Flash MoE Simulator - GUI 版本
使用 tkinter 构建图形界面
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# 导入核心仿真模块
from nand import (
    NandGeometry, place_experts_page_rr, place_experts_page_rr_pl_first,
    visualize_layout, print_sequential_latency_table, parse_expert_ids
)


class RedirectText(io.StringIO):
    """重定向 stdout/stderr 到 Tkinter 文本框"""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        
    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update()


class NandSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NAND Flash MoE Simulator")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # 创建主框架
        self.create_widgets()
        
    def create_widgets(self):
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置 grid 权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(3, weight=1)
        
        # ========== 硬件几何参数 ==========
        hw_frame = ttk.LabelFrame(main_frame, text="硬件几何参数 (NAND Geometry)", padding="5")
        hw_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        hw_frame.columnconfigure(1, weight=1)
        
        # Channels
        ttk.Label(hw_frame, text="通道数 (Channels):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.channels_var = tk.IntVar(value=8)
        ttk.Spinbox(hw_frame, from_=1, to=32, textvariable=self.channels_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Planes
        ttk.Label(hw_frame, text="平面数/通道 (Planes):").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.planes_var = tk.IntVar(value=8)
        ttk.Spinbox(hw_frame, from_=1, to=16, textvariable=self.planes_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Page Size
        ttk.Label(hw_frame, text="页大小 (Bytes):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.page_size_var = tk.IntVar(value=16384)
        page_sizes = [4096, 8192, 16384, 32768]
        ttk.Combobox(hw_frame, textvariable=self.page_size_var, values=page_sizes, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # ========== 性能参数 ==========
        perf_frame = ttk.LabelFrame(main_frame, text="性能参数 (Performance)", padding="5")
        perf_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        perf_frame.columnconfigure(1, weight=1)
        
        # Bandwidth (单通道)
        ttk.Label(perf_frame, text="单通道带宽 (GB/s):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.bw_var = tk.DoubleVar(value=3.75)
        ttk.Entry(perf_frame, textvariable=self.bw_var, width=12).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # tR
        ttk.Label(perf_frame, text="读延迟 tR (us):").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.tr_var = tk.DoubleVar(value=22.0)
        ttk.Entry(perf_frame, textvariable=self.tr_var, width=12).grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # ========== 专家参数 ==========
        expert_frame = ttk.LabelFrame(main_frame, text="专家参数 (Expert)", padding="5")
        expert_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        expert_frame.columnconfigure(1, weight=1)
        expert_frame.columnconfigure(3, weight=1)
        
        # Expert IDs
        ttk.Label(expert_frame, text="专家 IDs:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.experts_var = tk.StringVar(value="0,1,2")
        ttk.Entry(expert_frame, textvariable=self.experts_var, width=20).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(expert_frame, text="(如: 0,1,2 或 0-5)").grid(row=0, column=2, sticky=tk.W)
        
        # Num Experts
        ttk.Label(expert_frame, text="总专家数:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.num_experts_var = tk.IntVar(value=10)
        ttk.Spinbox(expert_frame, from_=1, to=100, textvariable=self.num_experts_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Expert sizes
        ttk.Label(expert_frame, text="Gate (Bytes):").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.gate_bytes_var = tk.IntVar(value=294912)
        ttk.Entry(expert_frame, textvariable=self.gate_bytes_var, width=12).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(expert_frame, text="Up (Bytes):").grid(row=2, column=2, sticky=tk.W, padx=5)
        self.up_bytes_var = tk.IntVar(value=294912)
        ttk.Entry(expert_frame, textvariable=self.up_bytes_var, width=12).grid(row=2, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(expert_frame, text="Down (Bytes):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.down_bytes_var = tk.IntVar(value=294912)
        ttk.Entry(expert_frame, textvariable=self.down_bytes_var, width=12).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # ========== 布局选项 ==========
        layout_frame = ttk.LabelFrame(main_frame, text="布局选项 (Layout)", padding="5")
        layout_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=(0, 5))
        
        self.layout_var = tk.StringVar(value="ch-first")
        ttk.Radiobutton(layout_frame, text="CH-first (跳跃SLC)", variable=self.layout_var,
                       value="ch-first").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(layout_frame, text="PL-first (默认SLC)", variable=self.layout_var,
                       value="pl-first").grid(row=1, column=0, sticky=tk.W, padx=5)
        
        # ========== 缓存选项 ==========
        cache_frame = ttk.LabelFrame(main_frame, text="缓存选项 (Cache)", padding="5")
        cache_frame.grid(row=3, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.intra_var = tk.BooleanVar(value=True)
        self.inter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cache_frame, text="Intra-expert 预取", variable=self.intra_var).grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Checkbutton(cache_frame, text="Inter-expert 预取", variable=self.inter_var).grid(row=1, column=0, sticky=tk.W, padx=5)
        
        # ========== 输出选项 ==========
        output_frame = ttk.LabelFrame(main_frame, text="输出选项 (Output)", padding="5")
        output_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        output_frame.columnconfigure(1, weight=1)
        
        self.csv_var = tk.StringVar()
        ttk.Checkbutton(output_frame, text="导出 CSV:", command=self.toggle_csv).grid(row=0, column=0, sticky=tk.W, padx=5)
        self.csv_entry = ttk.Entry(output_frame, textvariable=self.csv_var, width=30, state="disabled")
        self.csv_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(output_frame, text="浏览...", command=self.browse_csv).grid(row=0, column=2, padx=5)
        
        self.viz_var = tk.StringVar()
        ttk.Checkbutton(output_frame, text="保存布局图:", command=self.toggle_viz).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.viz_entry = ttk.Entry(output_frame, textvariable=self.viz_var, width=30, state="disabled")
        self.viz_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(output_frame, text="浏览...", command=self.browse_viz).grid(row=1, column=2, padx=5, pady=5)
        
        # 显示布局图选项
        self.show_viz_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="运行后显示布局图", variable=self.show_viz_var).grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        # ========== 运行按钮 ==========
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="运行仿真", command=self.run_simulation, 
                  width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清除输出", command=self.clear_output, 
                  width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="加载预设", command=self.load_preset, 
                  width=15).pack(side=tk.LEFT, padx=5)
        
        # ========== 结果显示区 ==========
        result_frame = ttk.LabelFrame(main_frame, text="仿真结果", padding="5")
        result_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, 
                                                     width=80, height=20, font=("Consolas", 9))
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def toggle_csv(self):
        if self.csv_entry.cget("state") == "disabled":
            self.csv_entry.config(state="normal")
        else:
            self.csv_entry.config(state="disabled")
            self.csv_var.set("")
    
    def toggle_viz(self):
        if self.viz_entry.cget("state") == "disabled":
            self.viz_entry.config(state="normal")
        else:
            self.viz_entry.config(state="disabled")
            self.viz_var.set("")
    
    def browse_csv(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_var.set(filename)
    
    def browse_viz(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filename:
            self.viz_var.set(filename)
    
    def load_preset(self):
        """加载预设配置"""
        presets = {
            "消费级 SSD": {"channels": 4, "planes": 4, "bw": 1.75, "tr": 50},
            "企业级 SSD": {"channels": 8, "planes": 8, "bw": 1.75, "tr": 22},
            "高性能 SSD": {"channels": 16, "planes": 8, "bw": 1.875, "tr": 20},
        }
        
        preset_window = tk.Toplevel(self.root)
        preset_window.title("选择预设")
        preset_window.geometry("300x150")
        
        ttk.Label(preset_window, text="选择配置预设:").pack(pady=10)
        
        preset_var = tk.StringVar()
        for name in presets:
            ttk.Radiobutton(preset_window, text=name, variable=preset_var, 
                           value=name).pack(anchor=tk.W, padx=20)
        
        def apply():
            if preset_var.get():
                p = presets[preset_var.get()]
                self.channels_var.set(p["channels"])
                self.planes_var.set(p["planes"])
                self.bw_var.set(p["bw"])
                self.tr_var.set(p["tr"])
                self.status_var.set(f"已加载预设: {preset_var.get()}")
            preset_window.destroy()
        
        ttk.Button(preset_window, text="应用", command=apply).pack(pady=10)
    
    def clear_output(self):
        self.result_text.delete(1.0, tk.END)
        self.status_var.set("输出已清除")
    
    def _show_layout_in_main_thread(self):
        """在主线程中显示布局图（避免 matplotlib 多线程警告）"""
        if hasattr(self, '_viz_data'):
            try:
                sim = self._viz_data['sim']
                expert_ids = self._viz_data['expert_ids']
                layout = self._viz_data['layout']
                # 使用 block=False 让 GUI 不会被阻塞，同时关闭交互模式避免警告
                import matplotlib.pyplot as plt
                plt.ioff()  # 关闭交互模式
                visualize_layout(sim, expert_ids=expert_ids, max_pages=20,
                               title=f"Expert Layout ({layout})", block=True)
            except Exception as e:
                print(f"\n[显示布局图失败: {e}]")
            finally:
                delattr(self, '_viz_data')
    
    def run_simulation(self):
        """在后台线程运行仿真"""
        self.status_var.set("正在运行仿真...")
        self.result_text.delete(1.0, tk.END)
        
        # 启动后台线程
        thread = threading.Thread(target=self._do_simulation)
        thread.daemon = True
        thread.start()
    
    def _do_simulation(self):
        """实际执行仿真"""
        try:
            # 获取参数
            channels = self.channels_var.get()
            planes = self.planes_var.get()
            page_size = self.page_size_var.get()
            bw = self.bw_var.get() * 1e9  # GB/s -> B/s
            tr = self.tr_var.get() * 1e-6  # us -> s
            experts_str = self.experts_var.get()
            num_experts = self.num_experts_var.get()
            gate_bytes = self.gate_bytes_var.get()
            up_bytes = self.up_bytes_var.get()
            down_bytes = self.down_bytes_var.get()
            layout = self.layout_var.get()
            intra = self.intra_var.get()
            inter = self.inter_var.get()
            csv_path = self.csv_var.get() if self.csv_var.get() else None
            viz_path = self.viz_var.get() if self.viz_var.get() else None
            show_viz = self.show_viz_var.get()
            
            # 解析 expert_ids
            try:
                expert_ids = parse_expert_ids(experts_str)
            except ValueError as e:
                self.root.after(0, lambda: messagebox.showerror("参数错误", f"专家ID格式错误: {e}"))
                self.root.after(0, lambda: self.status_var.set("参数错误"))
                return
            
            # 验证参数
            if channels <= 0 or planes <= 0 or page_size <= 0:
                self.root.after(0, lambda: messagebox.showerror("参数错误", "channels, planes, page-size 必须大于0"))
                self.root.after(0, lambda: self.status_var.set("参数错误"))
                return
            
            if bw <= 0 or tr <= 0:
                self.root.after(0, lambda: messagebox.showerror("参数错误", "bw 和 tr 必须大于0"))
                self.root.after(0, lambda: self.status_var.set("参数错误"))
                return
            
            # 重定向输出
            redirect = RedirectText(self.result_text)
            
            with redirect_stdout(redirect), redirect_stderr(redirect):
                # 创建几何配置
                geo = NandGeometry(
                    channels=channels,
                    planes_per_channel=planes,
                    page_size_bytes=page_size
                )
                
                print(f"\n{'='*60}")
                print("NAND Flash MoE Simulator")
                print(f"{'='*60}")
                total_bw = bw * channels
                print(f"硬件配置: {channels}通道 x {planes}平面, 页大小={page_size}字节")
                print(f"性能参数: 单通道={bw/1e9:.2f}GB/s, 总带宽={total_bw/1e9:.1f}GB/s, tR={tr*1e6:.1f}us")
                print(f"布局方式: {layout}")
                print(f"缓存选项: intra={'ON' if intra else 'OFF'}, inter={'ON' if inter else 'OFF'}")
                print(f"模拟专家: {expert_ids}")
                print(f"{'='*60}\n")
                
                # 选择布局函数
                if layout == 'pl-first':
                    sim = place_experts_page_rr_pl_first(
                        geo, num_experts, gate_bytes, up_bytes, down_bytes
                    )
                else:
                    sim = place_experts_page_rr(
                        geo, num_experts, gate_bytes, up_bytes, down_bytes
                    )
                
                # 可视化
                if viz_path:
                    try:
                        visualize_layout(sim, expert_ids=expert_ids, max_pages=20,
                                       title=f"Expert Layout ({layout})", save_path=viz_path)
                        print(f"\n[布局图已保存] {viz_path}")
                    except Exception as e:
                        print(f"\n[可视化失败: {e}]")
                
                # 显示布局图（如果用户选择）- 延迟到主线程执行
                if show_viz:
                    print(f"\n[正在准备布局图...]")
                    # 保存 sim 和参数供后续使用
                    self._viz_data = {
                        'sim': sim,
                        'expert_ids': expert_ids,
                        'layout': layout
                    }
                    # 使用 after 在主线程中延迟显示
                    self.root.after(100, self._show_layout_in_main_thread)
                
                # 顺序延迟仿真
                result = print_sequential_latency_table(
                    sim, expert_ids,
                    bw_total_Bps=bw, tR_sec=tr,
                    intra_expert_cache=intra,
                    inter_expert_cache=inter,
                    csv_path=csv_path
                )
                
                if csv_path:
                    print(f"\n[CSV已保存] {csv_path}")
                
                # 输出有效带宽
                eff_bw = result['effective_bw_Bps']
                total_bw = bw * channels  # 总带宽 = 单通道 × 通道数
                utilization = (eff_bw / total_bw) * 100 if total_bw > 0 else 0
                print(f"\n{'='*60}")
                print(f"有效带宽: {eff_bw/1e9:.3f} GB/s (利用率: {utilization:.1f}%)")
                print(f"{'='*60}\n")
            
            self.root.after(0, lambda: self.status_var.set(f"仿真完成 - 带宽: {eff_bw/1e9:.2f} GB/s"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"仿真失败: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("仿真失败"))


def main_gui():
    """GUI 入口"""
    root = tk.Tk()
    app = NandSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main_gui()
