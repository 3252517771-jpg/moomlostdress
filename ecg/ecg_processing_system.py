import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd

class ECGProcessingSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("交互式心电信号处理系统")
        self.root.geometry("1200x800")
        
        # 数据存储
        self.ecg_data = None
        self.fs = 500  # 采样频率
        self.filtered_data = None
        self.qrs_peaks = None
        
        # 创建界面
        self.create_widgets()
    
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # 1. 数据加载区域
        data_frame = ttk.LabelFrame(main_frame, text="数据加载", padding="10")
        data_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), columnspan=2)
        
        ttk.Button(data_frame, text="加载心电数据", command=self.load_data).grid(row=0, column=0, padx=5)
        self.data_label = ttk.Label(data_frame, text="未加载数据")
        self.data_label.grid(row=0, column=1, padx=5)
        
        # 2. 滤波器设置区域
        filter_frame = ttk.LabelFrame(main_frame, text="滤波器设置", padding="10")
        filter_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), columnspan=2)
        
        # 滤波器类型
        ttk.Label(filter_frame, text="滤波器类型:").grid(row=0, column=0, padx=5, pady=5)
        self.filter_type = ttk.Combobox(filter_frame, values=["None", "Notch", "IIR", "FIR"])
        self.filter_type.current(0)
        self.filter_type.grid(row=0, column=1, padx=5, pady=5)
        
        # 截止频率
        ttk.Label(filter_frame, text="截止频率(Hz):").grid(row=0, column=2, padx=5, pady=5)
        self.cutoff_freq = ttk.Entry(filter_frame, width=10)
        self.cutoff_freq.insert(0, "50")
        self.cutoff_freq.grid(row=0, column=3, padx=5, pady=5)
        
        # 滤波器阶数
        ttk.Label(filter_frame, text="阶数:").grid(row=0, column=4, padx=5, pady=5)
        self.filter_order = ttk.Entry(filter_frame, width=10)
        self.filter_order.insert(0, "4")
        self.filter_order.grid(row=0, column=5, padx=5, pady=5)
        
        # 应用按钮
        ttk.Button(filter_frame, text="应用滤波器", command=self.apply_filter).grid(row=0, column=6, padx=5, pady=5)
        
        # 3. 特征提取区域
        feature_frame = ttk.LabelFrame(main_frame, text="特征提取", padding="10")
        feature_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), columnspan=2)
        
        ttk.Button(feature_frame, text="提取QRS波", command=self.extract_qrs).grid(row=0, column=0, padx=5)
        self.hr_label = ttk.Label(feature_frame, text="心率: -- bpm")
        self.hr_label.grid(row=0, column=1, padx=5)
        
        # 4. 频谱分析区域
        spectrum_frame = ttk.LabelFrame(main_frame, text="频谱分析", padding="10")
        spectrum_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), columnspan=2)
        
        ttk.Button(spectrum_frame, text="显示频谱", command=self.show_spectrum).grid(row=0, column=0, padx=5)
        
        # 5. 图形显示区域
        plot_frame = ttk.LabelFrame(main_frame, text="信号显示", padding="10")
        plot_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), columnspan=2)
        
        # 创建图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def load_data(self):
        # 打开文件对话框选择数据文件
        file_path = filedialog.askopenfilename(
            initialdir="c:\\Users\\32525\\Desktop\\code\\ecg\\实验八-资料\\实验八-资料\\心电信号处理实验",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                # 读取数据
                df = pd.read_csv(file_path, header=None)
                
                # 如果数据包含两列（索引和值），只取第二列
                if df.shape[1] > 1:
                    self.ecg_data = df.iloc[:, 1].values
                else:
                    self.ecg_data = df.iloc[:, 0].values
                
                self.data_label.config(text=f"已加载: {len(self.ecg_data)} 个采样点")
                self.filtered_data = self.ecg_data.copy()
                self.plot_signal()
            except Exception as e:
                self.data_label.config(text=f"加载错误: {str(e)}")
    
    def apply_filter(self):
        if self.ecg_data is None:
            return
        
        try:
            filter_type = self.filter_type.get()
            cutoff = float(self.cutoff_freq.get())
            order = int(self.filter_order.get())
            
            if filter_type == "None":
                self.filtered_data = self.ecg_data.copy()
            elif filter_type == "Notch":
                # 50Hz陷波滤波
                b, a = signal.iirnotch(cutoff, 30, self.fs)
                self.filtered_data = signal.filtfilt(b, a, self.ecg_data)
            elif filter_type == "IIR":
                # IIR低通滤波
                b, a = signal.butter(order, cutoff, btype='low', fs=self.fs)
                self.filtered_data = signal.filtfilt(b, a, self.ecg_data)
            elif filter_type == "FIR":
                # FIR低通滤波
                b = signal.firwin(order+1, cutoff, fs=self.fs, pass_zero='lowpass')
                self.filtered_data = signal.filtfilt(b, 1, self.ecg_data)
            
            self.plot_signal()
        except Exception as e:
            print(f"滤波错误: {str(e)}")
    
    def extract_qrs(self):
        if self.filtered_data is None:
            return
        
        try:
            # 使用scipy的find_peaks函数提取QRS波
            self.qrs_peaks, _ = signal.find_peaks(
                self.filtered_data,
                height=np.max(self.filtered_data) - 200,
                distance=250  # 最小峰间距，500Hz采样时对应0.5秒
            )
            
            # 计算心率
            if len(self.qrs_peaks) > 1:
                rr_intervals = np.diff(self.qrs_peaks) / self.fs
                heart_rate = 60 / np.median(rr_intervals)
                self.hr_label.config(text=f"心率: {int(heart_rate)} bpm")
            else:
                self.hr_label.config(text="心率: -- bpm")
            
            self.plot_signal(show_qrs=True)
        except Exception as e:
            print(f"QRS提取错误: {str(e)}")
    
    def show_spectrum(self):
        if self.filtered_data is None:
            return
        
        try:
            # 计算频谱
            n = len(self.filtered_data)
            fft_result = np.fft.fft(self.filtered_data - np.mean(self.filtered_data))
            freq = np.fft.fftfreq(n, 1/self.fs)
            amp = np.abs(fft_result) / np.max(np.abs(fft_result))
            
            # 只显示正频率部分
            pos_freq = freq[:n//2]
            pos_amp = amp[:n//2]
            
            # 绘制频谱
            self.ax1.clear()
            self.ax2.clear()
            
            t = np.arange(len(self.filtered_data)) / self.fs
            self.ax1.plot(t, self.filtered_data)
            self.ax1.set_title("滤波后的心电信号")
            self.ax1.set_xlabel("时间 (s)")
            self.ax1.set_ylabel("幅值")
            
            self.ax2.plot(pos_freq, pos_amp)
            self.ax2.set_title("心电信号频谱")
            self.ax2.set_xlabel("频率 (Hz)")
            self.ax2.set_ylabel("归一化幅值")
            self.ax2.set_xlim([0, 100])  # 只显示0-100Hz
            
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"频谱分析错误: {str(e)}")
    
    def plot_signal(self, show_qrs=False):
        if self.filtered_data is None:
            return
        
        try:
            self.ax1.clear()
            self.ax2.clear()
            
            t = np.arange(len(self.filtered_data)) / self.fs
            
            # 绘制原始信号
            if self.ecg_data is not None:
                self.ax1.plot(t, self.ecg_data, label="原始信号")
            
            # 绘制滤波后信号
            self.ax1.plot(t, self.filtered_data, label="滤波后信号")
            
            # 标记QRS波
            if show_qrs and self.qrs_peaks is not None:
                self.ax1.plot(t[self.qrs_peaks], self.filtered_data[self.qrs_peaks], 'ro', label="QRS波")
            
            self.ax1.set_title("心电信号")
            self.ax1.set_xlabel("时间 (s)")
            self.ax1.set_ylabel("幅值")
            self.ax1.legend()
            
            # 绘制局部放大图（显示第一个QRS波附近）
            if show_qrs and self.qrs_peaks is not None and len(self.qrs_peaks) > 0:
                peak_idx = self.qrs_peaks[0]
                start_idx = max(0, peak_idx - 100)
                end_idx = min(len(self.filtered_data), peak_idx + 100)
                self.ax2.plot(t[start_idx:end_idx], self.filtered_data[start_idx:end_idx])
                self.ax2.plot(t[peak_idx], self.filtered_data[peak_idx], 'ro')
                self.ax2.set_title("QRS波放大图")
                self.ax2.set_xlabel("时间 (s)")
                self.ax2.set_ylabel("幅值")
            
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"绘图错误: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ECGProcessingSystem(root)
    root.mainloop()