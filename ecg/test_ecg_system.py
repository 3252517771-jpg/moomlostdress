import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd

def test_filter_functions():
    """测试滤波器功能"""
    # 加载示例数据
    file_path = "c:\\Users\\32525\\Desktop\\code\\ecg\\实验八-资料\\实验八-资料\\心电信号处理实验\\心电0x30演示数据-01.csv"
    
    try:
        # 读取数据
        df = pd.read_csv(file_path, header=None)
        
        # 如果数据包含两列（索引和值），只取第二列
        if df.shape[1] > 1:
            ecg_data = df.iloc[:, 1].values
        else:
            ecg_data = df.iloc[:, 0].values
        
        fs = 500
        t = np.arange(len(ecg_data)) / fs
        
        print(f"加载数据成功，共 {len(ecg_data)} 个采样点")
        
        # 1. 测试陷波滤波器
        print("\n1. 测试陷波滤波器...")
        b, a = signal.iirnotch(50, 30, fs)
        notch_filtered = signal.filtfilt(b, a, ecg_data)
        
        # 2. 测试IIR低通滤波器
        print("2. 测试IIR低通滤波器...")
        b, a = signal.butter(4, 50, btype='low', fs=fs)
        iir_filtered = signal.filtfilt(b, a, ecg_data)
        
        # 3. 测试FIR低通滤波器
        print("3. 测试FIR低通滤波器...")
        b = signal.firwin(5, 50, fs=fs, pass_zero='lowpass')
        fir_filtered = signal.filtfilt(b, 1, ecg_data)
        
        # 4. 测试QRS波提取
        print("4. 测试QRS波提取...")
        peaks, _ = signal.find_peaks(iir_filtered, height=np.max(iir_filtered)-200, distance=250)
        
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs
            heart_rate = 60 / np.median(rr_intervals)
            print(f"   检测到 {len(peaks)} 个QRS波，心率: {int(heart_rate)} bpm")
        else:
            print("   未检测到足够的QRS波")
        
        # 5. 绘制结果
        print("5. 绘制测试结果...")
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        # 原始信号
        axes[0, 0].plot(t, ecg_data)
        axes[0, 0].set_title("原始心电信号")
        axes[0, 0].set_xlabel("时间 (s)")
        axes[0, 0].set_ylabel("幅值")
        
        # 陷波滤波
        axes[0, 1].plot(t, notch_filtered)
        axes[0, 1].set_title("陷波滤波后信号")
        axes[0, 1].set_xlabel("时间 (s)")
        axes[0, 1].set_ylabel("幅值")
        
        # IIR滤波
        axes[1, 0].plot(t, iir_filtered)
        axes[1, 0].set_title("IIR低通滤波后信号")
        axes[1, 0].set_xlabel("时间 (s)")
        axes[1, 0].set_ylabel("幅值")
        
        # FIR滤波
        axes[1, 1].plot(t, fir_filtered)
        axes[1, 1].set_title("FIR低通滤波后信号")
        axes[1, 1].set_xlabel("时间 (s)")
        axes[1, 1].set_ylabel("幅值")
        
        # QRS检测
        axes[2, 0].plot(t, iir_filtered)
        axes[2, 0].plot(t[peaks], iir_filtered[peaks], 'ro', label="QRS波")
        axes[2, 0].set_title("QRS波检测结果")
        axes[2, 0].set_xlabel("时间 (s)")
        axes[2, 0].set_ylabel("幅值")
        axes[2, 0].legend()
        
        # 频谱分析
        n = len(iir_filtered)
        fft_result = np.fft.fft(iir_filtered - np.mean(iir_filtered))
        freq = np.fft.fftfreq(n, 1/fs)
        amp = np.abs(fft_result) / np.max(np.abs(fft_result))
        
        pos_freq = freq[:n//2]
        pos_amp = amp[:n//2]
        
        axes[2, 1].plot(pos_freq, pos_amp)
        axes[2, 1].set_title("滤波后信号频谱")
        axes[2, 1].set_xlabel("频率 (Hz)")
        axes[2, 1].set_ylabel("归一化幅值")
        axes[2, 1].set_xlim([0, 100])
        
        plt.tight_layout()
        plt.savefig("ecg_test_results.png")
        print("测试结果已保存到 ecg_test_results.png")
        
        return True
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试心电信号处理系统功能...")
    success = test_filter_functions()
    if success:
        print("\n测试完成，所有功能正常工作!")
    else:
        print("\n测试失败，请检查系统配置!")