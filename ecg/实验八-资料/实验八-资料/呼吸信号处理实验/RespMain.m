% 呼吸信号处理实验脚本文件
%   对原始的呼吸信号进行平滑滤波，然后，设计IIR滤波器和FIR滤波器并对原始的呼吸信号进行滤波，
%   最后，计算呼吸率以及呼吸信号的幅度谱
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

%rawData = xlsread('呼吸0x31演示数据-01.csv', 'A1 : A10000'); % 读取呼吸波形数据
rawData = xlsread('respWave2022.04.26 14-07-58.csv', 'A1 : A10000'); % 读取呼吸波形数据
SmoothFilterResp(rawData); % 对原始的呼吸信号进行平滑滤波
IIRFilterResp(rawData); % 设计IIR滤波器对原始的呼吸信号进行滤波
FIRFilterResp(rawData); % 设计FIR滤波器对原始的呼吸信号进行滤波
CalcRespRate(rawData); % 计算呼吸率
CalcAmpSpec(rawData); % 计算呼吸信号的幅度谱