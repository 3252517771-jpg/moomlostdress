% 心电信号处理实验脚本文件
%   设计陷波器并去除心电工频干扰，然后，设计IIR滤波器和FIR滤波器并对原始的心电信号进行滤波，
%   最后，计算心率以及心电信号的幅度谱
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

rawData = xlsread('心电0x30演示数据-03.csv', 'A1 : A6000'); % 读取6000个心电数据

NotchFilterECG(Add50HzNoise(rawData)); % 设计陷波器并去除心电工频干扰
IIRFilterECG(rawData); % 设计IIR滤波器对原始的心电信号进行滤波
FIRFilterECG(rawData); % 设计FIR滤波器对原始的心电信号进行滤波

CalcHeartRate(rawData); % 计算心率
CalcAmpSpec(rawData); % 计算心电信号的幅度谱
% CalcAmpSpec(Add50HzNoise(rawData)); % 计算心电信号的幅度谱（含工频干扰的心电信号）