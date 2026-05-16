function dataOut = IIRFilterECG(dataIn)
% 使用IIR滤波器进行滤波
%   输入参数dataIn为滤波前的数据
%   输出参数dataOut为滤波后的数据
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

degree = 4; % 滤波器阶数
Fs = 500; % 采样频率
T = 1 / Fs; % 采样间隔
Fc = 50; % 截止频率
Wc = Fc / (Fs / 2); % 归一化截止频率
[b, a] = butter(degree, Wc); % 设计低通滤波器
dataOut = filtfilt(b, a, dataIn); % 滤波

plot(t, dataIn); % 绘制原始波形
title('心电信号原始波形'); % 标注标题
xlabel('时间(s)'); % 标注X坐标
ylabel('幅值'); % 标注Y坐标

subplot(2, 1, 2); % 将figure划分为2*1，在第2块创建坐标系
plot(t, dataOut); % 绘制滤波后波形
title('IIR滤波器滤波后波形'); % 标注标题
xlabel('时间(s)'); % 标注X坐标
ylabel('幅值'); % 标注Y坐标
