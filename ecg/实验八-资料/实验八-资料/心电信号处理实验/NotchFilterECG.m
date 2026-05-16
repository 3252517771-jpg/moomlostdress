function dataOut = NotchFilterECG(dataIn)
% 设计陷波器并去除心电工频干扰
%   输入参数dataIn为原始心电波形
%   输出参数dataOut为陷波器处理后的波形
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

Fs = 500;    % 采样频率
T = 1 / Fs; % 采样间隔
Fc = 50;     % 陷波频率

alpha = -2 * cos(2 * pi * Fc * T);         
beta  = 0.96;
b = [1, alpha, 1];
a = [1, alpha * beta, beta^2];

dataOut = dlsim(b, a, dataIn);

L = length(dataIn); % 数据长度 
t = (1 : L) * T; % 时间横坐标

figure; % 创建窗口
set(gcf, 'name', '去除心电信号中的工频干扰'); % 设置窗口的标题名
subplot(2, 1, 1); % 将figure划分为2*1，在第1块创建坐标系
plot(t, dataIn); % 绘制原始波形
title('心电信号原始波形'); % 标注标题
xlabel('时间(s)'); % 标注X坐标
ylabel('幅值'); % 标注Y坐标

subplot(2, 1, 2); % 将figure划分为2*1，在第2块创建坐标系
plot(t, dataOut); % 绘制滤波后波形
title('去除工频干扰后的波形'); % 标注标题
xlabel('时间(s)'); % 标注X坐标
ylabel('幅值'); % 标注Y坐标