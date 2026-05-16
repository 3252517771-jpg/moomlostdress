function dataOut = SmoothFilterResp(dataIn)
% 对原始的呼吸信号进行平滑滤波
%   输入参数dataIn为滤波前的数据
%   输出参数dataOut为滤波后的数据
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

b = [1 1 1 1 1 1 1 1 1 1 1 1] / 12; % 取平均
dataOut = filter(b, 1, dataIn); % 进行平滑滤波

Fs = 500; % 采样频率
T = 1 / Fs; % 采样周期
L = length(dataIn);  % 数据长度 
t = (0 : L - 1) * T; % 时间向量

figure; % 创建窗口
set(gcf, 'name', '呼吸信号平滑滤波'); % 设置窗口的标题名
subplot(2, 1, 1); % 将figure划分为2*1，在第1块创建坐标系
plot(t, dataIn); % 绘制原始波形
title('呼吸信号原始波形'); % 标注标题
xlabel('时间(s)'); % 标注X坐标
ylabel('幅值'); % 标注Y坐标

subplot(2, 1, 2); % 将figure划分为2*1，在第2块创建坐标系
plot(t, dataOut); % 绘制滤波后波形
title('平滑滤波后波形'); % 标注标题
xlabel('时间(s)'); % 标注X坐标
ylabel('幅值'); % 标注Y坐标