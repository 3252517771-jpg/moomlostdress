function dataOut = Add50HzNoise(dataIn)
% 对输入的数据叠加50Hz工频干扰信号
%   输入参数rawData为原始波形数据
%   输出参数dataOut为叠加50Hz工频干扰信号后的数据
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

Fs = 500;    % 采样频率
Ts = 1 / Fs; % 采样间隔
L = length(dataIn); % 数据长度 
t = (1 : L) * Ts; % 时间横坐标

dataOut = dataIn' + 1000 * sin(2 * pi * 50 * t); % dataIn是列向量，因此要转置