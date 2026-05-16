function indexPulseValley = FindPulseValleyIndex(indexPulsePeak, pulseWave)
% 在脉搏波中寻找波谷对应的索引序列
%   输入参数indexPulsePeak为波峰对应的索引序列
%   输入参数pulseWaveData为脉搏波数据
%   输入参数cuffPresData为袖带压数据
%   输出参数indexPulseValley为波谷对应的索引序列
%   注意：波谷是基于波峰索引，由于波峰索引序列已经去除了伪波峰索引，因此，波谷索引序列也就不包含伪波谷索引
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

mirrorPulseWave = 5000 - pulseWave; % 计算镜像脉搏波
indexPulseValley = zeros(1, length(indexPulsePeak)); % 初始化波谷对应的索引序列

indexPulseValley(1) = 1; % 默认第一个索引为1，这样就可以确保波谷与波峰对应的索引数一致
for k = 2 : length(indexPulsePeak)
	[~, maxIndex]= max(mirrorPulseWave(indexPulsePeak(k - 1) : indexPulsePeak(k))); % 计算波谷值
	indexPulseValley(k) = maxIndex + indexPulsePeak(k - 1); % maxIndex是以峰值索引为起点，因此要求和 
end