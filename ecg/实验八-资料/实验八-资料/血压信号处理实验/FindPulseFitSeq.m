function pulseFitSeq =  FindPulseFitSeq(pulseWave, indexPulsePeak, indexPulseValley)
% 查找脉搏波峰峰值拟合序列
%   输入参数pulseWave为脉搏波数据
%   输入参数indexPulsePeak为波峰索引序列
%   输入参数indexPulseValley为波谷索引序列
%   输出参数pulseFitSeq为脉搏波峰峰值拟合序列
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

% 初始化脉搏波拟合序列
pulseFitSeq = zeros(1, min(length(indexPulsePeak), length(indexPulseValley)));

% 计算波峰与波谷的差值，得到脉搏波峰峰值
for k = 1 : min(length(indexPulsePeak), length(indexPulseValley))
  pulseFitSeq(k) = pulseWave(indexPulsePeak(k)) - pulseWave(indexPulseValley(k));
end