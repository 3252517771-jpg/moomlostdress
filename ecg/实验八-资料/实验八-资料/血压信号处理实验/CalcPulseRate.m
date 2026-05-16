function pulseRate = CalcPulseRate(dataIn)
% 计算脉率
%   输入参数dataIn为波形数据
%   输出参数pulseRate为脉率
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

MIN_PULSE_RATE = 0;   % 脉率最小值
MAX_PULSE_RATE = 300; % 脉率最大值

[~, index] = findpeaks(dataIn, 'MinPeakProminence', 100, 'MinPeakDistance', 50);

arrX = zeros(length(index) - 1, 1); % 预分配内存
for iCnt = 1 : length(index) - 1
    arrX(iCnt) = index(iCnt + 1) - index(iCnt);
end

medianY = median(arrX); % 求数组的中值
pulseRate = int16(30000 / medianY); % 转换为bpm为单位的脉率值，并取值为整数

if pulseRate < MIN_PULSE_RATE || pulseRate > MAX_PULSE_RATE 
    pulseRate = int16(-100); % 赋值为无效值
end 