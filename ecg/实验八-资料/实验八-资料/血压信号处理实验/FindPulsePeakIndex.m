function indexPulsePeak = FindPulsePeakIndex(pulseWave)
% 在脉搏波中寻找波峰对应的索引序列
%   输入参数pulseWave为脉搏波数据
%   输出参数indexPulsePeak为波峰对应的索引序列
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

global gStepEndSeq; % 阶梯终点对应的索引序列
global gFakePeakIndex; % 伪波峰索引序列，要在波峰对应的索引中将伪波峰索引删除
gFakePeakIndex = []; % 初始化伪波峰索引序列

% 在脉搏波数据中，查找所有波峰索引，当然也包括伪波峰索引
[~, indexPulsePeak] = findpeaks(pulseWave, 'MinPeakProminence', 100, 'MinPeakDistance', 50);

% 基于阶梯终点索引序列，查找伪波峰的索引序列
EXP_NUM = 100; % 延伸的点数
for m = 1 : length(indexPulsePeak)
    for n = 1 : length(gStepEndSeq)
        if indexPulsePeak(m) >= gStepEndSeq(n) - EXP_NUM && indexPulsePeak(m) <= gStepEndSeq(n) + EXP_NUM 
            gFakePeakIndex = [gFakePeakIndex, indexPulsePeak(m)];
            break;
        end
    end
end

% 在indexPulsePeak中找到并删除伪波峰索引
for p = 1 : length(gFakePeakIndex)
    pos = find(indexPulsePeak == gFakePeakIndex(p));
    indexPulsePeak(pos) = []; % 删除伪波峰索引
end

% 在阶梯终点索引序列中，删除第一个索引之前的峰值对应的所有索引
for q = 1 : length(indexPulsePeak)
    if indexPulsePeak(q) <= gStepEndSeq(1)
        indexPulsePeak(q) = []; % 删除第一个索引之前的峰值对应的所有索引
    elseif indexPulsePeak(q) > gStepEndSeq(1)
        break; % 对第一个索引之后的峰值对应的索引，不予处理，因此，直接退出循环
    end
end
