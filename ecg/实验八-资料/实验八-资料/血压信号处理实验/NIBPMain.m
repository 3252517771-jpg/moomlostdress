% 血压控制与脉搏波袖带压信号处理实验脚本文件
%   计算收缩压、平均压、舒张压和脉率
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

global gStepEndSeq; % 阶梯终点对应的索引序列
adCuffPres  = xlsread('血压0x34演示数据-01.csv', 'A1 : A20480'); % 读取袖带压数据
adPulseWave = xlsread('血压0x34演示数据-01.csv', 'B1 : B20480'); % 读取脉搏波数据
gStepEndSeq = xlsread('gStepEndSeq-01.csv', 'A1 : A30'); % 读取索引序列
gStepEndSeq = gStepEndSeq(find(gStepEndSeq ~= 0)); % 删除序列中为0的元素

IIRFilterPulseWave(adPulseWave); % 设计IIR滤波器对原始的脉搏波信号进行滤波
FIRFilterPulseWave(adPulseWave); % 设计FIR滤波器对原始的脉搏波信号进行滤波

valCuffPres = zeros(length(adCuffPres), 0); % 初始化袖带压序列

% 根据袖带压AD采样值计算袖带压，建议根据血压硬件测量系统调整校准系数coef1和coef2
coef1 = 460; % 压力校准系数coef1
coef2 = 0.1295; % 压力校准系数coef2
for k = 1 : length(adCuffPres)
    valCuffPres(k) = (adCuffPres(k) - coef1) * coef2; % 将压力AD值转换为压力值
end

% 计算并显示收缩压、平均压、舒张压和脉率
[sysPres, meanPres, diaPres, pulseRate] = CalcNIBPRslt(valCuffPres, adPulseWave); 
disp(['平均压：' num2str(meanPres) 'mmHg']); % 在命令行窗口显示平均压
disp(['收缩压：' num2str(sysPres) 'mmHg']); % 在命令行窗口显示收缩压
disp(['舒张压：' num2str(diaPres) 'mmHg']); % 在命令行窗口显示舒张压
disp(['脉率：' num2str(pulseRate) 'bpm']); % 在命令行窗口显示脉率