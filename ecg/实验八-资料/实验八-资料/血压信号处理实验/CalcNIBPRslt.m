function [sysPres, meanPres, diaPres, pulseRate] = CalcNIBPRslt(cuffPres, pulseWave)
% 计算收缩压、平均压、舒张压和脉率
%   输入参数cuffPres为袖带压数据
%   输入参数pulseWave为脉搏波数据
%   输出参数sysPres、meanPres、diaPres、pulseRate分别为收缩压、平均压、舒张压和脉率
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

global gStepEndSeq; % 阶梯终点对应的索引序列

len = length(gStepEndSeq); % 计算索引序列长度
pulseRateSeq = zeros(1, len - 1); % 初始化脉率序列 

% 根据阶梯终点索引序列，在两个索引序列之间计算脉率值
CUT_NUM = 35; % 截去的点数
for k = 1 : len - 1
    if gStepEndSeq(k + 1) >= length(pulseWave) % 如果右侧索引已经超出脉搏波的数据长度
        k = k - 1; % 有效索引为左侧索引，因此减1之后退出
        break;
    end
    pulseRateSeq(k) = CalcPulseRate(pulseWave(gStepEndSeq(k) + CUT_NUM: gStepEndSeq(k + 1) - CUT_NUM)); % 计算脉率值  
end

pLen = length(pulseRateSeq); % 计算脉率序列的长度
sortSeq = sort(pulseRateSeq); % 对脉率序列进行排序

if pLen >= 9 % 如果脉率序列中的元素数大于等于9
    pulseRate = mean(sortSeq(4 : pLen - 3)); % 去掉首尾3个数，然后将剩余的数求平均  
else % 如果脉率序列中的元素数小于9
    minIndex = round(pLen / 2); % 取中间值对应的索引
    pulseRate = pulseRateSeq(minIndex); % 直接取中间值作为脉率值
end

pulseRate = round(pulseRate); % 四舍五入

% 拟合抛物线，计算三压 
indexPulsePeak = FindPulsePeakIndex(pulseWave); % 寻找脉搏波的波峰位置     
indexPulseValley = FindPulseValleyIndex(indexPulsePeak, pulseWave); % 寻找脉搏波的波谷位置  

pulseFitSeq = FindPulseFitSeq(pulseWave, indexPulsePeak, indexPulseValley); % 拟合脉搏波序列
cuffFitSeq = FindCuffFitSeq(cuffPres, indexPulsePeak,indexPulseValley); % 拟合袖带压序列

pulseFitSeq(1) = pulseFitSeq(2); % 第一个值是负值，去除第一个值
pulseFitSeq = ClusterPulseWave(pulseFitSeq, cuffFitSeq); % 聚类，取最大值

figure(3); % 创建窗口
set(gcf, 'name', '血压原始信号及拟合序列'); % 设置窗口的标题名
subplot(2, 1, 1); % 将figure划分为2*1，在第1块创建坐标系
L = length(pulseWave); % 数据长度
t = (1 : L); % 时间横坐标
plot(t, pulseWave, t, cuffPres * 20, ... % 绘制原始脉搏波和袖带压波形，袖带压放大20倍为了便于观察
    t(indexPulsePeak), pulseWave(indexPulsePeak), 'ro', ... % 标定脉搏波波峰（已去除伪顶点）        
    t(indexPulseValley), pulseWave(indexPulseValley), 'r*');  % 标定脉搏波波谷（已去除伪波谷）
title('袖带压和脉搏波信号原始波形（注意：袖带压放大了20倍）'); % 标注标题
xlabel('点数'); % 标注X坐标
ylabel('AD值'); % 标注Y坐标

subplot(2, 1, 2); % 将figure划分为2*1，在第2块创建坐标系
plot(pulseFitSeq); % 绘制脉搏波序列
hold on; % 保留已有曲线
plot(cuffFitSeq * 20); % 绘制袖带压序列，袖带压放大20倍为了便于观察
title('血压袖带压和脉搏波拟合序列（注意：袖带压放大了20倍）'); % 标注标题
xlabel('序列序号'); % 标注X坐标
ylabel('AD值'); % 标注Y坐标

xPres = 1 : 160; % 压力横坐标    
xPres = xPres(end : -1 : 1); % 向量逆序排列
% 平均压补偿算法，因为放气阶梯过大，需要补偿
resultCalibOffset = 2; % 校准偏置，该值需要通过与模拟器校对之后获取
cuffFitSeq = cuffFitSeq - resultCalibOffset;  
fitobject = fit(cuffFitSeq', pulseFitSeq', 'gauss1');
% 一维高斯函数 f(x) = a1*exp(-(x-b1)^2/c1^2)
fitCurve = fitobject.a1.*exp(-((xPres - fitobject.b1)./fitobject.c1).^2);
[peakVal, peakIndex] = max(fitCurve); % 获取抛物线的最大值和索引

c = find(fitCurve >= 0.58 * peakVal);
sysPres = xPres(c(1)); % 收缩压，序列的第一个点为收缩压对应的位置

d = find(fitCurve <= 0.78 * peakVal);
d = d(d > peakIndex(1)); % 峰值索引右侧，序列的第一个点为舒张压对应的位置
diaPres = xPres(d(1)); % 舒张压

meanPres = round((sysPres + 2 * diaPres) / 3); % 计算平均压

figure(4); % 创建窗口
set(gcf, 'name', '血压拟合抛物线及三压位置'); % 设置窗口的标题名
% 绘制抛物线离散点，横坐标为袖带压拟合序列，纵坐标为脉搏波峰峰值拟合序列
plot(cuffFitSeq, pulseFitSeq, '*'); % 绘制袖带压序列-脉搏波序列关系图
hold on; % 保留已有曲线
plot(xPres, fitCurve, ... % 绘制高斯拟合抛物线
    xPres(peakIndex), fitCurve(peakIndex), 'ro', ... % 标定抛物线顶点位置
    xPres(c(1)), fitCurve(c(1)), 'r+', ... % 标定收缩压位置
    xPres(d(1)), fitCurve(d(1)), 'r*'); % 标定舒张压位置
set(gca, 'XDir', 'reverse'); % X轴倒置
title('血压拟合抛物线'); % 标注标题
xlabel('压力值'); % 标注X坐标
ylabel('脉搏波峰峰值'); % 标注Y坐标