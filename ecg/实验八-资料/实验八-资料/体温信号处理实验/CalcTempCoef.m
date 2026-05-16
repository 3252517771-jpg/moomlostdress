function [C1, C2] = CalcTempCoef(adA, adB)
% 计算体温系数
%   输入参数adA和adB分别为校准A和B点的采样值，这些值均在0至4095之间
%   输出参数C1和C2为体温系数
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

RA = 10; % 校准A点的电阻
RB = 7355.0; % 校准B点的电阻

meanA = nanmean(adA); % 计算平均值
meanB = nanmean(adB); % 计算平均值
VA = 5 * meanA / 4095; % 转换为电压值
VB = 5 * meanB / 4095; % 转换为电压值

C1 = (VB - VA) / (VA / RA - VB / RB);
C2 = ((RA / RB) * VB - VB) / ((RA / RB) * (VB / VA) - 1);