% 体温信号处理实验脚本文件
%   检测体温探头状态，计算体温系数，然后计算体温探头阻值及其对应的温度值
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

clear all; % 清除所有的变量

% 计算体温系数
rawADCalcA = xlsread('体温0x32CalcA.csv', 'A1:A200'); % 读取200个校准A点的采样值
rawADCalcB = xlsread('体温0x32CalcB.csv', 'A1:A200'); % 读取200个校准B点的采样值
[C1, C2] = CalcTempCoef(rawADCalcA, rawADCalcB); % 计算体温系数

% 计算体温探头阻值，并计算与其对应的温度值
rawADTemp1 = xlsread('体温0x32演示数据-01.csv', 'A1:A200'); % 读取200个体温探头1的采样值
rawADTemp2 = xlsread('体温0x32演示数据-01.csv', 'B1:B200'); % 读取200个体温探头2的采样值
[temp1, temp2] = CalcTempVal(C1, C2, rawADTemp1, rawADTemp2); % 计算体温值
