function cuffFitSeq = FindCuffFitSeq(cuffPres, indexPulsePeak, indexPulseValley)
% 查找袖带压拟合序列
%   输入参数cuffPres为袖带压数据
%   输入参数indexPulsePeak为波峰索引序列
%   输入参数indexPulseValley为波谷索引序列
%   输出参数cuffFitSeq为袖带压拟合序列
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

cuffFitSeq = zeros(1, length(indexPulseValley)); % 初始化袖带压拟合序列

 % 获取袖带压拟合序列
for k = 1 : length(indexPulseValley)
  cuffFitSeq(k) = cuffPres(indexPulsePeak(k));
end