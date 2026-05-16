function dataOut = ClusterPulseWave(dataIn, cuffPres)
% 根据袖带压对脉搏波进行聚类处理
%   输入参数dataIn为脉搏波数据
%   输入参数cuffPres为袖带压数据
%   输出参数dataOut为进行聚类处理之后的脉搏波数据
%   COPYRIGHT 2018-2020 LEYUTEK. All rights reserved.

% 最小的袖带压间隔
MIN_CP_INTERVAL = 2;

clusterIndex = []; % 将聚类索引序列的初值赋值为空

% 从第二个索引开始，同一压力阶梯中第一个压力值对应的索引保存到变量clusterIndex
for k = 2 : length(cuffPres)
    if abs(cuffPres(k) - cuffPres(k-1)) >= MIN_CP_INTERVAL
       clusterIndex = [clusterIndex, k];
    end
end

% 将袖带压拟合序列的最后一个元素的索引作为聚类索引序列的最后一个元素
clusterIndex = [clusterIndex, length(dataIn)];

% 在同一压力阶梯，将所有的压力值更新为该阶梯中最大的压力值
for k = 1 : length(clusterIndex)
	if k == 1
        if clusterIndex(1) - 1 >= 1
            dataIn(1 : clusterIndex(1) - 1) = max(dataIn(1 : clusterIndex(1) - 1)); 
        end
	else
        if clusterIndex(k) - 1 >= clusterIndex(k - 1)
            dataIn(clusterIndex(k - 1) : clusterIndex(k) - 1) = max(dataIn(clusterIndex(k - 1) : clusterIndex(k) - 1));
        end
	end
end

dataOut = dataIn; % 将dataIn赋值给dataOut