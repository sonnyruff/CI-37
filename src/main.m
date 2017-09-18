clear all
close all
clc
%rng(2077333712)

nodeCounts = [10, 10, 7];
layerCount = numel(nodeCounts) - 1;

weightMatrix = generateWeightMatrix(nodeCounts);