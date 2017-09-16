clear all
close all
clc

% Training input part
input = dlmread(fullfile('C:','Users','Thomas van der Pas','Documents','TU Delft', 'Vakken', 'Computational Intelligence','CI-37','data','features.txt'));
% input = in(1:15,:);

n1 = size(input);


wantedOutput = dlmread(fullfile('C:','Users','Thomas van der Pas','Documents','TU Delft', 'Vakken', 'Computational Intelligence','CI-37','data','targets.txt'));
% wantedOutput = wantedOut(1:15,:);


n2 = max(wantedOutput);
out = full(ind2vec(wantedOutput',n2))';

n3 = size(out);
output = zeros(1,n3(2));

errorSumSquared = [];

% Learning rate
a = 0.1;

% Initial weights
w = 2.*rand(1,n1(2)) - 1;

solutionFound = false;
iter = 1;
epoch = 1;

        
        