% Copyright Sonny 2k17

clear all
close all
clc
%rng(2077333712)

input = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]];
output = zeros(1,4);
wantedOutput = [0,0,0,1];

errorSumSquared = [];

% Learning rate
a = 0.1;

% Initial weights
w = 2.*rand(1,2) - 1;

solutionFound = false;
iter = 1;
epoch = 1;

while(~solutionFound)
    for i = 1:4
        output(i) = perceptron(input(i,:), w(iter,:));
        
        error(iter) = wantedOutput(i) - output(i);
        w(iter + 1,:) = w(iter,:) + a * input(i,:) * error(iter);
        iter = iter + 1;
    end
    errorSumSquared(epoch) = sumsqr(error((epoch-1)*4+1:epoch*4));
    
    disp(epoch + " : " + output(1) + ", " + output(2) + ", " + output(3) + ", " + output(4));
    
    epoch = epoch + 1;
    
    if output == wantedOutput | iter > 10000
        solutionFound = true;
    end
    pause
end

plot(errorSumSquared);
figure
plot(w);