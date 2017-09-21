function [weightL1, weightL2, mse] = learningFunc(features, targets)

learningRate= 0.1;
epoch = 1;
error1 = 0.002;
[m, n] = size(features);
error = zeros(m, 7); 
achieved = 0; %if ssq is low enough we stop the iter


[weightL1, weightL2] = initialisation(10, 15, 7)
[iter, hiddenLayerOutput, outputLayer, layerOne] = activation(features, m, weightL1, weightL2, achieved);
[weightOutputNeuron, weightHiddenNeuron] = weightTraining(layerOne, outputLayer, hiddenLayerOutput, iter, learningRate, weightL1, weightL2, error)


 %calculate mse
mse(epoch) = meansqr(error);
clc;
 
%check whether we achieved goal
if(mse(epoch) <= error1)
            achieved = 1;
end
 
epoch = epoch + 1;
end

%Step 1; initialization
function [weightL1, weightL2] = initialisation(inputL, hiddenL, outputL)

weightL1 = randomWeights(hiddenL, inputL);
weightL2 = randomWeights(outputL,hiddenL);
disp(weightL1)

end

%step 2: activation
function [iter, hiddenLayerOutput, outputLayer, layerOne] = activation(features, m, weightL1, weightL2, achieved)

while achieved < 1

    for iter = 1:m
    layerOne = features(iter,:);   %loop through layer one
    hiddenLayerOutput = sigmoidFunc(layerOne*weightL1); %output of hidden layer
    outputLayer = sigmoidFunc(hiddenLayerOutput*weightL2); %output of output layer
    
    end

end

end

%step 3: weight training
function [weightOutputNeuron, weightHiddenNeuron] = weightTraining(layerOne, outputLayer, hiddenLayerOutput, iter, learningRate, weightL1, weightL2, error)
    
    %calculate error gradient in output layer and update weights
    errorGradOutput = outputLayer .* (1-outputLayer)' .* error(iter, :);
    weightCorrectionOutput = learningRate * hiddenLayerOutput * errorGradOutput;
    weightOutputNeuron = weightL1 + weightCorrectionOutput;
    
    
    %calculate error gradient in hidden layer and update weights
    errorGradHidden = hiddenLayerOutput .* (1-hiddenLayerOutput)' .* error(iter,:);
    weightCorrectionHidden = learningRate * layerOne * errorGradHidden;
    weightHiddenNeuron = weightL2 + weightCorrectionHidden;
    
end
