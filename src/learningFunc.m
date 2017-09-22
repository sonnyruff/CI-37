function [weightL1, weightL2, mse] = learningFunc(features, targets)

learningRate= 0.1;
epoch = 1;
error1 = 0.002;
[m, ~] = size(features);
error = zeros(m, 7); 
achieved = 0; %if ssq is low enough we stop the iter

[weightL1, weightL2] = initialisation(10, 15, 7);

while achieved < 1

    for iter = 1:m
        layerOne = features(iter,:);   %loop through layer one
        hiddenLayerOutput = sigmoidFunc(layerOne*weightL1); %output of hidden layer
        outputLayer = sigmoidFunc(hiddenLayerOutput*weightL2); %output of output layer
    
        %calculate error gradient in output layer and update weights
        errorGradOutput = outputLayer .* (1-outputLayer)' .* error(iter, :);
        weightCorrectionOutput = learningRate * hiddenLayerOutput * errorGradOutput;
        weightOutputNeuron = weightL1 + weightCorrectionOutput;
    
                    %outputs calculated for errors
            exp =   transformer(targets(i,:));
            error(i,:) = exp - weightOutputNeuron;
        
        %calculate error gradient in hidden layer and update weights
        errorGradHidden = hiddenLayerOutput .* (1-hiddenLayerOutput)' .* error(iter,:);
        weightCorrectionHidden = learningRate * layerOne * errorGradHidden;
        weightHiddenNeuron = weightL2 + weightCorrectionHidden;
    end

        mse(epoch) = meansqr(error);
        clc;

        %check whether we achieved goal
        if(mse(epoch) <= error1)
        achieved = 1;
        
        epoch = epoch + 1;
        
        end
end

%[iter, hiddenLayerOutput, outputLayer, layerOne] = activation(error, targets,features, m, weightL1, weightL2, achieved);



%Step 1; initialization
function [weightL1, weightL2] = initialisation(inputL, hiddenL, outputL)

weightL1 = randomWeights(hiddenL, inputL);
weightL2 = randomWeights(outputL,hiddenL);


end

function desired = transformer(classnum)
    desired = [0;0;0;0;0;0;0];
    desired(classnum,:) = 1;
end
end

