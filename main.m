clear all 
close all

% Setting variables
learningSpeed = 0.01;
neuronsPerLayer = 21;
iterations = 20;
wantedMeanError = 0.007;

% Reading in files
inputMatrices = dlmread("data/features.txt");
encodedOutputs = dlmread("data/targets.txt");
databaseSize = size(encodedOutputs, 1);
errorSumSquared = zeros(iterations, 1);

% Determaining sizes of input and output
inputCount = size(inputMatrices(1,:)',1);
outputCount = max(encodedOutputs);


outputLayer1 = [neuronsPerLayer, 1];
actualOutputs = zeros(outputCount,1);

outputGradient = zeros(outputCount, 1);
layer1Gradient = zeros(neuronsPerLayer, 1);
 
% Dividing it into sets:
trainingInput = inputMatrices(1:round(3/5 * databaseSize), :);
validationInput = inputMatrices(round(3/5 * databaseSize:4/5 * databaseSize), :);
testInput = inputMatrices(round(4/5 * databaseSize: databaseSize), :);

trainingOutput = encodedOutputs(1:round(3/5 * databaseSize), :);
validationOutput = encodedOutputs(round(3/5 * databaseSize:4/5 * databaseSize), :);
testOutput = encodedOutputs(round(4/5 * databaseSize: databaseSize), :);


weightsL1 = [neuronsPerLayer, inputCount];

for i = 1:neuronsPerLayer
    for j = 1:inputCount
        weightsL1(i,j) = 4*rand()-2;
    end
end
weightsL2 = [outputCount, neuronsPerLayer];
for i = 1:outputCount
    for j = 1:neuronsPerLayer
        weightsL2(i,j) = 4*rand()-2;
    end
end

train = 1;
meanError = wantedMeanError + ones(1000, 1);

goal = false;
while(~goal)
    disp("Running epoch " + train)
    
    error = zeros(size(trainingOutput,1), outputCount);
    for sampleNo = 1 : size(trainingOutput,1)
        correctOutputs = zeros(outputCount,1);
        correctOutputs(encodedOutputs(sampleNo)) = 1;
        
        % feed forward
        for neuron = 1 : neuronsPerLayer
            outputLayer1(neuron) = sigmoid(inputMatrices(sampleNo, :)', weightsL1(neuron, :)');
        end
        for neuron = 1 : outputCount
            actualOutputs(neuron) = sigmoid(outputLayer1(:), weightsL2(neuron, :));
        end
       
	    % backpropagation second layer
        for neuron = 1 : outputCount
            outputGradient(neuron) = actualOutputs(neuron) * (1 - actualOutputs(neuron)) * (correctOutputs(neuron) - actualOutputs(neuron));
        end
        for toNeuron = 1 : outputCount
            for fromNeuron = 1 : neuronsPerLayer
                weightsL2(toNeuron, fromNeuron) = weightsL2(toNeuron, fromNeuron) + (learningSpeed * outputLayer1(fromNeuron) * outputGradient(toNeuron));
            end
        end
       
		% backpropagation first layer
        for neuron = 1 : neuronsPerLayer
            layer1Gradient(neuron) = outputLayer1(neuron) * (1 - outputLayer1(neuron)) * dot(outputGradient, weightsL2(:, neuron));
        end
        for toNeuron = 1 : neuronsPerLayer
            for fromNeuron = 1 : inputCount
                weightsL1(toNeuron, fromNeuron) = weightsL1(toNeuron, fromNeuron) + (learningSpeed * inputMatrices(sampleNo, fromNeuron) * layer1Gradient(toNeuron));
            end
        end
        
	    error(sampleNo,:) = (correctOutputs - actualOutputs)';
        errorSumSquared(sampleNo,train) = sumsqr(error(sampleNo,:));
    end
	meanError(train) = sum(errorSumSquared(:,train))/size(trainingOutput,1);
    plot(meanError(1:train));
    xlim([1 max(20,train)]);
    ylim([0 max(meanError)]);
    pause(0.0001);
    
    if(meanError(train) < wantedMeanError)
        goal = true;
    end
    
    train = train + 1;
    
    
end

a = ['The goal is reached after ' , num2str(train-1), ' iterations.'];
fprintf(a);