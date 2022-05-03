function network = layers(train, validation, test)

layers = [
    imageInputLayer([300 300 1]) %images are 300x300 greyscale
    
    convolution2dLayer(5,64,'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(5,'Stride',5)
    
    convolution2dLayer(5,64,'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(5,'Stride',5)
    
    convolution2dLayer(5,128,'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,'Stride',3)

    convolution2dLayer(3,256,'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    globalAveragePooling2dLayer
    
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer
    ];
    

miniBatchSize = 16;
initialLearnRate = 1e-1*miniBatchSize/256;

options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu', ... % Turn on automatic multi-gpu support.
    'InitialLearnRate',initialLearnRate, ... % Set the initial learning rate.
    'MiniBatchSize',miniBatchSize, ... % Set the MiniBatchSize.
    'Verbose',false, ... % Do not send command line output.
    'Plots','training-progress', ... % Turn on the training progress plot.
    'L2Regularization',1e-10, ...
    'MaxEpochs',40, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validation, ...
    'ValidationFrequency',floor(numel(train.Files)/miniBatchSize), ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20);

%train network
network = trainNetwork(train,layers,options);

%display test accuracy
YPred = classify(network, test, 'miniBatchSize', miniBatchSize);
[data, YTest] = read(test);
YTest = YTest.Label;
if(length(YPred) ~= length(YTest))
    YPred = YPred(1:length(YTest));
end
accuracy = sum(YPred == YTest)/numel(YTest);
disp(accuracy);
end