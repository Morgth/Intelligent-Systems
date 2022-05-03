function alexNetLayers = alexNetLayers(trainAlex,validationAlex, testAlex)
%load the AlexNet
alexNet = alexnet;
%transfer all the layers of the network, excluding the last three
layerTransfer = alexNet.Layers(1:end-3);

miniBatchSize = 16;

layers = [ 
    layerTransfer 
    fullyConnectedLayer(4, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor',20) 
    softmaxLayer 
    classificationLayer];




options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationAlex, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%train network
alexNetLayers = trainNetwork(trainAlex,layers,options);



%display test accuracy
YPred = classify(alexNetLayers, testAlex, 'miniBatchSize', miniBatchSize);
[data, YTest] = read(testAlex);
YTest = YTest.Label;
if(length(YPred) ~= length(YTest))
    YPred = YPred(1:length(YTest));
end
accuracy = sum(YPred == YTest)/numel(YTest);
disp(accuracy);
end