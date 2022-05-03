function [train, validation, test] = trainTestSplit(dataSet)

trainSize = 0.6;
validationSize = 0.2;
testSize = 0.2;

[train, validation, test] = splitEachLabel(dataSet, trainSize, validationSize, testSize, 'randomize');

end