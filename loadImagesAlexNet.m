function [trainAlex, validationAlex, testAlex] = loadImagesAlexNet()

%provide path to dataset on current machine.
filename = 'C:\Users\User\OneDrive - Northumbria University - Production Azure AD\Year 2\Intelligent Systems\archive\Project\Data';
cancerDatasetPath = fullfile(filename);

%load images into an imageDatastore to split it into train, validation and
%test sets
imageDS = imageDatastore(cancerDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainSet, validationSet, testSet] = trainTestSplit(imageDS);

outputsize = [227 227 3];

% Resize the images to match the network input layer.
trainAlex = augmentedImageDatastore(outputsize,trainSet, 'ColorPreprocessing', 'gray2rgb');
validationAlex = augmentedImageDatastore(outputsize,validationSet, 'ColorPreprocessing', 'gray2rgb');
testAlex = augmentedImageDatastore(outputsize,testSet,'ColorPreprocessing', 'gray2rgb');

end