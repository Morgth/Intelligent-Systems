load images for CNN:
[train, validation, test] = loadImages()


load imagse for AlexNet CNN:
[trainAlex, validationAlex, testAlex] = loadImagesAlexNet()


train CNN:
network = layers(train, validation, test)


train AlexNet CNN:
alexNet = alexNetLayers(trainAlex, validationAlex, testAlex)


The link bellow is a download link for the pretained models used in the assessment. Networks1,2,3 
are in order as were talked about in the research paper.

I made the download link of the trained models because their size was too large to push into github.

Network1 - 100 epochs with the original architecture.
Network2 - additional conv layer and slightly adjusted architecture parameters, still trained over 100 epochs.
Network3 - reduced epochs to 40. same architecture as Network2. Adjusted options.
alexNet - model which was trained over 10 epochs, not the initially attempted 6.

http://unn-w20036800.newnumyspace.co.uk/downloadModels.html