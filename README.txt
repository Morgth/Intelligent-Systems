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


_______________________________________________________________________________________________________________

Notes:

1) IEEE format papers order references by order of appearance, not alphabetically.
Yet since the Intelligent Systems module leader instructed to write references alphabetically in 
the academic writing guide for this module, that is how references are ordered in my paper.

2) If the download link is not working, that means that the northumbria service of newNuMySpace.co.uk is down.
If that's the case attempt using this link at a later time.