# The FaCells (under construction).
The FaCells. An Exploratory Study about LSTM Layers on Face Sketches Classifiers.
Code implementing the paper www.arxiv.link

1. uncompress the zip file in the root directory, total 15.1 Gb. It will create a folder named 'data' with the input draws, the model, and predictions pre-calculated.
2. install requirements.txt
3. see the initial_script.ipynb with the following lines.
 
##
    from functions.paperFunctions import *
    from functions.modelExploring import *
    from functions.FaCells import FaCellsProject

    facellsProject = FaCellsProject('absolute-sorted','data')
    facellsProject.loadModel("modelX.h5", 'data')
    facellsProject.loadPredictions()
    #facellsProject.model.summary()
    
    #plotOneFeature (featureId, numberOfDrawsToOverlap, threshold, facellsProject)
    plotOneFeature(6,1000, 8, facellsProject)

the script will create the photo '6-1000-8.pdf' in the output dir
