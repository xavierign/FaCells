# The FaCells (under construction).
The FaCells. An Exploratory Study about LSTM Layers on Face Sketches Classifiers.
Code implementing the paper www.arxiv.link
 
    from functions.paperFunctions import *
    from functions.modelExploring import *
    from functions.FaCells import FaCellsProject

    facellsProject = FaCellsProject('absolute-sorted','data')
    facellsProject.loadModel("modelX.h5", 'data')
    facellsProject.loadPredictions()
    #facellsProject.model.summary()
    
    #plotOneFeature (featureId, numberOfDrawsToOverlap, threshold, facellsProject)
    plotOneFeature(6,1000, 8, facellsProject)
