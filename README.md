# The FaCells.
The FaCells. An Exploratory Study about LSTM Layers on Face Sketches Classifiers.
Code implementing the paper [https://arxiv.org/abs/2102.11361](https://arxiv.org/abs/2102.11361).

Try the demo in [http://www.columbia.edu/~xig2000/FaCells/](http://www.columbia.edu/~xig2000/FaCells/) 

To install the repo:

1. uncompress the data folder [data](https://drive.google.com/drive/folders/1dqiZDDf7n-pcDXZkYwL7zvTDX6TKqiDE?usp=share_link) file in the root directory, total 15.1 Gb. It will create a folder named 'data' with the input draws, the model, and previously calculated predictions.
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
