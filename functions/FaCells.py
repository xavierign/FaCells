#TODO move to a utils file

from os import listdir
from os.path import isfile, join
import numpy as np
import pickle
from keras.models import load_model, Model
from collections import Counter
import random, time
import pandas as pd

class FaCellsProject:
    def __init__(self, formatName):
        self.Xdct = {}
        self.ydct = {}
        self.dfdct = {}
        self.formatName = formatName
        self.idLocator = {}
        self.lengthList =[]
        self.drawWeightsDct = {}
        self.lengthListCounted = []
        self.dfFull = []
        self.yPredDct = {}
        self.yPredDfFull = pd.DataFrame()
        
        # Using readlines() 
        fileTemp = open('data/colNames.txt', 'r')
        self.columnNames = fileTemp.read().split("\n")

        # formatName (S0s,S3s,all)
        if formatName!="multi":
            mypath = "/Users/xaviering/Desktop/AI experiments/deepFaceDraw/data/" + formatName + "/split/"
        else:
            mypath = "/Users/xaviering/Desktop/AI experiments/deepFaceDraw/data/S0s/multi/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for f in onlyfiles:
            if "X" in f and 'icloud' not in f:
                length = int(f.split(".")[0].split("_")[1])
                self.Xdct[length] = np.load(mypath + 'X_' + str(length) + '.npy')
                self.ydct[length] = np.load(mypath + 'y_' + str(length) + '.npy')
                self.dfdct[length]  = np.load(mypath + 'df_' + str(length) + '.npy')
                self.lengthList.append( length)
                self.lengthListCounted.append(len(self.Xdct[length]))

        #TODO load dfDict and merge them into one
        with open('data/dfFull.pickle', 'rb') as handle:
            self.dfFull = pickle.load(handle)
            
    def loadModel(self, modelName):
        modelDir = "/Users/xaviering/Desktop/AI experiments/deepFaceDraw/models/"
        self.model = load_model(modelDir + modelName,compile=False) 
        
    def loadPredictions(self):
        with open('data/drawWeightsDct.pickle', 'rb') as handle:
            self.drawWeightsDct = pickle.load(handle)    
        
        with open('data/yPredDct.pickle', 'rb') as handle:
            self.yPredDct = pickle.load(handle)  
        
        with open('data/yPredDfFull.pickle', 'rb') as handle:
            self.yPredDfFull = pickle.load(handle)  
            self.yPredDfFull = self.yPredDfFull.reset_index()

    def selectNDraws (self, nDraws, feature='', invertedFeature = False):   
        
        if feature == '':
            rowIds = random.sample(range(len(yPredDfFull)), nDraws)
        else:
            if not invertedFeature:
                rowIds = self.yPredDfFull.index[self.yPredDfFull[feature] > 0.5].tolist()
            else:
                rowIds = self.yPredDfFull.index[self.yPredDfFull[feature] < 0.5].tolist()
            rowIds = random.sample(rowIds, nDraws)
        return rowIds
    
    def calculatePredictions(self):
        
        weights_dense = facellsProject.model.get_layer('dense').get_weights()
        weights_output = facellsProject.model.get_layer('output').get_weights()
        input_layer = facellsProject.model.input
        output_layer = facellsProject.model.get_layer('bi3').output
        mtemp = Model(inputs=input_layer,outputs=output_layer)
        
        #for length in [340]:
        for length in self.lengthList:
            print(length)
            bi_output = mtemp.predict(facellsProject.Xdct[length])
            nRec = len(facellsProject.Xdct[length])
            bi_output = mtemp.predict(facellsProject.Xdct[length])       
            bias_temp = np.repeat(np.array(weights_dense[1])[np.newaxis,:], length, axis=0)
            bias_dense = np.repeat(bias_temp[np.newaxis,:,:],nRec, axis=0)
            bias_temp2 = np.repeat(np.array(weights_output[1])[np.newaxis,:], length, axis=0)
            bias_output = np.repeat(bias_temp2[np.newaxis,:,:],nRec, axis=0)
            #check dimensions
            dense_output = bi_output.dot(np.array(weights_dense[0])) + bias_dense
            final_output = dense_output.dot(np.array(weights_output[0])) + bias_output
            self.drawWeightsDct[length] = np.squeeze(np.asarray(final_output))
            self.yPredDict[length] = pd.DataFrame(facellsProject.model.predict(
                                    facellsProject.Xdct[length]))
            self.yPredDict[length]['length'] = length
            self.yPredDict[length]['rowId'] = range(len(self.yPredDict[length]))
            self.yPredDfFull = self.yPredDfFull.append(self.yPredDict[length])
            