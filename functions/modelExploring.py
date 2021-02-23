from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from keras import Model

def parseLogFile(filepath):
    #filepath = 'Iliad.txt'
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        df = []
        while line:
            if "s/step" in str(line.strip()):
                l_txt = line.strip().split()
                df.append([float(l_txt[7]),
                           float(l_txt[10]),
                           float(l_txt[13]),
                             float(l_txt[16])])
            line = fp.readline()
            cnt += 1
        
    return(np.array(df))


def exploreLogDir(dir_path = "logs2compare/"):

    onlyfilesRaw = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

    onlyfiles = []
    for i in onlyfilesRaw:
        if '.out' in i:
            onlyfiles.append(i)

    for f in onlyfiles:
        perf = parseLogFile(dir_path + f)
        plt.plot(perf[:,0])
    plt.legend(onlyfiles,loc='upper right')
    plt.show()

    for f in onlyfiles:
        perf = parseLogFile(dir_path + f)
        plt.plot(perf[:,2],)
    plt.legend(onlyfiles,loc='upper right')
    plt.show()

    for f in onlyfiles:
        perf = parseLogFile(dir_path + f)
        plt.plot(perf[:,1])
    plt.legend(onlyfiles,loc='lower right')
    plt.show()

    for f in onlyfiles:
        perf = parseLogFile(dir_path+ f)
        plt.plot(perf[:,3],)
    plt.legend(onlyfiles,loc='lower right')
    plt.show()

def analizeLayerWithOneDraw(X, model, layer_name, layer_node_id=0):
    inp = model.input     
    
    mtemp = Model(inputs=inp,outputs=model.get_layer(layer_name).output)
    
    out = mtemp.predict([X])[0]
    
    return(out[:,layer_node_id])

def analizeLayerWithX(X, model, layer_name, layer_node_id=0):
    inp = model.input     
    
    mtemp = Model(inputs=inp,outputs=model.get_layer(layer_name).output)
    
    out = mtemp.predict(X)[0]
    
    return(out)
    # input placeholder
    #outputs = [layer.output for layer in model.layers]          # all layer outputs
    #functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
   
    #return outputs, functors

def plotDrawHorizontaly(draw, output1, output2, output3,l1 = 0, l2=800, inverse = False, 
                            ticks=False): # ticks =-0.5 trace eol.
    
    ticksEach = 20
    def plotTicks(ticksAt):
        for xc in ticksAt:
            plt.axvline(x=xc, lw = 0.5, color = "gray")

    
    zoom_range=range(l1,l2)
    if inverse:
        zoom_range2 =  range(800-l1-1,800-l2-1,-1)
    else: 
        zoom_range2 = zoom_range

    ticksAt = []

    for p in zoom_range2:

        if draw[p,2] == ticks:
            ticksAt.append(p)

    #ticksEach = [100,200,300]
    ticksEach = ticksAt

    plt.figure(
        figsize=(356*3/80,436/80))
    plt.subplot(4, 1, 1)
    plt.plot(zoom_range,draw[zoom_range2,0]/500,"-",color="green")
    plt.plot(zoom_range,draw[zoom_range2,1]/500,"-",color="blue")

    if ticks:
        plotTicks(ticksEach)
    #print(outConv[0,range2Conv,i3])
    
    plt.subplot(4, 1, 2)
    plt.plot(zoom_range,output1[zoom_range],"-",color="violet")
    plt.axhline(y=0, lw = 0.5, color = "gray")
    if ticks:
        plotTicks(ticksEach)

    plt.subplot(4, 1, 3)
    plt.plot(zoom_range,output2[zoom_range],"-",color="brown")
    plt.axhline(y=0, lw = 0.5, color = "gray")
    if ticks:
        plotTicks(ticksEach)
        
    plt.subplot(4, 1, 4)
    plt.plot(zoom_range,output3[zoom_range],"-",color="orange")
    plt.axhline(y=0, lw = 0.5, color = "gray")
    if ticks:
        plotTicks(ticksEach)
    plt.show()