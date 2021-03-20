import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from keras.models import Model

def plotOneRecord3D_weigths(draw, series, thres, lowerOrEqual = False, show=True, alpha_=0.15, linewidth=0.3,color="black"):
    #def plotOneFilterToDraw(convOut,id_filter):
    #    print(min())

    decreaseTransparency = 1
    
    if lowerOrEqual:
        transparency = [-x*decreaseTransparency if x < thres else 0 for x in series]
    else:
        transparency = [x*decreaseTransparency if x > thres else 0 for x in series]
    line = []
    
    for i in range(len(draw)-1):
        p1 = draw[i]
        p2 = draw[i+1]
        #print(p1,p2)
        line.append(p1)
        
        if p1[2]==1:
        #if p1[2]==0.5:
            line = np.array(line)
            #print(line[:,0], line[:,1])
            #print("---")
            if transparency[i] > 0.0:

                #plt.plot(line[:,0],1-line[:,1],"-",alpha=transparency[i], color="black",linewidth=0.30)                
                plt.plot(line[:,0],1-line[:,1],"-",alpha=alpha_, color=color,linewidth=linewidth)                
            line = []
    
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',# both major and minor ticks are affected
        left=False,
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft=False) 
    plt.box(False)
    plt.xlim([-88, 88-1])
    plt.ylim([-108+1+1, 108])
    
    #print("new function")
    #plt.savefig('foo.pdf')
    if show:            
        plt.show()

def plotOneFeature(featureId, nDraws, threshold, facellsProject, outputDir = 'output', invertedFeature = False):
    
    print(facellsProject.columnNames[featureId])
    #try:

    rowIds = facellsProject.selectNDraws(nDraws, feature=featureId, invertedFeature = invertedFeature)
    xMax = 362
    yMax = 442
    plt.figure(
            figsize=(xMax/80,yMax/80))    
    for rowId in rowIds:

        length = facellsProject.yPredDfFull.loc[rowId,'length']
        rowIdDct = facellsProject.yPredDfFull.loc[rowId,'rowId']

        draw = facellsProject.Xdct[length][rowIdDct]
        draw = draw.reshape(1,draw.shape[0],3)
        if len(facellsProject.drawWeightsDct[length].shape)==2:
            drawWeightsDctFixed = facellsProject.drawWeightsDct[length][np.newaxis,:,:]
        else:
            drawWeightsDctFixed = facellsProject.drawWeightsDct[length]
        draw_weights = np.squeeze(np.asarray(drawWeightsDctFixed[rowIdDct,:,featureId]))
        plotOneRecord3D_weigths(draw[0], draw_weights, threshold, lowerOrEqual = invertedFeature, show=False)
        #plotOneRecord3D_weigths(draw[0], [10 for i in range(len(draw[0]))], threshold, lowerOrEqual = False, show=False)

    #plt.show()
    plt.savefig(outputDir + '/' + str(featureId) + '-' + str(nDraws) + '-' + str(threshold) + '.pdf',
               bbox_inches='tight')  
    #except:
    #    print('error')

#Python 3

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageOps

def convertMatplotlibToImage(draw, eol = 0.5):
    fig = plt.figure(
        figsize=(178/50,218/50))

    canvas = FigureCanvas(fig)
    line = []
    #plt.axis('off')
    plt.xlim(0,178*2)
    plt.ylim(-218*2,0)
    ax = fig.add_subplot(1,1,1)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    for point in draw:
        #if end of the line
        if point[2]==eol:
            line.append(point)
            if len(line)>1:
                line = np.array(line)

                ax.plot(line[:,0],-line[:,1],"-",color="black",  linewidth=0.8)
            line = []
        else:
            line.append(point)
            
    canvas.draw()
    
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    
    width, height = fig.get_size_inches() * fig.get_dpi() 

    image = image.reshape(int(height), int(width), 3)
   
    #pixels = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    #.reshape(height, width, 3)
    #print(pixels)
    #print(type(pixels))
    image = Image.fromarray(image)
    
    return image
    
###do not use use v2    
def plotImageAndDraw(draw,id_):
    
    #drawImage = convertMatplotlibToImage(inDataVL.X[id_])

    #load jpg image
    
    
    id_ = id_ + 1
    jpgDir = '/Users/xaviering/data/img_align_celeba/img_align_celeba/'
    drawDir = '/Users/xaviering/data/img_align_celeba/linesJpg/'
    
    ## convert id_ to fileName
    idName = ""
    for i in range(6 - len(str(id_))):
        idName = idName + "0"
    idName = idName + str(id_) + ".jpg"
    
    drawImage = Image.open(drawDir +'l' + idName)

    imgJpg = Image.open(jpgDir + idName)
    imgJpg = imgJpg.resize(drawImage.size)
    
    outImg = get_concat_h(imgJpg,drawImage)
    
    old_size = outImg.size
    
    border = 1

    outImg = ImageOps.expand(outImg,border=border,fill='gray')
    
    return outImg
    #fig = plt.figure(figsize=(178/50,218/50))
    #ax = fig.add_subplot(1,1,1)
    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)
    #i = ax.imshow(imgJpg[...,::-1])
    
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width , im1.height+ im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

#TO DO
def analizeLayerWithX(X, model, layer_name):
    inp = model.input     
    
    mtemp = Model(inputs=inp,outputs=model.get_layer(layer_name).output)
    
    out = mtemp.predict(X)
    
    return(out)


#def plotBiLayerAccrossX(draw, id_layer):


def plotOneRecord2D_00_filter(draw, filterValues, thres):
    #normalize filters
    normalized_filterValues = [0 if x > thres else 1 for x in filterValues]
    normalized_filterValuesTxt = [str(i*i) for i in normalized_filterValues]
    xMax = 362
    yMax = 442

    plt.figure(
        figsize=(xMax/80,yMax/80))
    
    line = []

    for i in range(len(draw)-1):
        p1 = draw[i]
        p2 = draw[i+1]
        if p1[0]!=0:
            line.append(p1)
            if p2[0]==0:
                #end of line
                line = np.array(line)
                plt.plot(line[:,0],1-line[:,1],"-",color=normalized_filterValuesTxt[i])                
                line = []
                
    plt.show()

def plotOneRecord3D_series(draw, series, thres, lowerOrEqual = False, show=True):
    #def plotOneFilterToDraw(convOut,id_filter):
    #    print(min())

    decreaseTransparency = 0.2
    
    if lowerOrEqual:
        transparency = [-x*decreaseTransparency if x < thres else 0 for x in series]
    else:
        transparency = [x*decreaseTransparency if x > thres else 0 for x in series]
    line = []


    
    for i in range(len(draw)-1):
        p1 = draw[i]
        p2 = draw[i+1]
        #print(p1,p2)
        line.append(p1)
        
        if p1[2]==1:
        #if p1[2]==0.5:
            line = np.array(line)
            #print(line[:,0], line[:,1])
            #print("---")
            if transparency[i] > 0.1:
                plt.plot(line[:,0],1-line[:,1],"-",alpha=transparency[i], color="black",linewidth=0.3)                
            line = []
    
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',# both major and minor ticks are affected
        left=False,
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft=False) 
    plt.box(False)
    plt.xlim([-88, 88-1])
    plt.ylim([-108+1+1, 108])
    
    #print("new function")
    #plt.savefig('foo.pdf')
    if show:            
        plt.show()

def crop_center(pil_img, percentage = 1):
    
    
    img_width, img_height = pil_img.size
    
    w = round(img_width*percentage/100,0)
    h = round(img_height*percentage/100,0)
    print((w, h, img_width -w,img_height - h))
    return pil_img.crop((w, h, img_width -w,img_height - h))


def plotImageAndDraw2(draw,id_):
    
    #drawImage = convertMatplotlibToImage(inDataVL.X[id_])

    #load jpg image
    
    
    id_ = id_ + 1
    jpgDir = '/Users/xaviering/data/img_align_celeba/img_align_celeba/'
    #drawDir = '/Users/xaviering/data/img_align_celeba/linesJpg/'
    
    ## convert id_ to fileName
    idName = ""
    for i in range(6 - len(str(id_))):
        idName = idName + "0"
    idName = idName + str(id_) + ".jpg"
    
    #drawImage = Image.open(drawDir +'l' + idName)
    drawImage = Image.open('id_' + str(id_ -1)+'.jpg')

    print(drawImage.size)

    imgJpg = Image.open(jpgDir + idName)
    print(imgJpg.size)
    #drawImage = crop_center(drawImage, percentage=2)
    imgJpg = imgJpg.resize(drawImage.size)
    
    outImg = get_concat_h(imgJpg,drawImage)
    
    old_size = outImg.size
    
    border = 1

    outImg = ImageOps.expand(outImg,border=border,fill='gray')
    
    return outImg

### resolution of faces 178 * 218 
def plotOneDraw(idToPlot, facellsProject):
    length = facellsProject.dfFull.iloc[idToPlot]['length']
    rowId = facellsProject.dfFull.iloc[idToPlot]['rowId']

    draw = facellsProject.Xdct[length][rowId]

    img0 = plotImageAndDraw(draw,idToPlot)
    img0.save("id_" + str(idToPlot) + ".jpg")


    xMax = 362
    yMax = 442
    plt.figure(
            figsize=(xMax/80,yMax/80))   

    plotOneRecord3D_weigths(draw, series = [1 for i in range(len(draw))], 
                            thres=0, 
                            lowerOrEqual = False, show=False, alpha_=1, linewidth= 1.2)


    plt.savefig('id_' + str(idToPlot) + '.jpg',quality = 100,dpi = 600,
                   bbox_inches='tight',pad_inches=0, facecolor=(0.98, 0.98, 0.98))  
  

