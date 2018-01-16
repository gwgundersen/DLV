import copy
import os
import numpy as np
import random
import time

from dlv.safety.regionSynth import regionSynth
from dlv.safety.precisionSynth import precisionSynth
from dlv.safety.safetyanalysis import safety_analysis

from dlv.configuration import configuration as cfg
from dlv.basics import basics

from dlv.basics.searchtree import SearchTree
from dlv.mcts.searchmcts import SearchMCTS
from dlv.mcts.mctstwoplayer import MCTSTwoPlayer
from dlv.basics.datacollection import DataCollection

from dlv.basics.manipulations import applyManipulation

# ------------------------------------------------------------------------------

class Solver(object):

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.n_samples = len(dataset.X_train)

    def verify(self, out_dir, idx):
        if not out_dir:
            raise AttributeError('Please provide an output directory.')
        mkdir(out_dir)
        cfg.directory_pic_string = out_dir
        if not idx:
            for i in range(self.n_samples):
                cfg.startIndexOfImage = i
                _verify(self.model, self.dataset)
        else:
            cfg.startIndexOfImage = idx
            _verify(self.model, self.dataset)

    def example(self):
        from dlv.nn.mnistmodel import MnistModel
        from dlv.datasets.mnistdata import MnistData
        _verify(MnistModel(), MnistData())

# ------------------------------------------------------------------------------

def mkdir(name):
    if not os.path.exists(name):
        os.makedirs(name)
    return name

# ------------------------------------------------------------------------------

def _verify(model, dataset):
    dc = DataCollection()
    n_succ = 0
    start = cfg.startIndexOfImage
    end = cfg.startIndexOfImage + cfg.dataProcessingBatchNum
    for i in range(start, end):
        print("\n\nprocessing input of index %s in the dataset: " % (str(i)))
        succ = handleOne(model, dataset, dc, i)
        if succ:
            n_succ += 1

    pct = n_succ / float(cfg.dataProcessingBatchNum)
    dc.addSuccPercent(pct)
    dc.provideDetails()
    dc.summarise()
    dc.close()

# ------------------------------------------------------------------------------
# Safety checking starting from the a specified hidden layer.
# ------------------------------------------------------------------------------

## how many branches to expand
numOfPointsAfterEachFeature = 1

mcts_mode  = "sift_twoPlayer"
#mcts_mode  = "singlePlayer"

def handleOne(model, dataset, dc, imgIdx):
    print(imgIdx)

    # get an image to interpolate
    image = dataset.getTestImage(imgIdx)
    print("the shape of the input is "+ str(image.shape))

    if cfg.dataset == "twoDcurve": image = np.array([3.58747339,1.11101673])

    dc.initialiseIndex(imgIdx)
    originalImage = copy.deepcopy(image)

    if cfg.checkingMode == "stepwise":
        k = cfg.startLayer
    elif cfg.checkingMode == "specificLayer":
        k = cfg.maxLayer

    while k <= cfg.maxLayer:

        layerType = model.getLayerType(k)
        re = False
        start_time = time.time()

        # only these layers need to be checked
        if layerType in ["Convolution2D","Conv2D", "Dense", "InputLayer"] and k >= 0 :

            dc.initialiseLayer(k)

            st = SearchTree(image, k)
            st.addImages(model,[image])

            print("\n================================================================")
            print "\nstart checking the safety of layer "+str(k)

            (originalClass,originalConfident) = model.predict(image)
            origClassStr = dataset.labels(int(originalClass))

            path0="%s/%s_original_as_%s_with_confidence_%s.png"%(cfg.directory_pic_string, imgIdx, origClassStr, originalConfident)
            dataset.save(-1,originalImage, path0)

            # for every layer
            f = 0
            while f < cfg.numOfFeatures :

                f += 1
                print("\n================================================================")
                print("Round %s of layer %s for image %s" % (f, k, imgIdx))
                index = st.getOneUnexplored()
                imageIndex = copy.deepcopy(index)

                # for every image
                # start from the first hidden layer
                t = 0
                re = False
                while True and index != (-1,-1):

                    # pick the first element of the queue
                    print "(1) get a manipulated input ..."
                    (image0,span,numSpan,numDimsToMani,_) = st.getInfo(index)

                    print "current layer: %s."%(t)
                    print "current index: %s."%(str(index))

                    path2 = cfg.directory_pic_string+"/temp.png"
                    print "current operated image is saved into %s"%(path2)
                    dataset.save(index[0],image0,path2)

                    print "(2) synthesise region from %s..."%(span.keys())
                     # ne: next region, i.e., e_{k+1}
                    #print "manipulated: %s"%(st.manipulated[t])
                    (nextSpan,nextNumSpan,numDimsToMani) = regionSynth(model,cfg.dataset,image0,st.manipulated[t],t,span,numSpan,numDimsToMani)
                    st.addManipulated(t,nextSpan.keys())

                    (nextSpan,nextNumSpan,npre) = precisionSynth(model,image0,t,span,numSpan,nextSpan,nextNumSpan)

                    print "dimensions to be considered: %s"%(nextSpan)
                    print "spans for the dimensions: %s"%(nextNumSpan)

                    if t == k:

                        # only after reaching the k layer, it is counted as a pass
                        print "(3) safety analysis ..."
                        # wk for the set of counterexamples
                        # rk for the set of images that need to be considered in the next precision
                        # rs remembers how many input images have been processed in the last round
                        # nextSpan and nextNumSpan are revised by considering the precision npre
                        (nextSpan,nextNumSpan,rs,wk,rk) = safety_analysis(model, dataset, t, imgIdx, st, index, nextSpan, nextNumSpan, npre)
                        if len(rk) > 0:
                            rk = (zip (*rk))[0]

                            print "(4) add new images ..."
                            random.seed(time.time())
                            if len(rk) > numOfPointsAfterEachFeature:
                                rk = random.sample(rk, numOfPointsAfterEachFeature)
                            diffs = basics.diffImage(image0,rk[0])
                            print("the dimensions of the images that are changed in the this round: %s"%diffs)
                            if len(diffs) == 0:
                                st.clearManipulated(k)
                                return

                            st.addImages(model,rk)
                            st.removeProcessed(imageIndex)
                            (re,percent,eudist,l1dist,l0dist) = reportInfo(image,wk)
                            print "euclidean distance %s"%(basics.euclideanDistance(image,rk[0]))
                            print "L1 distance %s"%(basics.l1Distance(image,rk[0]))
                            print "L0 distance %s"%(basics.l0Distance(image,rk[0]))
                            print "manipulated percentage distance %s\n"%(basics.diffPercent(image,rk[0]))
                            break
                        else:
                            st.removeProcessed(imageIndex)
                            break
                    else:
                        print "(3) add new intermediate node ..."
                        index = st.addIntermediateNode(image0,nextSpan,nextNumSpan,npre,numDimsToMani,index)
                        re = False
                        t += 1
                if re == True:
                    dc.addManipulationPercentage(percent)
                    print "euclidean distance %s"%(eudist)
                    print "L1 distance %s"%(l1dist)
                    print "L0 distance %s"%(l0dist)
                    print "manipulated percentage distance %s\n"%(percent)
                    dc.addEuclideanDistance(eudist)
                    dc.addl1Distance(l1dist)
                    dc.addl0Distance(l0dist)
                    (ocl,ocf) = model.predict(wk[0])
                    dc.addConfidence(ocf)
                    break

            if f == cfg.numOfFeatures:
                print "(6) no adversarial example is found in this layer within the distance restriction."
            st.destructor()

        elif layerType in ["Input"]  and k < 0 and mcts_mode  == "sift_twoPlayer" :

            print "directly handling the image ... "

            dc.initialiseLayer(k)

            (originalClass,originalConfident) = model.predict(image)
            origClassStr = dataset.labels(int(originalClass))
            path0="%s/%s_original_as_%s_with_confidence_%s.png"%(cfg.directory_pic_string, imgIdx, origClassStr, originalConfident)
            dataset.save(-1,originalImage, path0)

            # initialise a search tree
            st = MCTSTwoPlayer(model, model, image, image, -1, "cooperator", dataset)
            st.initialiseActions()

            st.setManipulationType("sift_twoPlayer")

            start_time_all = time.time()
            runningTime_all = 0
            numberOfMoves = 0
            while st.terminalNode(st.rootIndex) == False and st.terminatedByControlledSearch(st.rootIndex) == False and runningTime_all <= cfg.MCTS_all_maximal_time:
                print("the number of moves we have made up to now: %s"%(numberOfMoves))
                eudist = st.euclideanDist(st.rootIndex)
                l1dist = st.l1Dist(st.rootIndex)
                l0dist = st.l0Dist(st.rootIndex)
                percent = st.diffPercent(st.rootIndex)
                diffs = st.diffImage(st.rootIndex)
                print("euclidean distance %s"%(eudist))
                print("L1 distance %s"%(l1dist))
                print("L0 distance %s"%(l0dist))
                print("manipulated percentage distance %s"%(percent))
                print("manipulated dimensions %s"%(diffs))

                start_time_level = time.time()
                runningTime_level = 0
                childTerminated = False
                while runningTime_level <= cfg.MCTS_level_maximal_time:
                    (leafNode,availableActions) = st.treeTraversal(st.rootIndex)
                    newNodes = st.initialiseExplorationNode(leafNode,availableActions)
                    for node in newNodes:
                        (childTerminated, value) = st.sampling(node,availableActions)
                        #if childTerminated == True: break
                        st.backPropagation(node,value)
                    #if childTerminated == True: break
                    runningTime_level = time.time() - start_time_level
                    basics.nprint("best possible one is %s"%(str(st.bestCase)))
                bestChild = st.bestChild(st.rootIndex)
                #st.collectUselessPixels(st.rootIndex)
                st.makeOneMove(bestChild)

                image1 = st.applyManipulationToGetImage(st.spans[st.rootIndex],st.numSpans[st.rootIndex])
                diffs = st.diffImage(st.rootIndex)
                path0="%s/%s_temp_%s.png"%(cfg.directory_pic_string, imgIdx, len(diffs))
                dataset.save(-1,image1,path0)
                (newClass,newConfident) = model.predict(image1)
                print("confidence: %s"%(newConfident))

                if childTerminated == True: break

                # store the current best
                (_,bestSpans,bestNumSpans) = st.bestCase
                image1 = st.applyManipulationToGetImage(bestSpans,bestNumSpans)
                path0="%s/%s_currentBest.png"%(cfg.directory_pic_string, imgIdx)
                dataset.save(-1,image1,path0)

                numberOfMoves += 1
                runningTime_all = time.time() - start_time_all

            (_,bestSpans,bestNumSpans) = st.bestCase
            #image1 = applyManipulation(st.image,st.spans[st.rootIndex],st.numSpans[st.rootIndex])
            image1 = st.applyManipulationToGetImage(bestSpans,bestNumSpans)
            (newClass,newConfident) = model.predict(image1)
            newClassStr = dataset.labels(int(newClass))
            re = newClass != originalClass

            if re == True:
                path0="%s/%s_%s_%s_modified_into_%s_with_confidence_%s.png"%(cfg.directory_pic_string, imgIdx, "sift_twoPlayer", origClassStr, newClassStr, newConfident)
                dataset.save(-1,image1,path0)
                path0="%s/%s_diff.png"%(cfg.directory_pic_string, imgIdx)
                dataset.save(-1,np.subtract(image,image1),path0)
                print("\nfound an adversary image within prespecified bounded computational resource. The following is its information: ")
                print("difference between images: %s"%(basics.diffImage(image,image1)))

                print("number of adversarial examples found: %s"%(st.numAdv))

                eudist = basics.euclideanDistance(st.image,image1)
                l1dist = basics.l1Distance(st.image,image1)
                l0dist = basics.l0Distance(st.image,image1)
                percent = basics.diffPercent(st.image,image1)
                print("euclidean distance %s"%(eudist))
                print("L1 distance %s"%(l1dist))
                print("L0 distance %s"%(l0dist))
                print("manipulated percentage distance %s"%(percent))
                print("class is changed into %s with confidence %s\n"%(newClassStr, newConfident))
                dc.addRunningTime(time.time() - start_time_all)
                dc.addConfidence(newConfident)
                dc.addManipulationPercentage(percent)
                dc.addEuclideanDistance(eudist)
                dc.addl1Distance(l1dist)
                dc.addl0Distance(l0dist)

                #path0="%s/%s_original_as_%s_heatmap.png"%(directory_pic_string,imgIdx,origClassStr)
                #plt.imshow(GMM(image),interpolation='none')
                #plt.savefig(path0)
                #path1="%s/%s_%s_%s_modified_into_%s_heatmap.png"%(directory_pic_string,imgIdx,"sift_twoPlayer", origClassStr,newClassStr)
                #plt.imshow(GMM(image1),interpolation='none')
                #plt.savefig(path1)
            else:
                print("\nfailed to find an adversary image within prespecified bounded computational resource. ")


        elif layerType in ["Input"]  and k < 0 and mcts_mode  == "singlePlayer" :

            print "directly handling the image ... "

            dc.initialiseLayer(k)

            (originalClass,originalConfident) = model.predict(image)
            origClassStr = dataset.labels(int(originalClass))
            path0="%s/%s_original_as_%s_with_confidence_%s.png"%(cfg.directory_pic_string, imgIdx, origClassStr, originalConfident)
            dataset.save(-1,originalImage, path0)

            # initialise a search tree
            st = SearchMCTS(model, image, k)
            st.initialiseActions()

            start_time_all = time.time()
            runningTime_all = 0
            numberOfMoves = 0
            while st.terminalNode(st.rootIndex) == False and st.terminatedByControlledSearch(st.rootIndex) == False and runningTime_all <= cfg.MCTS_all_maximal_time:
                print("the number of moves we have made up to now: %s"%(numberOfMoves))
                eudist = st.euclideanDist(st.rootIndex)
                l1dist = st.l1Dist(st.rootIndex)
                l0dist = st.l0Dist(st.rootIndex)
                percent = st.diffPercent(st.rootIndex)
                diffs = st.diffImage(st.rootIndex)
                print "euclidean distance %s"%(eudist)
                print "L1 distance %s"%(l1dist)
                print "L0 distance %s"%(l0dist)
                print "manipulated percentage distance %s"%(percent)
                print "manipulated dimensions %s"%(diffs)

                start_time_level = time.time()
                runningTime_level = 0
                childTerminated = False
                while runningTime_level <= cfg.MCTS_level_maximal_time:
                    (leafNode,availableActions) = st.treeTraversal(st.rootIndex)
                    newNodes = st.initialiseExplorationNode(leafNode,availableActions)
                    for node in newNodes:
                        (childTerminated, value) = st.sampling(node,availableActions)
                        if childTerminated == True: break
                        st.backPropagation(node,value)
                    if childTerminated == True: break
                    runningTime_level = time.time() - start_time_level
                    print("best possible one is %s"%(st.showBestCase()))
                bestChild = st.bestChild(st.rootIndex)
                #st.collectUselessPixels(st.rootIndex)
                st.makeOneMove(bestChild)

                image1 = applyManipulation(st.image,st.spans[st.rootIndex],st.numSpans[st.rootIndex])
                diffs = st.diffImage(st.rootIndex)
                path0="%s/%s_temp_%s.png"%(cfg.directory_pic_string, imgIdx, len(diffs))
                dataset.save(-1,image1,path0)
                (newClass,newConfident) = model.predict(image1)
                print "confidence: %s"%(newConfident)

                if childTerminated == True: break

                # store the current best
                (_,bestSpans,bestNumSpans) = st.bestCase
                image1 = applyManipulation(st.image,bestSpans,bestNumSpans)
                path0="%s/%s_currentBest.png"%(cfg.directory_pic_string, imgIdx)
                dataset.save(-1,image1,path0)

                runningTime_all = time.time() - runningTime_all
                numberOfMoves += 1

            (_,bestSpans,bestNumSpans) = st.bestCase
            #image1 = applyManipulation(st.image,st.spans[st.rootIndex],st.numSpans[st.rootIndex])
            image1 = applyManipulation(st.image,bestSpans,bestNumSpans)
            (newClass,newConfident) = model.predict(image1)
            newClassStr = dataset.labels(int(newClass))
            re = newClass != originalClass
            path0="%s/%s_%s_modified_into_%s_with_confidence_%s.png"%(cfg.directory_pic_string, imgIdx, origClassStr, newClassStr, newConfident)
            dataset.save(-1,image1,path0)
            #print np.max(image1), np.min(image1)
            print("difference between images: %s"%(basics.diffImage(image,image1)))
            #plt.imshow(image1 * 255, cmap=mpl.cm.Greys)
            #plt.show()

            if re == True:
                eudist = basics.euclideanDistance(st.image,image1)
                l1dist = basics.l1Distance(st.image,image1)
                l0dist = basics.l0Distance(st.image,image1)
                percent = basics.diffPercent(st.image,image1)
                print "euclidean distance %s"%(eudist)
                print "L1 distance %s"%(l1dist)
                print "L0 distance %s"%(l0dist)
                print "manipulated percentage distance %s"%(percent)
                print "class is changed into %s with confidence %s\n"%(newClassStr, newConfident)
                dc.addEuclideanDistance(eudist)
                dc.addl1Distance(l1dist)
                dc.addl0Distance(l0dist)
                dc.addManipulationPercentage(percent)

            st.destructor()


        else:
            print("layer %s is of type %s, skipping"%(k,layerType))
            #return

        runningTime = time.time() - start_time
        dc.addRunningTime(runningTime)
        if re == True and cfg.exitWhen == "foundFirst":
            break
        k += 1

    print("Please refer to the file %s for statistics."%(dc.fileName))
    return re


def reportInfo(image,wk):

    # exit only when we find an adversarial example
    if wk == []:
        print "(5) no adversarial example is found in this round."
        return (False,0,0,0,0)
    else:
        print "(5) an adversarial example has been found."
        image0 = wk[0]
        eudist = basics.euclideanDistance(image,image0)
        l1dist = basics.l1Distance(image,image0)
        l0dist = basics.l0Distance(image,image0)
        percent = basics.diffPercent(image,image0)
        return (True,percent,eudist,l1dist,l0dist)
