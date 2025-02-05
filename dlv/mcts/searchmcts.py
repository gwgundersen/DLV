#!/usr/bin/env python

"""
A data structure for organising search

author: Xiaowei Huang
"""

import copy
import math
import operator
import random

from dlv.configuration import configuration as cfg
from dlv.safety.regionSynth import initialiseRegions

from dlv.basics.manipulations import applyManipulation
from dlv.basics import basics

# tunable parameter for MCTS
cp = 0.5


class SearchMCTS:

    def __init__(self, model, image, k):
        self.image = image
        self.model = model
        
        self.spans = {}
        self.numSpans = {}
        self.cost = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}
        self.numberOfVisited = {}
        
        self.indexToNow = 0
        # the layer currently working on 
        self.maxilayer = k 
        # current root node
        self.rootIndex = 0
        
        # initialise root node
        self.spans[-1] = {}
        self.numSpans[-1] = {} 
        self.initialiseLeafNode(0,-1,[],[])
        
        # local actions
        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}

        # best case
        self.bestCase = (0,{},{})
        
        # useless points
        self.uselessPixels = []
        
        (self.originalClass,self.originalConfident) = self.model.predict(image)
        
    def initialiseActions(self): 
        allChildren = initialiseRegions(self.model,self.image,[])
        for i in range(len(allChildren)): 
            self.actions[i] = allChildren[i] 
        
    def initialiseLeafNode(self,index,parentIndex,newSpans,newNumSpans):
        basics.nprint("initialising a leaf node %s from the node %s"%(index,parentIndex))
        self.spans[index] = basics.mergeTwoDicts(self.spans[parentIndex],newSpans)
        self.numSpans[index] = basics.mergeTwoDicts(self.numSpans[parentIndex],newNumSpans)
        self.cost[index] = 0
        self.parent[index] = parentIndex 
        self.children[index] = []
        self.fullyExpanded[index] = False
        self.numberOfVisited[index] = 0    
        

    def destructor(self): 
        self.image = 0
        self.model = 0
        self.spans = {}
        self.numSpans = {}
        self.cost = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}
        self.numberOfVisited = {}
        
        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}
        
    # move one step forward
    # it means that we need to remove children other than the new root
    def makeOneMove(self,newRootIndex): 
        print "making a move into the new root %s, whose value is %s and visited number is %s"%(newRootIndex,self.cost[newRootIndex],self.numberOfVisited[newRootIndex])
        self.removeChildren(self.rootIndex,[newRootIndex])
        self.rootIndex = newRootIndex
    
    def removeChildren(self,index,indicesToAvoid): 
        if self.fullyExpanded[index] == True: 
            for childIndex in self.children[index]: 
                if childIndex not in indicesToAvoid: self.removeChildren(childIndex,[])
        self.spans.pop(index,None)
        self.numSpans.pop(index,None)
        self.cost.pop(index,None) 
        self.parent.pop(index,None) 
        self.children.pop(index,None) 
        self.fullyExpanded.pop(index,None)
        self.numberOfVisited.pop(index,None)
        
    def collectUselessPixels(self,index):
        potentialUseless = []
        for childIndex in self.children[index]: 
            if self.cost[childIndex] == 0:
                diffPixels = [ x for x in self.spans[childIndex].keys() if x not in self.spans[index].keys() ] 
                potentialUseless.append(diffPixels)
        pixels = set(diffPixels)
        occurrence = {}
        for p in pixels: 
            occurrence[p] = potentialUseless.count(p)
            
        uselessNum = len(pixels) / 5
        for i in range(uselessNum): 
            k = max(occurrence.iteritems(), key=operator.itemgetter(1))[0]
            self.uselessPixels.append(k) 
            
    
    def bestChild(self,index):
        allValues = {}
        for childIndex in self.children[index]: 
            allValues[childIndex] = self.cost[childIndex]
        basics.nprint("finding best children from %s"%(allValues))
        return max(allValues.iteritems(), key=operator.itemgetter(1))[0]
        
    def treeTraversal(self,index):
        if self.fullyExpanded[index] == True: 
            basics.nprint("tree traversal on node %s"%(index))
            allValues = {}
            for childIndex in self.children[index]: 
                allValues[childIndex] = (self.cost[childIndex] / float(self.numberOfVisited[childIndex])) + cp * math.sqrt(math.log(self.numberOfVisited[index]) / float(self.numberOfVisited[childIndex]))
            nextIndex = max(allValues.iteritems(), key=operator.itemgetter(1))[0]
            self.usedActionsID.append(self.indexToActionID[nextIndex])
            return self.treeTraversal(nextIndex)
        else: 
            basics.nprint("tree traversal terminated on node %s"%(index))
            availableActions = copy.deepcopy(self.actions)
            for i in self.usedActionsID: 
                availableActions.pop(i, None)
            return (index,availableActions)
        
    def initialiseExplorationNode(self,index,availableActions):
        basics.nprint("expanding %s"%(index))
        for (actionId, (span,numSpan,_)) in availableActions.iteritems() : 
            self.indexToNow += 1
            self.indexToActionID[self.indexToNow] = actionId
            self.initialiseLeafNode(self.indexToNow,index,span,numSpan)
            self.children[index].append(self.indexToNow)
        self.fullyExpanded[index] = True
        self.usedActionsID = []
        return self.children[index]

    def backPropagation(self,index,value): 
        self.cost[index] += value
        self.numberOfVisited[index] += 1
        if self.parent[index] in self.parent : 
            print("start backPropagating the value %s from node %s, whose parent node is %s"%(value,index,self.parent[index]))
            self.backPropagation(self.parent[index],value)
        else: 
            print("backPropagating ends on node %s"%(index))
            
    # start random sampling and return the eclidean value as the value
    def sampling(self,index,availableActions):
        print("start sampling node %s"%(index))
        availableActions2 = copy.deepcopy(availableActions)
        availableActions2.pop(self.indexToActionID[index], None)
        sampleValues = []
        i = 0
        for i in range(cfg.MCTS_multi_samples):
            #allChildren = copy.deepcopy(self.actions)
            (childTerminated, val) = self.sampleNext(self.spans[index],self.numSpans[index],0,availableActions2.keys(),[])
            sampleValues.append(val)
            if childTerminated == True: break
            i += 1
        return (childTerminated, max(sampleValues))

    
    def sampleNext(self,spansPath,numSpansPath,depth,availableActionIDs,usedActionIDs): 
        #print spansPath.keys()
        image1 = applyManipulation(self.image,spansPath,numSpansPath)
        (newClass,newConfident) = self.model.predict(image1)
        #print euclideanDistance(self.image,image1), newConfident, newClass
        (distMethod,distVal) = cfg.controlledSearch
        if distMethod == "euclidean": 
            dist = basics.euclideanDistance(image1,self.image)
            termValue = 0.0
            termByDist = dist > distVal
        elif distMethod == "L1": 
            dist = basics.l1Distance(image1,self.image)
            termValue = 0.0
            termByDist = dist > distVal
        elif distMethod == "Percentage": 
            dist = basics.diffPercent(image1,self.image)
            termValue = 0.0
            termByDist = dist > distVal
        elif distMethod == "NumDiffs": 
            dist = basics.diffPercent(image1,self.image) * self.image.size
            termValue = 0.0
            termByDist = dist > distVal

        if newClass != self.originalClass: 
            print("sampling a path ends in a terminal node with depth %s... "%depth)
            if self.bestCase[0] < dist: self.bestCase = (dist,spansPath,numSpansPath)
            return (depth == 0, dist)
        elif termByDist == True: 
            print("sampling a path ends by controlled search with depth %s ... "%depth)
            return (depth == 0, termValue)
        else: 

            randomActionIndex = random.choice(list(set(availableActionIDs)-set(usedActionIDs))) #random.randint(0, len(allChildren)-1)
            (span,numSpan,_) = self.actions[randomActionIndex]
            availableActionIDs.remove(randomActionIndex)
            usedActionIDs.append(randomActionIndex)
            #print span.keys()
            newSpanPath = self.mergeSpan(spansPath,span)
            newNumSpanPath = self.mergeNumSpan(numSpansPath,numSpan)
            return self.sampleNext(newSpanPath,newNumSpanPath,depth+1,availableActionIDs,usedActionIDs)
            
    def terminalNode(self,index): 
        image1 = applyManipulation(self.image,self.spans[index],self.numSpans[index])
        (newClass,_) = self.model.predict(image1)
        return newClass != self.originalClass 
        
    def terminatedByControlledSearch(self,index): 
        image1 = applyManipulation(self.image,self.spans[index],self.numSpans[index])
        (distMethod,distVal) = cfg.controlledSearch
        if distMethod == "euclidean": 
            dist = basics.euclideanDistance(image1,self.image)
        elif distMethod == "L1": 
            dist = basics.l1Distance(image1,self.image)
        elif distMethod == "Percentage": 
            dist = basics.diffPercent(image1,self.image)
        elif distMethod == "NumDiffs": 
            dist = basics.diffPercent(image1,self.image)
        print "terminated by controlled search"
        return dist > distVal 
        
    def showBestCase(self):
        return "(%s,%s)"%(self.bestCase[0],self.bestCase[1].keys())
        
    def euclideanDist(self,index): 
        image1 = applyManipulation(self.image,self.spans[index],self.numSpans[index])
        return basics.euclideanDistance(self.image,image1)
        
    def l1Dist(self,index): 
        image1 = applyManipulation(self.image,self.spans[index],self.numSpans[index])
        return basics.l1Distance(self.image,image1)
        
    def l0Dist(self,index): 
        image1 = applyManipulation(self.image,self.spans[index],self.numSpans[index])
        return basics.l0Distance(self.image,image1)
        
    def diffImage(self,index): 
        image1 = applyManipulation(self.image,self.spans[index],self.numSpans[index])
        return basics.diffImage(self.image,image1)
        
    def diffPercent(self,index): 
        image1 = applyManipulation(self.image,self.spans[index],self.numSpans[index])
        return basics.diffPercent(self.image,image1)

    def mergeSpan(self,spansPath,span): 
        return basics.mergeTwoDicts(spansPath, span)
        
    def mergeNumSpan(self,numSpansPath,numSpan):
        return basics.mergeTwoDicts(numSpansPath, numSpan)
        