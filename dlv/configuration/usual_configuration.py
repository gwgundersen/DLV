#!/usr/bin/env python

"""
Define paramters
author: Xiaowei Huang
"""

def usual_configuration(dataset):

    if dataset == "twoDcurve": 
    
        # which image to start with or work with 
        # from the database
        startIndexOfImage = 0
        
        # the start layer to work from 
        startLayer = 0

        # the maximal layer to work until 
        maxLayer = 2

        ## number of features of each layer 
        # in the paper, dims_L = numOfFeatures * featureDims
        numOfFeatures = 1
        featureDims = 2
        
        ## control by distance
        controlledSearch = ("euclidean",1)
        #controlledSearch = ("L1",0.05)
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 30
        MCTS_all_maximal_time = 120
        MCTS_multi_samples = 3

        # point-based or line-based, or only work with a specific point
        #enumerationMethod = "convex"
        enumerationMethod = "line"

        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        exitWhen = "foundAll"
        #exitWhen = "foundFirst"
                
    elif dataset == "mnist": 

        # which image to start with or work with 
        # from the database
        startIndexOfImage = 100
        
        # the start layer to work from 
        startLayer = -1
        # the maximal layer to work until 
        maxLayer = 2
        
        ## number of features of each layer, used for heuristic search 
        numOfFeatures = 150  
        featureDims = 5 
        
        ## control by distance
        #controlledSearch = ("euclidean",10)
        controlledSearch = ("L1",40)
        #controlledSearch = ("Percentage",0.12)
        #controlledSearch = ("NumDiffs",30)
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 30
        MCTS_all_maximal_time = 120
        MCTS_multi_samples = 5

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"

        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        #exitWhen = "foundAll"
        exitWhen = "foundFirst"
                
        
    elif dataset == "gtsrb": 

        # which image to start with or work with 
        # from the database
        startIndexOfImage = 1
        
        # the start layer to work from 
        startLayer = -1
        # the maximal layer to work until 
        maxLayer = 0

        ## number of features of each layer 
        numOfFeatures = 500 
        featureDims = 5
        
        ## control by distance
        #controlledSearch = ("euclidean",10)
        controlledSearch = ("L1",20)
        #controlledSearch = ("Percentage",0.12)
        #controlledSearch = ("NumDiffs",30)
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 30
        MCTS_all_maximal_time = 120
        MCTS_multi_samples = 3

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"


        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        #exitWhen = "foundAll"
        exitWhen = "foundFirst"
                
    elif dataset == "cifar10": 
    
        # which image to start with or work with 
        # from the database
        startIndexOfImage = 1
        
        # the start layer to work from 
        startLayer = -1
        # the maximal layer to work until 
        maxLayer = 0

        ## number of features of each layer 
        numOfFeatures = 500
        featureDims = 5

        
        ## control by distance
        #controlledSearch = ("euclidean",10)
        controlledSearch = ("L1",20)
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 30
        MCTS_all_maximal_time = 120
        MCTS_multi_samples = 3

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"

        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        #exitWhen = "foundAll"
        exitWhen = "foundFirst"
        
    elif dataset == "imageNet": 
    
        # which image to start with or work with 
        # from the database
        startIndexOfImage = 1
        
        # the start layer to work from 
        startLayer = 1
        # the maximal layer to work until 
        maxLayer = 1

        ## number of features of each layer 
        numOfFeatures = 20000
        featureDims = 5
        
        ## control by distance
        controlledSearch = ("L1",10000)
        #controlledSearch = ("L1",50)
        
        # MCTS_level_maximal_time
        MCTS_level_maximal_time = 300
        MCTS_all_maximal_time = 1800
        MCTS_multi_samples = 3

        # point-based or line-based, or only work with a specific point
        enumerationMethod = "convex"
        #enumerationMethod = "line"
        #enumerationMethod = "point"

        #checkingMode = "specificLayer"
        checkingMode = "stepwise"
        
        # exit whenever an adversarial example is found
        #exitWhen = "foundAll"
        exitWhen = "foundFirst"
        
    
    return (featureDims,startIndexOfImage,startLayer,maxLayer,numOfFeatures,controlledSearch,MCTS_all_maximal_time, MCTS_level_maximal_time,MCTS_multi_samples,enumerationMethod,checkingMode,exitWhen)