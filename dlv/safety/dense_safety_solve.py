#!/usr/bin/env python

"""
author: Xiaowei Huang
"""

import copy
import time
import random
import multiprocessing

import z3

from dlv.basics import basics
from dlv.configuration import configuration as cfg


def dense_safety_solve(nfeatures,nfilters,filters,bias,input,activations,pcl,pgl,span,numSpan,pk):  

    random.seed(time.time())
    rn = random.random()
    
    # number of clauses
    c = 0
    # number of variables 
    d = 0

    # variables to be used for z3
    variable={}
    
    #print(filters)
    #print(bias)
        
    s = z3.Tactic('qflra').solver()
    s.reset()
    for l in pcl.keys():
        variable[1,0,l+1] = z3.Real('1_x_%s' % (l+1))
        d += 1

                        
    for k in span.keys():
        variable[1,1,k+1] = z3.Real('1_y_%s' % (k+1))
        d += 1
        string = "variable[1,1,%s] ==  "%(k+1)
        for l in range(nfeatures):
            if l in pcl.keys(): 
                newstr1 = " variable[1,0,%s] * %s + "%(l+1,filters[l,k])
            else: 
                newstr1 = " %s + "%(input[l]*filters[l,k])
            string += newstr1
        string += str(bias[l,k])
        s.add(eval(string))
        #print(eval(string))
        c += 1
                            
        pStr1 = "variable[1,1,%s] == %s"%(k+1, activations[k])

        s.add(eval(pStr1))
        c += 1

    basics.nprint("Number of variables: " + str(d))
    basics.nprint( "Number of clauses: " + str(c))

    p = multiprocessing.Process(target=s.check)
    p.start()

    # Wait for timeout seconds or until process finishes
    p.join(cfg.timeout)

    # If thread is still active
    if p.is_alive():
        print "Solver running more than timeout seconds (default="+str(cfg.timeout)+"s)! Skip it"
        p.terminate()
        p.join()
    else:
        s_return = s.check()

    if 's_return' in locals():
        if s_return == z3.sat:
            inputVars = [ (l,eval("variable[1,0,%s]"%(l+1))) for l in pcl.keys() ]
            cex = copy.deepcopy(input)
            for (l,x) in inputVars:
                cex[l] = getDecimalValue(s.model()[x])
                

            basics.nprint( "satisfiable!")
            return (True, cex)
        else:
            basics.nprint( "unsatisfiable!")
            return (False, input)
    else:
        print "unsatisfiable!"
        return (False, input)


def getDecimalValue(v0): 
    v = z3.RealVal(str(v0))
    return float(v.numerator_as_long())/v.denominator_as_long()
    

     