# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:51:21 2023

@author: User
"""
import numpy as np
import copy

class Model:
    def __init__(self, mod):
        self.model = mod
        self.wind_dx = np.nan
        self.wind_dy = np.nan
        self.maxDist = np.nan
        self.nbins = np.nan
        self.objMask = np.nan
        self.coordList = np.nan
        
    def setWind(self, dx, dy):
        self.wind_dx = dx
        self.wind_dy = dy
    
    def setBins(self, nbins):
        self.nbins = nbins
        
    def setDist(self, maxDist):
        self.maxDist = maxDist
        
    def calculate(self):
        # creation of initial mask
        mask = np.zeros((int(len(self.model)/self.wind_dy),
                        int(np.shape(self.model)[1]/self.wind_dx)))

        # setting parameters of hist
        minV, maxV = np.nanmin(self.model), np.nanmax(self.model)
        nbins = self.nbins
        vals = np.linspace(minV, maxV, nbins)

        # arr mask x hist
        dimMatr = np.zeros((np.shape(mask)[0], np.shape(mask)[1], len(vals)))
        
        # filling dimMatr with hist
        for i in range(int(len(self.model)/self.wind_dy)):
            for j in range(int(np.shape(self.model)[1]/self.wind_dx)):
                # print(i)
                window = self.model[i*self.wind_dy:(i+1)*self.wind_dy, j*self.wind_dx:(j+1)*self.wind_dx]
                cvals = np.zeros(len(vals))
                winVals = np.ravel(window)
                for iv in range(len(vals)-1):
                    curLen = len(
                        np.where((vals[iv] <= winVals) & (winVals < vals[iv+1]))[0])
                    cvals[iv] = curLen
                    # print(curLen)
                cvals[-1] = len(np.where(winVals >= vals[-1]))
                dimMatr[i, j, :] = cvals
        
        # setting list for groups
        groups = []

        # counting distance between hist of ij window and existing groups
        for i in range(int(len(self.model)/self.wind_dy)):
            for j in range(int(np.shape(self.model)[1]/self.wind_dx)):

                # bordEl = (i,i+1,j,j+1)
                chist = dimMatr[i, j, :]
                # chist = np.random.rand(10)
                if len(groups) == 0:
                    fisrtGr = [(i, j, chist)]
                    groups.append(fisrtGr)
                else:
                    corList = np.zeros(len(groups))
                    for igr, cgr in enumerate(groups):
                        corListSmall = []
                        for iigr, el in enumerate(cgr):
                            smallDist = np.sum(np.abs((chist - el[2])))
                            corListSmall.append(smallDist)
                        corList[igr] = np.nanmean(corListSmall)
                    minArg = np.argmin(corList)
                    minCorr = corList[minArg]
                    
                    # edge value - should be based on size of window and nbins in hist
                    if minCorr < 1500:
                        
                        crdList = []
                        for el in groups[minArg]:
                            crdList.append([el[0], el[1]])

                        # if okBord(crdList, i,j) == True:
                        groups[minArg].append((i, j, chist))
                        # else:
                        #     groups.append([(i,j,chist)])
                    else:
                        groups.append([(i, j, chist)])
        
        # mask value in xy coords
        grIndex = []
        for igr, gr in enumerate(groups):
            for el in gr:
                grIndex.append((el[0], el[1], igr))

        # converting xyz to matrix
        grIndex = np.array(grIndex)
        newMask = np.zeros_like(mask)
        for i in range(len(grIndex)):
            newMask[grIndex[i][0], grIndex[i][1]] = grIndex[i][2]+1

        # collecting borderlines of each class
        borderList = []
        for gr in groups:
            brdList = []
            for el in gr:
                i = el[0]
                j = el[1]
                u = 'h'+str(i)+'/'+str(j)
                b = 'h'+str(i+1)+'/'+str(j)
                l = 'v'+str(i)+'/'+str(j)
                r = 'v'+str(i)+'/'+str(j+1)
                brdList.append(u)
                brdList.append(b)
                brdList.append(l)
                brdList.append(r)
            uniqueBrd = []
            for i in range(len(brdList)):
                count = 0
                for j in range(len(brdList)):
                    if brdList[i] == brdList[j]:
                        count = count+1
                if count == 1:
                    uniqueBrd.append(brdList[i])
            borderList.append(uniqueBrd)
            
        # collecting coordinates of cell borders for every object class
        # this variable will be deconstructed on next steps
        coordList = []
        for ibrd, brd in enumerate(borderList):
            coord = []
            for crd in brd:
                if 'h' in crd:
                    ij = crd.replace('h', '').replace('v', '').split('/')
                    i = np.float(ij[0])
                    j = np.float(ij[1])
                    x = [j*self.wind_dx, (j+1)*self.wind_dx]
                    y = [i*self.wind_dy, i*self.wind_dy]
                    coord.append([[x[0], y[0]], [x[1], y[1]]])
                else:
                    ij = crd.replace('h', '').replace('v', '').split('/')
                    i = np.float(ij[0])
                    j = np.float(ij[1])
                    x = [j*self.wind_dx, (j)*self.wind_dx]
                    y = [i*self.wind_dy, (i+1)*self.wind_dy]
                    coord.append([[x[0], y[0]], [x[1], y[1]]])
            coordList.append(coord)
        tmpcoordList = copy.deepcopy(coordList)
        self.coordList = tmpcoordList
        
        # function to check if coord of element continues previous border line
        # function is to separate figures of one class which are located on a distance between each others
        def checkCont(first, secondList):

            for second in secondList:
                if first[0] == second[0]:
                    return True
                elif first[0] == second[1]:
                    return True
                elif first[1] == second[0]:
                    return True
                elif first[1] == second[1]:
                    return True

            return False
        
        # collecting points of separate objects for every class
        bodies = []
        for grInd, coord in enumerate(coordList):
            
            # list of objects with continues borders
            coordGroups = []
            while len(coord) > 0:
                clrCoords = []
                clrCoords.append(coord[0])
                coord.remove(coord[0])
                iniLen = len(coord)
                for i in range(iniLen):
                    for el in coord[1:]:
                        if checkCont(el, clrCoords) == True:
                            clrCoords.append(el)
                            coord.remove(el)
                if len(clrCoords) > 1:
                    coordGroups.append(clrCoords)
            
            
            for igr, gr in enumerate(coordGroups):
                grArrList = []
                for el in gr:
                    grArrList.append(el[0])
                    grArrList.append(el[1])
                grArr = np.array(grArrList)
                
                xrange = np.arange(np.min(grArr[:,0]), np.max(grArr[:,0]), self.wind_dx)
                yrange = np.arange(np.min(grArr[:,1]), np.max(grArr[:,1]), self.wind_dy)
                
                listToCheck = grArrList
                for y in yrange:
                    # y=yrange[3]
                    xspace = []
                    for el in grArr:
                        if el[1] == y:
                            if el[0] not in xspace:
                                xspace.append(el[0])     
                    xspace.sort()
                    
                    points = list(np.arange(xspace[0],xspace[-1],self.wind_dx))
                    for point in points:
                        listToCheck.append([point,y])
                       
                
                bodyList = []
                for el in listToCheck:
                    i,j = int(el[1]/self.wind_dy), int(el[0]/self.wind_dx)
                    try:
                        if newMask[i,j] == grInd+1:
                            bodyList.append(el)
                            
                    except:
                        bonk = 1
                bodies.append(bodyList)
        
        sepMask = np.zeros_like(newMask)
        for i, body in enumerate(bodies):
            for point in body:
                ii,jj = int(point[1]/self.wind_dy), int(point[0]/self.wind_dx)
                sepMask[ii,jj] = i
                
        self.objMask = sepMask
    
    
                
                
    

  
    
    