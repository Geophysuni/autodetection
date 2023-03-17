# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:48:04 2023

@author: User
"""

import matplotlib.pyplot as plt
import imageio as iio
from autodetection import Model




# Importing initial data as np matrix

asis_image = iio.imread("oblaka-1.jpg")
ini_image = iio.imread("oblaka-1.jpg")[:, :, 0]

m = Model(ini_image)
m.setWind(30,30)
m.setBins(20)
m.setDist(1500)

m.calculate()

fig, ax = plt.subplots(1,2)
ax[0].imshow(ini_image)
ax[1].imshow(m.objMask)
for objClass in m.coordList:
    for cbrd in objClass:
        x = [cbrd[0][0],cbrd[1][0]]
        y = [cbrd[0][1],cbrd[1][1]]
        ax[0].plot(x,y, color = 'red')


