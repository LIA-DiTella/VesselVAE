from vmtk import pypes
from vmtk import vmtkscripts
import os

import vtk
from vmtk import vtkvmtk
import sys


#iterativo para hacerlo en todos los de la carpeta


myPype = pypes.PypeRun("vmtknetworkextraction -ifile smoothed.vtp   -ofile smoothed-network.vtp")
#cfolder = os.listdir('./centerlines')
#myPype = pypes.PypeRun("vmtkcenterlinesections -ifile ArteryObjAN25-7.vtp -centerlinesfile  centerlines/ArteryObjAN25-7-network.vtp -ofile A.vtp")

#myPype = pypes.PypeRun("vmtkcenterlines -ifile tr_reg_000.vtp  -ofile centerlines/A.vtp")
'''
for file in files:
    #print(file.split(".")[1])
    if file.split(".")[1] == "vtp":
        if file.split(".")[0] + '-network.vtp' not in cfolder:
            script = 'vmtknetworkextraction '
            input_file =  '-ifile ' + file 
            output_file = ' -ofile centerlines/' + file.split(".")[0] + '-network.vtp'
            myPype = pypes.PypeRun(script+input_file+output_file)
'''
#files = ["ArteryObjAN1-19.vtp"]
'''
sfolder = os.listdir('./crossSections')
for file in files:
    if file.split(".")[1] == "vtp":
        if file.split(".")[0] + '-section.vtp' not in sfolder:
            script = 'vmtkcenterlinesections '
            input_file =  '-ifile ' + file
            centerline_file = " -centerlinesfile centerlines/"+file.split(".")[0] + '-network.vtp '
            output_file = ' -ofile crossSections/' + file.split(".")[0] + '-section.vtp'
            #myPype = pypes.PypeRun(script+input_file+centerline_file +output_file)
'''
#myPype = pypes.PypeRun("vmtknetworkextraction -ifile ArteryObjAN129-17.vtp -advancementratio 1.05  -ofile centerlines/An.vtp")
#myPype = pypes.PypeRun("vmtkcenterlinesections -ifile ArteryObjAN1-7.vtp -centerlinesfile  ArteryObjAN1-7network.vtp -ofile ArteryObjAN1-7sections.vtp")

#1- sacar centerline
#myPype = pypes.PypeRun("vmtknetworkextraction -ifile ArteryObjAN1-2.vtp -advancementratio 1.0  -ofile An.vtp")


#2- cross section