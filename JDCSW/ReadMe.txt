JDCSW (Joint learning of dictionary and classifier with shared weights)
This file explains how JDCSW.py works.
The code consists of a main folder named as JDCSW with following structure

JDCSW (main folder)
   Data (This folder contains datasets)
        ExtendedYaleB.mat
        ARdat.npz
        caltechData.npz
        ucfdata.npz
        spatialpyramidfeatures4scene15
   Results (This folder contains the result files for each experiment)
   JDCSW.py (main code file)
How to run the code?
Please follow the instructions as below
1) Install python and all python packages listed in the file JDCSW.py.
2) Open the terminal and make the folder JDCSW the current location/folder
3) type python JDCSW.py and press return key
-The code will run five experiments as per protocols listed in the paper.
-Experiments can be run in one go or can also be run one by one as per selection from the code file.
-Our algorithm performs sampling of parameters while iterating through atoms one by one in each of the main iterations.
-We have designed and vetorized the code in such a way that we can treat atoms in groups that reduces the number of iterations at atoms level. We have named this variable as "batchsize" in the code.
-We have set minimum batchsize for each experiment in the code. Increasing batchsize beyond this will effect the accuracy of the result.
