# IsoPredict
A program for training, testing, and applying isotopic models for region-of-origin prediction in Latin America.

## Using the Program
This program has been extensively documented for ease-of-use. First, ensure all libraries and required functions are accesed by the program. The specific purpose of each created function is documented. The main program begins with accessing, sorting, and splitting the data,

Provide your uniique file name to the quotes ('INSERT FILE NAME HERE') to allow the variable 'fileName" to access the data. If accessing the program for the first time, ensure the 'splitData' line is uncommented. This line will perform an 80%/20% split of the data creating two new files in the working directory. These new files will automically be named train_'YOUR FILE NAME' and test_'YOUR FILE NAME' and will be used throughout the program. Once data is split, comment out the splitData line. 

The following 20 lines of code specify which parts of the data will be used. "Indices" refers to the data's columns, "Regions" defines the geographic regions for classification and passes these to the models. Varialbe 'X' specifies the isotopes to be used and uses the 'indices' matrix to specify where those data are in the dataset. Variable 'Y' specifices the regions for those isotope data to be classified into, again referring to the 'indices' matrix. Here, the use (or disuse) of a random chance classifier can be chosen by uncommenting (or commenting) the line to provide a baseline for expected minimum results.

Lines 175 to 243 provide parameter grids for six different machine learning algorithms (_k_-nearest neighbors, support vector machines, multilayer perceptron/artificial neural network, random forest classifier, decision tree classifier, and gaussian näive Bayes) for tuning purposes. If using different data, some of these parameters might need to be updated to be fully applicable (e.g., changing the max_depth of the random forest if attempting to classify into more than 10 regions). Line 248 provides the cross-validation results for each algorithm and provides the accuracies and parameters for the best three results per algorithm.

The results from tuning can then be passed to the algorithms in the following section. This section simply defines the algorithm used for analysis and the parameters they will use in the process. The section after allows the user to apply those models to the training data for performance validation.

When the user is ready to evaluate these models on novel data, the t



## Available Data
meswd_kriged_HR -> high resolution hydrogen and oxygen isotopic data for Mesoamerica from Emperical Bayesian Kriging isoscapes made from tap and bottled water data available from waterisotopes.org.

UBC_Isotope_Data -> hydrogen and oxygen isotope data from the bone and teeth of nine identified migrants. Hydrogen values interpolated from isoscapes as they were not available and oxygen values have undergone phosphate to carbonate conversion.

human -> template for curating data with the purpose of predicition. Can also serve as a template for curating isotopic data to train models on.

## Available Results
Performance results for the all region (six-region) and  northern regions (four-region) models as reported in Delgado, Thomas A. 2023. "Towards the Identification of Unidentified Remains at the US-Mexico Border: A Stable Isotope and Machine Learning Approach." MA Thesis, California State University, Chico.

