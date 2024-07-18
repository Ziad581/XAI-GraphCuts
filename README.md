The Main codes are "Grabcut.py" , "Graph-Cut.py" and "XAI-Graph-Cut.py"
The Grabcut code is the one where you draw a box around the object needed. (no model to load required)
The Graph-cut code is where u set the scribbles for foreground and background to segeemnt the image (no model to load required)
The Graphcut XAI code is the one where u load the model, corresponding XAI masks to segment.
in order to acquire the XAI masks u need to use each of the XAI masks codes to generate them using the "GoogLeNetAugmented.pt" which is a trained GoogLeNet CNN.
How the netword was trained is explanied in the "training.py" code. Here the learning rate was set to 0.001 due to the low computaiton power I had.
You can try to set it to 0.0001 in the AI Clustering.
Since it's a long proccess I combined All 5 codes in One called "Combinations.py" 
Here you just have to select the image u want to segement and to make sure the model is loaded correctly from your path
This code Generates the XAI masks and uses them to pilot the Graphcut segmentation without the need to select or draw anyhting.
