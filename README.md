The Main codes are "Grabcut.py" , "Graphcut mask.py" and "Graphcut XAI.py"
The Grabcut code is the one where draw a box around the object needed. (no model to load required)
The Graphcut mask code is where u set the scribbles for foreground and background to segeemnt the image (no model to load required)
The Graphcut XAI code is the one where u load the model, corresponding XAI masks to segment.
in order to acquire the XAI masks u need to use each of the XAI masks codes to generate them using the "Trained model" which is a trained GoogLeNet CNN.
Since its a long proccess I combined All 5 codes in One called "Combinations.py" 
Here you just have to select the image u want to segement and to make sure the model is loaded correctly from your path
This code Generates the XAI masks and uses them to pilot the Graphcut segmentation without the need to select or draw anyhting.
