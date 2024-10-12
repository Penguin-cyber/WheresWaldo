# WheresWaldo
A CNN trained to localize Waldo from a specific Where's Waldo Background.

To try the model, run the main.py file. It will load the model I have trained, "waldo_model.keras" and it will test it on randomly generated testing data.

The architecture of the model consists of convolutional layers and dense layers. There are 7 convolutional layers consisting of filters, batch normalizations, and max pooling. There are 3 dense layers. 

The model was optimized using mean squared error and AdamW. 

This was built using tensorflow and keras, pillow, and matplotlib. 
