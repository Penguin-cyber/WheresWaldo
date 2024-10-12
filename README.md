# WheresWaldo
A CNN trained to localize Waldo from a specific Where's Waldo Background.

To try the model yourself, you will have to train it and then test it. You can train it by running train.py and then test it by running test.py. Note that it may take a very long time to train. 

The architecture of the model consists of convolutional layers and dense layers. There are 7 convolutional layers consisting of filters, batch normalizations, and max pooling. There are 3 dense layers. You can see the model summary upon running train.py.

The model was optimized using mean squared error and AdamW. 

This was built using tensorflow and keras, pillow, and matplotlib. 

Total params: 16,555,042 (63.15 MB)

Trainable params: 16,552,002 (63.14 MB)

Non-trainable params: 3,040 (11.88 KB)
