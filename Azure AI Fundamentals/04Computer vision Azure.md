### Computer vision 
To a computer, an image is an **array of numeric pixel** values. An array consists of seven rows and seven columns, representing the pixel values for a 7x7 pixel image (which is known as the image's **resolution**分辨率).

The array of pixel values for this image is two-dimensional (representing rows and columns, or x and y coordinates) and defines a single rectangle of pixel values. A single layer of pixel values like this represents a grayscale image. In reality, most digital images are multidimensional and consist of three layers (known as channels) that represent red, green, and blue (**RGB**) color hues.

Using filters to process images
| -1 | -1 | -1 |
|----|----|----|
| -1 |  8 | -1 |
| -1 | -1 | -1 |

| 0   | 0   | 0   | 0   | 0   | 0   | 0   |
|-----|-----|-----|-----|-----|-----|-----|
| 0   | 0   | 0   | 0   | 0   | 0   | 0   |
| 0   | 0   | 255 | 255 | 255 | 0   | 0   |
| 0   | 0   | 255 | 255 | 255 | 0   | 0   |
| 0   | 0   | 255 | 255 | 255 | 0   | 0   |
| 0   | 0   | 0   | 0   | 0   | 0   | 0   |
| 0   | 0   | 0   | 0   | 0   | 0   | 0   |

the first value: (0 x -1) + (0 x -1) + (0 x -1) + (0 x -1) + (0 x 8) + (0 x -1) + (0 x -1) + (0 x -1) + (255 x -1) = -255

Some of the values might be outside of the 0 to 255-pixel value range, so the values are adjusted to fit into that range. Because of the shape of the filter, the outside edge of pixels isn't calculated, so a **padding value** (usually 0) is applied. The resulting array represents a new image in which the filter has transformed the original image. In this case, the filter has had the effect of highlighting the edges of shapes in the image.

#### Convolutional neural networks (CNNs)
- Images with known labels (for example, 0: apple, 1: banana, or 2: orange) are fed into the network to train the model.
- One or more layers of filters are used to extract features from each image as it is fed through the network. The filter kernels start with randomly assigned weights and generate arrays of numeric values called feature maps.
- The feature maps are flattened into a single-dimensional array of feature values.
- The feature values are fed into a fully connected neural network.
- The output layer of the neural network uses a softmax or similar function to produce a result that contains a probability value for each possible class, for example [0.2, 0.5, 0.3].
During training, the output probabilities are compared to the actual class label - for example, an image of a banana (class 1) should have the value [0.0, 1.0, 0.0]. The difference between the predicted and actual class scores is used to calculate the **loss** in the model, and the weights in the fully connected neural network and the filter kernels in the feature extraction layers are modified to reduce the loss. The training process repeats over multiple epochs until an optimal set of weights has been learned. Then, the **weights are saved**, and the model can be used to predict labels for new images for which the label is unknown.

#### Transformers and multi-modal models
Transformers work by processing huge volumes of data, and encoding language tokens (representing individual words or phrases) as vector-based embeddings (arrays of numeric values). You can think of an embedding as representing a set of dimensions that each represent some semantic attribute of the token. The embeddings are created such that tokens that are commonly used in the same context are closer together dimensionally than unrelated words.

Multi-modal models, in which the model is trained using a large volume of captioned images, with no fixed labels. An image encoder extracts features from images based on pixel values and combines them with text embeddings created by a language encoder. The overall model encapsulates relationships between natural language token embeddings and image features

### Azure AI Vision


