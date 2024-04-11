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
Microsoft's Azure AI Vision service provides prebuilt and customizable computer vision models that are based on the Florence foundation model and provide various powerful capabilities. With Azure AI Vision, you can create sophisticated computer vision solutions quickly and easily; taking advantage of "off-the-shelf" functionality for many common computer vision scenarios, while retaining the ability to create custom models using your own images.

Azure resources for Azure AI Vision service
- Azure AI Vision: A specific resource for the Azure AI Vision service. Use this resource type if you don't intend to use any other Azure AI services, or if you want to track utilization and costs for your Azure AI Vision resource separately.
- Azure AI services: A general resource that includes Azure AI Vision along with many other Azure AI services; such as Azure AI Language, Azure AI Custom Vision, Azure AI Translator, and others. Use this resource type if you plan to use multiple AI services and want to simplify administration and development.

Azure AI Vision supports multiple image analysis capabilities, including:
- Optical character recognition (OCR) - extracting text from images.
- Generating captions and descriptions of images.
- Detection of thousands of common objects in images.
- Tagging visual features in images: Azure AI Vision can suggest tags for an image based on its contents. These tags can be associated with the image as metadata that summarizes attributes of the image and can be useful if you want to index an image along with a set of key terms that might be used to search for images with specific attributes or contents.

Training custom models: If the built-in models provided by Azure AI Vision don't meet your needs, you can use the service to train a custom model for image classification or object detection. Azure AI Vision builds custom models on the pre-trained foundation model, meaning that you can train sophisticated models by using relatively few training images.

###  Fundamentals of Facial Recognition
**Face detection** involves identifying regions of an image that contain a human face, typically by returning bounding box coordinates that form a rectangle around the face. 

With **Face analysis**, facial features can be used to train machine learning models to return other information, such as facial features such as nose, eyes, eyebrows, lips, and others.

A further application of facial analysis is to train a machine learning model to identify known individuals from their facial features. This is known as facial recognition, and uses multiple images of an individual to train the model. This trains the model so that it can detect those individuals in new images on which it wasn't trained. When used responsibly, facial recognition is an important and useful technology that can improve efficiency, security, and customer experiences. 

Microsoft Azure provides multiple Azure AI services that you can use to detect and analyze faces, including:
- Azure AI Vision, which offers face detection and some basic face analysis, such as returning the bounding box coordinates around an image.
- Azure AI Video Indexer, which you can use to detect and identify faces in a video.
- Azure AI Face, which offers pre-built algorithms that can detect, recognize, and analyze faces.

The Azure Face service can return the rectangle coordinates for any human faces that are found in an image, as well as a series of attributes related to those faces such as:
- Accessories: indicates whether the given face has accessories. This attribute returns possible accessories, including headwear, glasses, and masks, with a confidence score between zero and one for each accessory.
- Blur: how blurred the face is, which can be an indication of how likely the face is to be the main focus of the image.
- Exposure: such as whether the image is underexposed or over exposed. This applies to the face in the image and not the overall image exposure.
- Glasses: whether or not the person is wearing glasses.
- Head pose: the face's orientation in a 3D space.
- Mask: indicates whether the face is wearing a mask.
- Noise: refers to visual noise in the image. If you have taken a photo with a high ISO setting for darker settings, you would notice this noise in the image. The image looks grainy or full of tiny dots that make the image less clear.
- Occlusion: determines if there might be objects blocking the face in the image.

To use the Face service, you must create one of the following types of resource in your Azure subscription:
- Face: Use this specific resource type if you don't intend to use any other Azure AI services, or if you want to track utilization and costs for Face separately.
- Azure AI services: A general resource that includes Azure AI Face along with many other Azure AI services such as Azure AI Content Safety, Azure AI Language, and others. Use this resource type if you plan to use multiple Azure AI services and want to simplify administration and development.

There are some considerations that can help improve the accuracy of the detection in the images:
- Image format - supported images are JPEG, PNG, GIF, and BMP.
- File size - 6 MB or smaller.
- Face size range - from 36 x 36 pixels up to 4096 x 4096 pixels. Smaller or larger faces will not be detected.
- Other issues - face detection can be impaired by extreme face angles, extreme lighting, and occlusion (objects blocking the face such as a hand).

### Fundamentals of optical character recognition 
Automating text processing can improve the speed and efficiency of work by removing the need for manual data entry. The ability to recognize printed and handwritten text in images is beneficial in scenarios such as note taking, digitizing medical records or historical documents, scanning checks for bank deposits, and more.

The ability for computer systems to process written and printed text is an area of AI where computer vision intersects with natural language processing. Vision capabilities are needed to "read" the text, and then natural language processing capabilities make sense of it.

Azure AI Vision service has the ability to extract machine-readable text from images. Azure AI Vision's Read API is the OCR engine that powers text extraction from images, PDFs, and TIFF files. OCR for images is optimized for general, non-document images that makes it easier to embed OCR in your user experience scenarios.

The Read API, otherwise known as Read OCR engine, uses the latest recognition models and is optimized for images that have a significant amount of text or have considerable visual noise. It can automatically determine the proper recognition model to use taking into consideration the number of lines of text, images that include text, and handwriting.

Calling the Read API returns results arranged into the following hierarchy:
- Pages - One for each page of text, including information about the page size and orientation.
- Lines - The lines of text on a page.
- Words - The words in a line of text, including the bounding box coordinates and text itself.



