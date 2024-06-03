Which natural language processing (NLP) workload is used to generate closed caption text for live presentations?  
**Azure AI Speech**
conversational language understanding (CLU)
question answering models
text analysis
_Azure AI Speech provides speech-to-text and text-to-speech capabilities through speech recognition and synthesis. You can use prebuilt and custom Speech service models for a variety of tasks, from transcribing audio to text with high accuracy to identifying speakers in conversations, creating custom voices, and more._

Which type of machine learning algorithm groups observations is based on the similarities of features?
classification
**clustering**
regression
supervised
_Clustering algorithms group data points that have similar characteristics. Regression algorithms are used to predict numeric values. Classification algorithms are used to predict a predefined category to which an input value belongs. Supervised learning is a category of learning algorithms that includes regression and classification, but not clustering._

You need to identify numerical values that represent the probability of humans developing diabetes based on age and body fat percentage.
Which type of machine learning model should you use?
hierarchical clustering
linear regression
**logistic regression**
multiple linear regression
_Multiple linear regression models a relationship between two or more features and a single label. Linear regression uses a single feature. Logistic regression is a type of classification model, which returns either a Boolean value or a categorical decision. Hierarchical clustering groups data points that have similar characteristics._

Which type machine learning algorithm predicts a numeric label associated with an item based on that item’s features?
classification
clustering
**regression**
unsupervised
The regression algorithms are used to predict numeric values. Clustering algorithms groups data points that have similar characteristics. Classification algorithms are used to predict the category to which an input value belongs. Unsupervised learning is a category of learning algorithms that includes clustering, but not regression or classification.

An electricity utility company wants to develop a mobile app for its customers to monitor their energy use and to display their predicted energy use for the next 12 months. The company wants to use machine learning to provide a reasonably accurate prediction of future energy use by using the customers’ previous energy-use data.
Which type of machine learning is this?
classification
clustering
multiclass classification
**regression**
_Regression is a machine learning scenario that is used to predict numeric values. In this example, regression will be able to predict future energy consumption based on analyzing historical time-series energy data based on factors, such as seasonal weather and holiday periods. Multiclass classification is used to predict categories of data. Clustering analyzes unlabeled data to find similarities present in the data. Classification is used to predict categories of data._

You plan to use machine learning to predict the probability of humans developing diabetes based on their age and body fat percentage.
What should the model include?
three features
three labels
**two features and one label**
two labels and one feature
_The scenario represents a model that is meant to establish a relationship between two features (age and body fat percentage) and one label (the likelihood of developing diabetes). The features are descriptive attributes (serving as the input), while the label is the characteristic you are trying to predict (serving as the output)._

In a regression machine learning algorithm, how are features and labels handled in a validation dataset?
Features are compared to the feature values in a training dataset.
**Features are used to generate predictions for the label, which is compared to the actual label values.**
Labels are compared to the label values in a training dataset.
The label is used to generate predictions for features, which are compared to the actual feature values.
_In a regression machine learning algorithm, features are used to generate predictions for the label, which is compared to the actual label value. There is no direct comparison of features or labels between the validation and training datasets._

A company is using machine learning to predict various aspects of its e-scooter hire service dependent on weather. This includes predicting the number of hires, the average distance traveled, and the impact on e-scooter battery levels. For the machine learning model, which two attributes are the features? Each correct answer presents a complete solution.
distance traveled
e-scooter battery levels
e-scooter hires
**weather temperature
weekday or weekend**
_Weather temperature and weekday or weekend are features that provide a weather temperature for a given day and a value based on whether the day is on a weekend or weekday. These are input variables for the model to help predict the labels for e-scooter battery levels, number of hires, and distance traveled. E-scooter battery levels, number of hires, and distance traveled are numeric labels you are attempting to predict through the machine learning model._

What is the purpose of a validation dataset used for as part of the development of a machine learning model?
cleaning missing data
**evaluating the trained model**
feature engineering
summarizing the data
_The validation dataset is a sample of data held back from a training dataset. It is then used to evaluate the performance of the trained model. Cleaning missing data is used to detect missing values and perform operations to fix the data or create new values. Feature engineering is part of preparing the dataset and related data transformation processes. Summarizing the data is used to provide summary statistics, such as the mean or count of distinct values in a column._

What should you do after preparing a dataset and before training the machine learning model?
clean missing data
normalize the data
**split data into training and validation datasets**
summarize the data
_Splitting data into training and validation datasets leaves you with two datasets, the first and largest of which is the training dataset you use to train the model. The second, smaller dataset is the held back data and is called the validation dataset, as it is used to evaluate the trained model. If normalizing or summarizing the data is required, it will be carried out as part of data transformation. Cleaning missing data is part of preparing the data and the data transformation processes._

You train a regression model by using automated machine learning (automated ML) in the Azure Machine Learning studio. You review the best model summary.
You need to publish the model for others to use from the internet.
What should you do next?
Create a compute cluster.
**Deploy the model to an endpoint.**
Split the data into training and validation datasets.
Test the deployed service.
_You can deploy the best performing model for client applications to use over the internet by using an endpoint. Compute clusters are used to train the model and are created directly after you create a Machine Learning workspace. Before you can test the model’s endpoint, you must deploy it first to an endpoint. Automated ML performs the validation automatically, so you do not need to split the dataset._

Which three supervised machine learning models can you train by using automated machine learning (automated ML) in the Azure Machine Learning studio? Each correct answer presents a complete solution.
**Classification**
Clustering
inference pipeline
**regression**
**time-series forecasting**
_Time-series forecasting, regression, and classification are supervised machine learning models. Automated ML learning can predict categories or classes by using a classification algorithm, as well as numeric values as part of the regression algorithm, and at a future point in time by using time-series data. Inference pipeline is not a machine learning model. Clustering is unsupervised machine learning and automated ML only works with supervised learning algorithms._

Which three data transformation modules are in the Azure Machine Learning designer? Each correct answer presents a complete solution.
**Clean Missing Data**
Model Evaluate Model
**Normalize Data**
**Select Columns in Dataset**
Train Clustering
_Normalize Data is a data transformation module that is used to change the values of numeric columns in a dataset to a common scale, without distorting differences in the range of values. The Clean Missing Data module is part of preparing the data and data transformation process. Select Columns in Dataset is a data transformation component that is used to choose a subset of columns of interest from a dataset. The train clustering model is not a part of data transformation. The evaluate model is a component used to measure the accuracy of training models._

Which machine learning algorithm module in the Azure Machine Learning designer is used to train a model?
Clean Missing Data
Evaluate Model
**Linear Regression**
Select Columns in Dataset

_Linear regression is a machine learning algorithm module used for training regression models. The Clean Missing Data module is part of preparing the data and data transformation process. Select Columns in Dataset is a data transformation component that is used to choose a subset of columns of interest from a dataset. Evaluate model is a component used to measure the accuracy of trained models._

Which computer vision solution provides the ability to identify a person's age based on a photograph?
**facial detection**
image classification
object detection
semantic segmentation

_Facial detection provides the ability to detect and analyze human faces in an image, including identifying a person's age based on a photograph. Image classification classifies images based on their contents. Object detection provides the ability to generate bounding boxes identifying the locations of different types of vehicles in an image. Semantic segmentation provides the ability to classify individual pixels in an image._

Which two specialized domain models are supported by Azure AI Vision when categorizing an image? Each correct answer presents a complete solution.
**celebrities**
image types
**landmarks**
people_
people_group
_When categorizing an image, the Azure AI Vision service supports two specialized domain models: celebrities and landmarks. Image types is an additional capability of the computer vision service, allowing it to detect the type of image, such as a clip art image or a line drawing. Both people_ and people_group are supported categories when performing image classification._

Which computer vision service provides bounding coordinates as part of its output?
image analysis
image classification
**object detection**
semantic segmentation
_Object detection provides the ability to generate bounding boxes that identify the locations of different types of objects in an image, including the bounding box coordinates, designating the location of the object in the image. Semantic segmentation provides the ability to classify individual pixels in an image. Image classification classifies images based on their contents. Image analysis extracts information from the image to label it with tags or captions._

What allows you to identify different vehicle types in traffic monitoring images?
image classification
linear regression
**object detection**
optical character recognition (OCR)
_Object detection can be used to evaluate traffic monitoring images to quickly classify specific vehicle types, such as car, bus, or cyclist. Linear regression is a machine learning training algorithm for training regression models. Image classification is part of computer vision that is concerned with the primary contents of an image. OCR is used to extract text and handwriting from images._

What can be used for an attendance system that can scan handwritten signatures?
face detection
image classification
object detection
**optical character recognition (OCR)**
_OCR is used to extract text and handwriting from images. In this case, it can be used to extract signatures for attendance purposes. Face detection can detect and verify human faces, not text, from images. Object detection can detect multiple objects in an image by using bounding box coordinates. It is not used to extract handwritten text. Image classification is the part of computer vision that is concerned with the primary contents of an image._

Which three parts of the machine learning process does the Azure AI Vision eliminate the need for? Each correct answer presents part of the solution.
Azure resource provisioning
**choosing a model**
**evaluating a model**
inferencing
**training a model**
_The computer vision service eliminates the need for choosing, training, and evaluating a model by providing pre-trained models. To use computer vision, you must create an Azure resource. The use of computer vision involves inferencing._

Which analytical task of the Azure AI Vision service returns bounding box coordinates?
image categorization
**object detection**
optical character recognition (OCR)
tagging
_Detecting objects identifies common objects and, for each, returns bounding box coordinates. Image categorization assigns a category to an image, but it does not return bounding box coordinates. Tagging involves associating an image with metadata that summarizes the attributes of the image, but it does not return bounding box coordinates. OCR detects printed and handwritten text in images, but it does not return bounding box coordinates._

