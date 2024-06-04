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

Which two specialized domain models are supported by using the Azure AI Vision service? Each correct answer presents a complete solution.
animals
cars
**celebrities
landmarks**
plants
_The Azure AI Vision service supports the celebrities and landmarks specialized domain models. It does not support specialized domain models for animals, cars, or plants._

Which additional piece of information is included with each phrase returned by an image description task of the Azure AI Vision?
bounding box coordinates
**confidence score**
endpoint
key
_Each phrase returned by an image description task of the Azure AI Vision includes the confidence score. An endpoint and a key must be provided to access the Azure AI Vision service. Bounding box coordinates are returned by services such as object detection, but not image description._

Which two prebuilt models allow you to use the Azure AI Document Intelligence service to scan information from international passports and sales accounts? Each correct answer presents part of the solution.
business card model
**ID document model
invoice model**
language model
receipt model
_The invoice model extracts key information from sales invoices and is suitable for extracting information from sales account documents. The ID document model is optimized to analyze and extract key information from US driver’s licenses and international passport biographical pages. The business card model, receipt model, and language model are not suitable to extract information from passports or sales account documents._

Which three sources can be used to generate questions and answers for a knowledge base? Each correct answer presents a complete solution.
**a webpage**
an audio file
**an existing FAQ document**
an image file
**manually entered data
**
_A webpage or an existing document, such as a text file containing question and answer pairs, can be used to generate a knowledge base. You can also manually enter the knowledge base question-and-answer pairs. You cannot directly use an image or an audio file to import a knowledge base._

[Answer choice] use plugins to provide end users with the ability to get help with common tasks from a generative AI model.
**Copilots**
Language Understanding solutions
Question answering models
RESTful API services
_Copilots are often integrated into applications to provide a way for users to get help with common tasks from a generative AI model. Copilots are based on a common architecture, so developers can build custom copilots for various business-specific applications and services._

At which layer can you apply content filters to suppress prompts and responses for a responsible generative AI solution?
metaprompt and grounding
model
**safety system**
user experience
_The safety system layer includes platform-level configurations and capabilities that help mitigate harm. For example, the Azure OpenAI service includes support for content filters that apply criteria to suppress prompts and responses based on the classification of content into four severity levels (safe, low, medium, and high) for four categories of potential harm (hate, sexual, violence, and self-harm)._

xxx can return responses, such as natural language, images, or code, based on natural language input.
Computer vision
Deep learning
**Generative AI**
Machine learning
Reinforcement learning
_Generative AI models offer the capability of generating images based on a prompt by using DALL-E models, such as generating images from natural language. The other AI capabilities are used in different contexts to achieve other goals._

xxx can used to identify constraints and styles for the responses of a generative AI model.
Data grounding
Embeddings
**System messages**
Tokenization
_System messages should be used to set the context for the model by describing expectations. Based on system messages, the model knows how to respond to prompts. The other techniques are also used in generative AI models, but for other use cases._

Which two capabilities are examples of a GPT model? Each correct answer presents a complete solution.
**Create natural language.**
Detect specific dialects of a language.
Generate closed captions in real-time from a video.
Synthesize speech.
**Understand natural language.**
_Azure OpenAI natural language models can take in natural language and generate responses. GPT models are excellent at both understanding and creating natural language._

You plan to develop an image processing solution that will use DALL-E as a generative AI model.
Which capability is NOT supported by the DALL-E model?
**image description**
image editing
image generation
image variations
_Image description is not a capability included in the DALL-E model, therefore, it is not a use case that can be implemented by using DALL-E, while the other three capabilities are offered by DALL-E in Azure OpenAI._

Which generative AI model is used to generate images based on natural language prompts?
**DALL-E**
Embeddings
GPT-3.5
GPT-4
Whisper
_DALL-E is a model that can generate images from natural language. GPT-4 and GPT-3.5 can understand and generate natural language and code but not images. Embeddings can convert text into numerical vector form to facilitate text similarity. Whisper can transcribe and translate speech to text._

xxx can search, classify, and compare sources of text for similarity.
Data grounding
**Embeddings**
Machine learning
System messages
_Embeddings is an Azure OpenAI model that converts text into numerical vectors for analysis. Embeddings can be used to search, classify, and compare sources of text for similarity._

What is the first step in the statistical analysis of terms in a text in the context of natural language processing (NLP)?
creating a vectorized model
counting the occurrences of each word
encoding words as numeric features
**removing stop words**
_Removing stop words is the first step in the statistical analysis of terms used in a text in the context of NLP. Counting the occurrences of each word takes place after stop words are removed. Creating a vectorized model is not part of statistical analysis. It is used to capture the sematic relationship between words. Encoding words as numeric features is not part of statistical analysis. It is frequently used in sentiment analysis._

What is the confidence score returned by the Azure AI Language detection service of natural language processing (NLP) for an unknown language name?

Select only one answer.
1
-1
**NaN**
Unknown
_NaN, or not a number, designates an unknown confidence score. Unknown is a value with which the NaN confidence score is associated. The score values range between 0 and 1, with 0 designating the lowest confidence score and 1 designating the highest confidence score._

Which part of speech synthesis in natural language processing (NLP) involves breaking text into individual words such that each word can be assigned phonetic sounds?
lemmatization
key phrase extraction
**tokenization**
transcribing
_Tokenization is part of speech synthesis that involves breaking text into individual words such that each word can be assigned phonetic sounds. Transcribing is part of speech recognition, which involves converting speech into a text representation. Key phrase extraction is part of language processing, not speech synthesis. Lemmatization, also known as stemming, is part of language processing, not speech synthesis._

Which two Azure AI Services features can be used to enable both text-to-text and speech-to-text between multiple languages? Each correct answer presents part of the solution.
Conversational Language Understanding
key phrase extraction
language detection
**the Speech service**
**the Translator service**
_The Azure AI Speech service can be used to generate spoken audio from a text source for text-to-speech translation. The Azure AI Translator service directly supports text-to-text translation in more than 60 languages. Key phrase extraction, Conversational Language Understanding, and language detection are not used for language translation for text-to-text and speech-to-text translation._

Which two features of Azure AI Services allow you to identify issues from support question data, as well as identify any people and products that are mentioned? Each correct answer presents part of the solution.

Azure AI Bot Service
Conversational Language Understanding
**key phrase extraction**
**named entity recognition**
Azure AI Speech service
_Key phrase extraction is used to extract key phrases to identify the main concepts in a text. It enables a company to identify the main talking points from the support question data and allows them to identify common issues. Named entity recognition can identify and categorize entities in unstructured text, such as people, places, organizations, and quantities. The Azure AI Speech service, Conversational Language Understanding, and Azure AI Bot Service are not designed for identifying key phrases or entities._

Which feature of the Azure AI Language service includes functionality that returns links to external websites to disambiguate terms identified in a text?
**entity recognition**
key phrase extraction
language detection
sentiment analysis
_Entity recognition includes the entity linking functionality that returns links to external websites to disambiguate terms (entities) identified in a text. Key phrase extraction evaluates the text of a document and identifies its main talking points. Azure AI Language detection identifies the language in which text is written. Sentiment analysis evaluates text and returns sentiment scores and labels for each sentence._

For which two scenarios is the Universal Language Model used by the speech-to-text API optimized? Each correct answer presents a complete solution.
acoustic
**conversational**
**dictation**
language
pronunciation
_The Universal Language Model used by the speech-to-text API is optimized for conversational and dictation scenarios. The acoustic, language, and pronunciation scenarios require developing your own model._

Which type of translation does the Azure AI Translator service support?
speech-to-speech
speech-to-text
text-to-speech
**text-to-text**
_The Azure AI Translator service supports text-to-text translation, but it does not support speech-to-text, text-to-speech, or speech-to-speech translation._

Which three features are elements of the Azure AI Language Service? Each correct answer presents a complete solution.
Azure AI Vision
Azure AI Content Moderator
**Entity Linking**
**Personally Identifiable Information (PII) detection**
**Sentiment analysis**
_Entity Linking, PII detection, and sentiment analysis are all elements of the Azure AI Service for Azure AI Language. Azure AI Vision deals with image processing. Azure AI Content Moderator is an Azure AI Services service that is used to check text, image, and video content for material that is potentially offensive._

Which type of artificial intelligence (AI) workload provides the ability to generate bounding boxes that identify the locations of different types of vehicles in an image?
image analysis
image classification
optical character recognition (OCR)
**object detection**
_Object detection provides the ability to generate bounding boxes identifying the locations of different types of vehicles in an image. The other answer choices also process images, but their outcomes are different._

Which type of artificial intelligence (AI) workload provides the ability to classify individual pixels in an image depending on the object that they represent?
image analysis
image classification
object detection
**semantic segmentation**
_Semantic segmentation provides the ability to classify individual pixels in an image depending on the object that they represent. The other answer choices also process images, but their outcomes are different._

Which type of service provides a platform for conversational artificial intelligence (AI)?
**Azure AI Bot Service**
Azure AI Document Intelligence
Azure AI Vision
Azure AI Translator
_Azure AI Bot Service provide a platform for conversational artificial intelligence (AI), which designates the ability of software agents to participate in a conversation. Azure AI Translator is part of Natural language processing (NLP), but it does not serve as a platform for conversational AI. Azure AI Vision deals with image processing. Azure AI Document Intelligence extracts information from scanned forms and invoices._

Which type of artificial intelligence (AI) workload has the primary purpose of making large amounts of data searchable?
image analysis
**knowledge mining**
object detection
semantic segmentation
_Knowledge mining is an artificial intelligence (AI) workload that has the purpose of making large amounts of data searchable. While other workloads leverage indexing for faster access to large amounts of data, this is not their primary purpose._

Which artificial intelligence (AI) workload scenario is an example of natural language processing (NLP)?
**extracting key phrases from a business insights report**
identifying objects in landscape images
monitoring for sudden increases in quantity of failed sign-in attempts
predicting whether customers are likely to buy a product based on previous purchases
_Extracting key phrases from text to identify the main terms is an NLP workload. Predicting whether customers are likely to buy a product based on previous purchases requires the development of a machine learning model. Monitoring for sudden increases in quantity of failed sign-in attempts is a different workload. Identifying objects in landscape images is a computer vision workload._

Which two artificial intelligence (AI) workload scenarios are examples of natural language processing (NLP)? Each correct answer presents a complete solution.
extracting handwritten text from online images
generating tags and descriptions for images
monitoring network traffic for sudden spikes
**performing sentiment analysis on social media data**
**translating text between different languages from product reviews**
_Translating text between different languages from product reviews is an NLP workload that uses the Azure AI Translator service and is part of Azure AI Services. It can provide text translation of supported languages in real time. Performing sentiment analysis on social media data is an NLP that uses the sentiment analysis feature of the Azure AI Service for Language. It can provide sentiment labels, such as negative, neutral, and positive for text-based sentences and documents._

Which two artificial intelligence (AI) workload features are part of the Azure AI Vision service? Each correct answer presents a complete solution.

entity recognition
key phrase extraction
**optical character recognition (OCR)**
sentiment analysis
**spatial analysis**
_OCR and Spatial Analysis are part of the Azure AI Vision service. Sentiment analysis, entity recognition, and key phrase extraction are not part of the computer vision service._

Which principle of responsible artificial intelligence (AI) has the objective of ensuring that AI solutions benefit all parts of society regardless of gender or ethnicity?

accountability
**inclusiveness**
privacy and security
reliability and safety
_The inclusiveness principle is meant to ensure that AI solutions empower and engage everyone, regardless of criteria such as physical ability, gender, sexual orientation, or ethnicity. Privacy and security, reliability and safety, and accountability do not discriminate based on these criteria, but also do not emphasize the significance of bringing benefits to all parts of the society._

Which principle of responsible artificial intelligence (AI) is applied in the design of an AI system to ensure that users understand constraints and limitations of AI?
fairness
inclusiveness
privacy and security
**transparency**
_The transparency principle states that AI systems must be designed in such a way that users are made fully aware of the purpose of the systems, how they work, and which limitations can be expected during use. The inclusiveness principle states that AI systems must empower people in a positive and engaging way. Fairness is applied to AI systems to ensure that users of the systems are treated fairly. The privacy and security principle are applied to the design of AI systems to ensure that the systems are secure and to respect user privacy._

Which two principles of responsible artificial intelligence (AI) are most important when designing an AI system to manage healthcare data? Each correct answer presents part of the solution.
**accountability**
fairness
inclusiveness
**privacy and security**

_The accountability principle states that AI systems are designed to meet any ethical and legal standards that are applicable. The system must be designed to ensure that privacy of the healthcare data is of the highest importance, including anonymizing data where applicable. The fairness principle is applied to AI systems to ensure that users of the systems are treated fairly. The inclusiveness principle states that AI systems must empower people in a positive and engaging way._

