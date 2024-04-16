### Fundamentals of Text Analysis with the Language Service
Natural language processing might be used to create:
- A social media feed analyzer that detects sentiment for a product marketing campaign.
- A document search application that summarizes documents in a catalog.
- An application that extracts brands and company names from text.

Azure AI Language is a cloud-based service that includes features for understanding and analyzing text. Azure AI Language includes various features that support sentiment analysis, key phrase identification, text summarization, and conversational language understanding.

Tokenization:  "we choose to go to the moon". The phrase can be broken down into the following tokens, with numeric identifiers:
- 1 we
- 2 choose
- 3 to
- 4 go
- 5 the
- 6 moon

Notice that "to" (token number 3) is used twice in the corpus. The phrase "we choose to go to the moon" can be represented by the tokens [1,2,3,4,3,5,6].

Text normalization, Stop word removal, n-grams, Stemming

As the state of the art for NLP has advanced, the ability to train models that encapsulate the semantic relationship between tokens has led to the emergence of powerful language models. At the heart of these models is the encoding of language tokens as vectors (multi-valued arrays of numbers) known as **embeddings**.

### Azure AI Language
Azure AI Language is a part of the Azure AI services offerings that can perform advanced natural language processing over unstructured text. Azure AI Language's text analysis features include:
- Named entity recognition identifies people, places, events, and more. This feature can also be customized to extract custom categories.
- Entity linking identifies known entities together with a link to Wikipedia.
- Personal identifying information (PII) detection identifies personally sensitive information, including personal health information (PHI).
- Language detection identifies the language of the text and returns a language code such as "en" for English.
- Sentiment analysis and opinion mining identifies whether text is positive or negative.
- Summarization summarizes text by identifying the most important information.
- Key phrase extraction lists the main concepts from unstructured text.

**Conversational AI** describes solutions that enable a dialog between an AI agent and a human. Generically, conversational AI agents are known as bots. People can engage with bots through channels such as web chat interfaces, email, social media platforms, and more.

You can easily create a user support bot solution on Microsoft Azure using a combination of two core services:
- Azure AI Language: includes a custom question-answering feature that enables you to create a knowledge base of question and answer pairs that can be queried using natural language input.
- Azure AI Bot Service: provides a framework for developing, publishing, and managing bots on Azure.

When your bot is ready to be delivered to users, you can connect it to multiple channels; making it possible for users to interact with it through web chat, email, Microsoft Teams, and other common communication media.

To work with conversational language understanding, you need to take into account three core concepts: **utterances, entities, and intents**. 话语（Utterances）：指用户在与系统进行交互时所说的话或输入的文本。这些话语可能是简短的句子、问题、指令或者任何形式的语言输入。实体（Entities）：表示话语中的关键信息或具体对象。在对话中，实体可以是人名、地点、日期、产品名称等。通过识别和提取实体，系统可以更好地理解用户的意图和需求。意图（Intents）：指用户在话语中表达的意图或目的。

To use conversational language capabilities in Azure, you need a resource in your Azure subscription. You can use the following types of resource:
- Azure AI Language: A resource that enables you to build apps with industry-leading natural language understanding capabilities without machine learning expertise. You can use a language resource for **authoring and prediction**.
- Azure AI services: A general resource that includes conversational language understanding along with many other Azure AI services. You can only use this type of resource for **prediction**.
  
**Authoring**
After you've created an authoring resource, you can use it to train a conversational language understanding model. To train a model, start by defining the entities and intents that your application will predict as well as utterances for each intent that can be used to train the predictive model.

Conversational language understanding provides a comprehensive collection of prebuilt domains that include pre-defined intents and entities for common scenarios; which you can use as a starting point for your model. You can also create your own entities and intents.

When you create entities and intents, you can do so in any order. You can create an intent, and select words in the sample utterances you define for it to create entities for them, or you can create the entities ahead of time and then map them to words in utterances as you're creating the intents.

**Training the model**
After you have defined the intents and entities in your model, and included a suitable set of sample utterances; the next step is to train the model. Training is the process of using your sample utterances to teach your model to match natural language expressions that a user might say to probable intents and entities.

After training the model, you can test it by submitting text and reviewing the predicted intents. Training and testing is an iterative process. After you train your model, you test it with sample utterances to see if the intents and entities are recognized correctly. If they're not, make updates, retrain, and test again.

**Predicting**
When you are satisfied with the results from the training and testing, you can publish your Conversational Language Understanding application to a prediction resource for consumption.

Client applications can use the model by connecting to the endpoint for the prediction resource, specifying the appropriate authentication key; and submit user input to get predicted intents and entities. The predictions are returned to the client application, which can then take appropriate action based on the predicted intent.

### Fundamentals of Azure AI Speech 
To enable this kind of interaction, the AI system must support two capabilities:
- Speech recognition - the ability to detect and interpret spoken input
- Speech synthesis - the ability to generate spoken output

Speech recognition takes the spoken word and converts it into data that can be processed - often by transcribing it into text. The spoken words can be in the form of a recorded voice in an audio file, or live audio from a microphone. Speech patterns are analyzed in the audio to determine recognizable patterns that are mapped to words. To accomplish this, the software typically uses multiple models, including:
- An acoustic model that converts the audio signal into phonemes (representations of specific sounds).
- A language model that maps phonemes to words, usually using a statistical algorithm that predicts the most probable sequence of words based on the phonemes.

Microsoft Azure offers both speech recognition and speech synthesis capabilities through Azure AI Speech service, which includes the following application programming interfaces (APIs):
- The Speech to text API
- The Text to speech API

## Document intelligence
Document intelligence describes AI capabilities that support processing text and making sense of information in text. As an extension of optical character recognition (OCR), document intelligence takes the next step a person might after reading a form or document. It automates the process of extracting, understanding, and saving the data in text.

Azure AI Document Intelligence consists of features grouped by model type:
- Prebuilt models - pretrained models that have been built to process common document types such as invoices, business cards, ID documents, and more. These models are designed to recognize and extract specific fields that are important for each document type.
- Custom models - can be trained to identify specific fields that are not included in the existing pretrained models.
- Document analysis - general document analysis that returns structured data representations, including regions of interest and their inter-relationships.

### Fundamentals of Knowledge Mining and Azure AI Search 
Azure AI Search provides the infrastructure and tools to create search solutions that extract data from various structured, semi-structured, and non-structured documents.

Azure AI Search comes with the following features:
- Data from any source: accepts data from any source provided in JSON format, with auto crawling support for selected data sources in Azure.
- Full text search and analysis: offers full text search capabilities supporting both simple query and full Lucene query syntax.
AI powered search: has Azure AI capabilities built in for image and text analysis from raw content.
- Multi-lingual offers linguistic analysis for 56 languages to intelligently handle phonetic matching or language-specific linguistics. Natural language processors available in Azure AI Search are also used by Bing and Office.
- Geo-enabled: supports geo-search filtering based on proximity to a physical location.
- Configurable user experience: has several features to improve the user experience including autocomplete, autosuggest, pagination, and hit highlighting.

#### Use a skillset to define an enrichment pipeline
**Built-in skills** are based on pretrained models from Microsoft, which means you can't train the model using your own training data. Skills that call the Azure AI services APIs have a dependency on those services and are billed at the Azure AI services pay-as-you-go price when you attach a resource. Other skills are metered by Azure AI Search, or are utility skills that are available at no charge.

Built-in skills fall into these categories:
- Natural language processing skills: with these skills, unstructured text is mapped as searchable and filterable fields in an index.
- Image processing skills: creates text representations of image content, making it searchable using the query capabilities of Azure AI Search.

#### Understand indexes 没明白
An Azure AI Search index can be thought of as a container of searchable documents. Conceptually you can think of an index as a table and each row in the table represents a document. Tables have columns, and the columns can be thought of as equivalent to the fields in a document. Columns have data types, just as the fields do on the documents.

Use an indexer to build an index: In order to index the documents in Azure Storage, they need to be exported from their original file type to JSON. In order to export data in any format to JSON, and load it into an index, we use an indexer.

Azure AI Search lets you create and load JSON documents into an index with two approaches:
- **Push method**: JSON data is pushed into a search index via either the REST API or the .NET SDK. Pushing data has the most flexibility as it has no restrictions on the data source type, location, or frequency of execution.
- **Pull method**: Search service indexers can pull data from popular Azure data sources, and if necessary, export that data into JSON if it isn't already in that format.


