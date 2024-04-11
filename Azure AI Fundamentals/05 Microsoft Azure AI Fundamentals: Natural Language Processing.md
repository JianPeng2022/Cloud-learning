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




