Microsoft Azure provides the **Azure Machine Learning service** - a cloud-based platform for creating, managing, and publishing machine learning models. Azure Machine Learning Studio offers multiple authoring experiences, such as:
- **Automated machine learning**: this feature enables non-experts to quickly create an effective machine learning model from data.
- **Azure Machine Learning Designer**: a graphical interface enabling no-code development of machine learning solutions.
- **Data metric visualization**: analyze and optimize your experiments with visualization.
- **Notebooks**: write and run your own code in managed Jupyter Notebook servers that are directly integrated in the studio.

**Computer vision**:
- Image classification: Image classification involves training a machine learning model to classify **images** based on their **contents**.
- Object detection: Object detection machine learning models are trained to classify individual objects **within an image**, and identify their location with a **bounding box**.
- Semantic segmentation: Semantic segmentation is an advanced machine learning technique in which **individual pixels**('mask') in the image are classified according to the object to which they belong. 
- Image analysis: You can create solutions that combine machine learning models with advanced image analysis techniques to extract information from images, including "tags" that could help catalog the image or even descriptive captions that summarize the scene shown in the image.
- Face detection, analysis, and recognition: Face detection is a specialized form of object detection that locates human faces in an image. This can be combined with classification and facial geometry analysis techniques to recognize individuals based on their facial features.
- Optical character recognition (OCR): Optical character recognition is a technique used to detect and read text in images. You can use OCR to read the text in photographs (for example, road signs or store fronts) or to extract information from scanned documents such as letters, invoices, or forms.

**Microsoft's Azure AI Vision** to develop computer vision solutions. The service features are available for use and testing in the Azure Vision Studio and other programming languages. Some features of Azure AI Vision include:
- Image Analysis: capabilities for analyzing images and video, and extracting descriptions, tags, objects, and text.
- Face: capabilities that enable you to build face detection and facial recognition solutions.
- Optical Character Recognition (OCR): capabilities for extracting printed or handwritten text from images, enabling access to a digital version of the scanned text.

**NLP** enables you to create software that can:
- Analyze and interpret text in documents, email messages, and other sources.
- Interpret spoken language, and synthesize speech responses.
- Automatically translate spoken or written phrases between languages.
- Interpret commands and determine appropriate actions.
**Microsoft's Azure AI Language** to build natural language processing solutions. Some features of Azure AI Language include understanding and analyzing text, training conversational language models that can understand spoken or text-based commands, and building intelligent applications. **Microsoft's Azure AI Speech **is another service that can be used to build natural language processing solutions. Azure AI Speech features include speech recognition and synthesis, real-time translations, conversation transcriptions, and more. You can explore Azure AI Language features in the **Azure Language Studio** and Azure AI Speech features in the **Azure Speech Studio**. The service features are available for use and testing in the studios and other programming languages.

**Document Intelligence** is the area of AI that deals with managing, processing, and using high volumes of a variety of data found in forms and documents. Document intelligence enables you to create software that can automate processing for contracts, health documents, financial forms, and more. You can use **Microsoft's Azure AI Document Intelligence** to build solutions that manage and accelerate data collection from scanned documents. Features of Azure AI Document Intelligence help automate document processing in applications and workflows, enhance data-driven strategies, and enrich document search capabilities. You can use prebuilt models to add intelligent document processing for invoices, receipts, health insurance cards, tax forms, and more. You can also use Azure AI Document Intelligence to create custom models with your own labeled datasets. The service features are available for use and testing in the **Document Intelligence Studio** and other programming languages.

**Knowledge mining** is the term used to describe solutions that involve extracting information from large volumes of often unstructured data to create a searchable knowledge store. One Microsoft knowledge mining solution is **Azure AI Search**, a private, enterprise, search solution that has tools for building indexes. The indexes can then be used for internal-only use, or to enable searchable content on public-facing internet assets. Azure AI Search can utilize the built-in AI capabilities of Azure AI services, such as image processing, document intelligence, and natural language processing to extract data. The product's AI capabilities make it possible to index previously unsearchable documents and to extract and surface insights from large amounts of data quickly.

**Generative AI** describes a category of capabilities within AI that create original content. People typically interact with generative AI that has been built into chat applications. Generative AI applications take in natural language input and return appropriate responses in a variety of formats including natural language, image, code, and audio. In Microsoft Azure, you can use the **Azure OpenAI service** to build generative AI solutions. Azure OpenAI Service is Microsoft's cloud solution for deploying, customizing, and hosting generative AI models. It brings together the best of OpenAI's cutting-edge models and APIs with the security and scalability of the Azure cloud platform. Azure OpenAI supports many foundation model choices that can serve different needs. The service features are available for use and testing in the **Azure OpenAI Studio** and other programming languages. You can use the Azure OpenAI Studio user interface to manage, develop, and customize generative AI models. 

| Challenge or Risk                      | Example                                                                                         |
|----------------------------------------|-------------------------------------------------------------------------------------------------|
| Bias can affect results                | A loan-approval model discriminates by gender due to bias in the data with which it was trained |
| Errors may cause harm                  | An autonomous vehicle experiences a system failure and causes a collision                       |
| Data could be exposed                  | A medical diagnostic bot is trained using sensitive patient data, which is stored insecurely    |
| Solutions may not work for everyone    | A home automation assistant provides no audio output for visually impaired users                |
| Users must trust a complex system      | An AI-based financial tool makes investment recommendations - what are they based on?           |
| Who's liable for AI-driven decisions?  | An innocent person is convicted of a crime based on evidence from facial recognition – who's responsible? |

**Responsible AI**: 
- Fairness 公平性在 AI 提供的上下文中是一个基本的社会与技术双重挑战，也就是说我们必须有足够多样化的人来开发和部署 AI 系统。团队在 AI 开发和部署生命周期中的每个阶段进行的假设和决定都可能带来偏见，因此这是一个很重要的主题。这不是我们能够委托给一两个人就撒手不管的事，而是每个人都必须非常主动地思考的问题。
- Reliability and safety 如果存在系统和模型可能犯错的情况，我们会对风险和损害进行量化，在深入了解这些风险和损害之后才会推出产品，并将我们了解到的情况告知用户。
- Privacy and security 当考虑这些人工智能系统的安全性时，你需要考虑数据从何而来、是如何到来的，如果是用户提交的数据，或是预测中使用的公共数据源，你如何防止数据被破坏，并配备异常检测或其他用于检测数据变化的系统，这些变化可能表明有对手试图影响系统结果。
- Inclusiveness 我们希望确保让所有社群都能受益。如果真正思考了如何能够为 3% 的特殊用户设计产品，那么我们同时也能为 97% 的普通用户解决问题。
- Transparency 透明度具有两面性；一方面，透明度意味着创建人工智能系统的人应该对他们使用人工智能的方式和原因持开放态度，同时也要对他们系统的局限性持开放态度。另一方面，透明度意味着人们应能够理解人工智能系统的行为，这就是人们常说的可解释性或可理解性。
- Accountability 虽然这些新模型和新技术的复杂性可能有些不可预测，并且在某些方面难以解释，但我们仍会为我们的技术对世界的影响负责。

