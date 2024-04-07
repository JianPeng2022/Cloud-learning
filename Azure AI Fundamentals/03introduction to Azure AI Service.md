Azure AI services are a portfolio of AI capabilities that unlock automation for workloads in language, vision, intelligent search, content generation, and much more. They are straightforward to implement and don’t require specialist AI knowledge.

Azure AI services are based on three principles that dramatically improve speed-to-market:
- Prebuilt and ready to use
- Accessed through APIs (REST API)
- Available on Azure

Azure AI services are cloud-based, and like all Azure services, you need to create a resource to use them. There are two types of AI service resources: multi-service and single-service. Your development requirements and how you want costs to be billed determine the types of resources you need.
- Multi-service resource: a resource created in the Azure portal that provides access to multiple Azure AI services with a single key and endpoint. Use the resource Azure AI services when you need several AI services or are exploring AI capabilities. When you use an Azure AI services resource, all your AI services are billed together.
- Single-service resources: a resource created in the Azure portal that provides access to a single Azure AI service, such as Speech, Vision, Language, etc. Each Azure AI service has a unique key and endpoint. These resources might be used when you only require one AI service or want to see cost information separately.

Understand authentication for Azure AI services: Most Azure AI services are accessed through a RESTful API, although there are other ways. The API defines what information is passed between two software components: the Azure AI service and whatever is using it. Part of what an API does is to handle authentication. Whenever a request is made to use an AI services resource, that request must be authenticated. The **endpoint** describes how to reach the AI service resource instance that you want to use, in a similar way to the way a URL identifies a website. 

After logging into one of the Azure studios, what is one task to complete to begin using the studio? 
- Input a key and endpoint into the studio
- Customize the API request.
- Associate a resource with the studio: To explore the capabilities of the service in the studio, you must first associate the resource with the studio.

What is an Azure AI services resource? 
- A bundle of several AI services in one resource: An Azure AI services resource is a bundle of several AI services in one resource.
- An AI service to recognize faces
- A single-service resource for Azure AI Search


