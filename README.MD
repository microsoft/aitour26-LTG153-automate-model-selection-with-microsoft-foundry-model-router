<p align="center">
<img src="img/Banner-MS-AI-Tour-26.png" alt="decorative banner" width="1200"/>
</p>

# [Microsoft AI Tour 2026](https://aitour.microsoft.com)

## LTG153 : Automate model selection with Microsoft Foundry Model Router 

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://aka.ms/AIFoundryDiscord-AITour26)
[![Microsoft Foundry Developer Forum](https://img.shields.io/badge/GitHub-Microsoft_Foundry_Developer_Forum-blue?style=for-the-badge&logo=github&color=adff2f&logoColor=fff)](https://aka.ms/AIFoundryForum-AITour26)

If you will be delivering this session, check the [session-delivery-sources](./session-delivery-resources/) folder for slides, scripts, and other resources.

### Session Description

> Full Title: <br/> **Automate model selection with Azure AI Foundry Model Router - to fast-track cost efficiencies and win big**

As 85% of enterprises embrace multi-model AI strategies, developers face growing complexity in selecting, deploying, and managing the right models for each task. In this session, discover how the model router in Azure AI Foundry intelligently automates model selection based on performance, cost, and latency. This intelligent orchestration layer reduces operational overhead, accelerates deployment, and unlocks scalable AI operations across your organization. Say goodbye to model fatigue and let the router do the strategic lifting. 


### 🧠 Learning Outcomes

By the end of this session, learners will be able to:

- Describe what Model Router does, and why that helps
- Deploy and use a Model Router in Azure AI Foundry
- Deploy a benchmark model - and compare their results
- Analyze the cost and latency optimization achieved

### 💻 Technologies Used

1. **Azure AI Foundry** - Unified end-to-end AI development platform 
1. **Model Router** - My primary model deployment (smart router)
1. **Azure OpenAI Service** - My Benchmark model deployment (GPT-5) 
1. **React + Vite** - Building the frontend UI/UX
1. **Flask + Python** - Building the backend API server

### 🔗 Session Resources
| Resources          | Links                             | Description        |
|:-------------------|:----------------------------------|:-------------------|
| Model Router Card | [Azure AI Foundry Catalog](https://ai.azure.com/catalog/models/model-router) | Supported Models, Overview |
| Model Router Concepts| [Microsoft Learn Docs](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/model-router) | Benefits & Limitations |
| Model Router Quickstart| [Microsoft Learn Docs](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/model-router) | Usage in Azure AI Foundry |
| GPT-5 in Azure AI Foundry | [Microsoft Azure Video](https://www.youtube.com/watch?v=xgdz-KCFpFI&pp=ygUhbW9kZW0gcm91dGVyIGZvciBhenVyZSBhaSBmb3VuZHJ5) | Featuring Model Router |
| Additional Resources for this session | [https://learn.microsoft.com](https://learn.microsoft.com?ocid=AITour26_resourcecenter_cnl) | Links specific to this session |

### 📚 Continued Learning Resources
| Resources          | Links                             | Description        |
|:-------------------|:----------------------------------|:-------------------|
| AI Tour 2026 Resource Center | [https://aka.ms/AITour26-Resource-Center](https://aka.ms/AITour26-Resource-Center?ocid=AITour26_resourcecenter_cnl) | Links to all repos for AI Tour 26 Sessions |
| Microsoft Foundry Community Discord | [![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://aka.ms/AIFoundryDiscord-AITour26)| Connect with the MIcrosoft Foundry Community! |
| Learn at AI Tour | [https://aka.ms/LearnAtAITour](https://aka.ms/LearnAtAITour?ocid=AITour26_resourcecenter_cnl) | Continue learning on Microsoft Learn |

### 🌐 Multi-Language Support

Additional Languages Coming Soon

## Content Owners

<!-- TODO: Add yourself as a content owner
1. Change the src in the image tag to {your github url}.png
2. Change INSERT NAME HERE to your name
3. Change the github url in the final href to your url. -->


<table>
<tr>
    <td align="center"><a href="http://github.com/sanjeev3">
        <img src="https://github.com/sanjeev3.png" width="100px;" alt="SANJEEV JAGTAP"
"/><br />
        <sub><b> SANJEEV JAGTAP
</b></sub></a><br />
            <a href="https://github.com/sanjeev3" title="talk">📢</a> 
    </td>
    <td align="center"><a href="http://github.com/nitya">
        <img src="https://github.com/nitya.png" width="100px;" alt="NITYA NARASIMHAN
"/><br />
        <sub><b>NITYA NARASIMHAN
</b></sub></a><br />
            <a href="https://github.com/nitya" title="talk">📢</a> 
    </td>
    <td align="center"><a href="http://github.com/luca-stamatescu">
        <img src="https://github.com/luca-stamatescu.png" width="100px;" alt="LUCA STAMATESCU"
"/><br />
        <sub><b>LUCA STAMATESCU
</b></sub></a><br />
            <a href="https://github.com/luca-stamatescu" title="talk">📢</a> 
    </td>
</tr></table>

## 

## Responsible AI 

Microsoft is committed to helping our customers use our AI products responsibly, sharing our learnings, and building trust-based partnerships through tools like Transparency Notes and Impact Assessments. Many of these resources can be found at [https://aka.ms/RAI](https://aka.ms/RAI).
Microsoft’s approach to responsible AI is grounded in our AI principles of fairness, reliability and safety, privacy and security, inclusiveness, transparency, and accountability.

Large-scale natural language, image, and speech models - like the ones used in this sample - can potentially behave in ways that are unfair, unreliable, or offensive, in turn causing harms. Please consult the [Azure OpenAI service Transparency note](https://learn.microsoft.com/legal/cognitive-services/openai/transparency-note?tabs=text) to be informed about risks and limitations.

The recommended approach to mitigating these risks is to include a safety system in your architecture that can detect and prevent harmful behavior. [Azure AI Content Safety](https://learn.microsoft.com/azure/ai-services/content-safety/overview) provides an independent layer of protection, able to detect harmful user-generated and AI-generated content in applications and services. Azure AI Content Safety includes text and image APIs that allow you to detect material that is harmful. Within Azure AI Foundry portal, the Content Safety service allows you to view, explore and try out sample code for detecting harmful content across different modalities. The following [quickstart documentation](https://learn.microsoft.com/azure/ai-services/content-safety/quickstart-text?tabs=visual-studio%2Clinux&pivots=programming-language-rest) guides you through making requests to the service.

Another aspect to take into account is the overall application performance. With multi-modal and multi-models applications, we consider performance to mean that the system performs as you and your users expect, including not generating harmful outputs. It's important to assess the performance of your overall application using [Performance and Quality and Risk and Safety evaluators](https://learn.microsoft.com/azure/ai-studio/concepts/evaluation-metrics-built-in). You also have the ability to create and evaluate with [custom evaluators](https://learn.microsoft.com/azure/ai-studio/how-to/develop/evaluate-sdk#custom-evaluators).

You can evaluate your AI application in your development environment using the [Azure AI Evaluation SDK](https://microsoft.github.io/promptflow/index.html). Given either a test dataset or a target, your generative AI application generations are quantitatively measured with built-in evaluators or custom evaluators of your choice. To get started with the azure ai evaluation sdk to evaluate your system, you can follow the [quickstart guide](https://learn.microsoft.com/azure/ai-studio/how-to/develop/flow-evaluate-sdk). Once you execute an evaluation run, you can [visualize the results in Azure AI Foundry portal ](https://learn.microsoft.com/azure/ai-studio/how-to/evaluate-flow-results).
