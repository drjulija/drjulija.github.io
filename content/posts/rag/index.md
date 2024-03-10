---
title: "Evolution of RAG-based Systems: Naive RAG, Advanced RAG & Modular RAG"
summary: "Read about what is RAG and how to set up a Basic RAG Pipeline using Langchain, OpenAI LLM and ChromaDB. Sample code included."
date: 2024-02-15
series: ["RAG"]
weight: 1
aliases: ["/evolution-of-rag"]
tags: ["RAG", "LLMs", "Naive RAG", "Advanced RAG", "Modular RAG"]
author: ["Dr Julija"]
ShowToc: true #table of content
TocOpen: true #table of content open by default
cover:
    image: "/posts/rag/images/rag-evolution-sketch.png"  # image path/url
    alt: "Hello" # alt text
    caption: "Three RAG Paradigms | 📔 DrJulija's Notebook | Follow my [Medium Blog](https://medium.com/p/938e4f6e03d1)" # display caption under cover
    relative: false # when using page bundles set this to true
---

## 📝 Overview
Here I describe my key learnings on how RAG systems evolved over the last few years. I share the differences between Naive RAG, Advanced RAG and Modular RAG frameworks. I summarize key insights from a great RAG technology survey paper [Gao et al. 2024](https://arxiv.org/abs/2312.10997).


## 🛠 What is a RAG Framework?

Large Language Models (LLMs) such as the [GPT](https://arxiv.org/abs/2005.14165) series from [OpenAI](https://openai.com/), [LLama](https://arxiv.org/abs/2307.09288) series by [Meta](https://ai.meta.com/research/), and [Gemini](https://arxiv.org/abs/2312.11805) by [Google](https://ai.google/) have achieved significant achievements in the generative AI field. 

But these models are non deterministic. Often, LLMs may produce content that is either inaccurate or irrelevant (known as hallucinations), rely on outdated information, and their decision-making processes are not transparent, leading to black-box reasoning.

Retrieval-Augmented Generation (RAG) framework is designed to help mitigate these challenges. RAG enhances LLMs’ knowledge base with additional, domain-specific data. 

For example, RAG-based systems are used in advanced question-answering (Q&A) applications - chatbots. To create a chatbot that can understand and respond to queries about private or specific topics, it's necessary to expand the knowledge of LLMs with the particular data needed. This is where the RAG can help.

🔗 **Read about how I built a Naive RAG pipeline [here](/posts/basic-rag/)** .

---

## 👩🏻‍💻 Naive RAG, Advanced RAG & Modular RAG

RAG framework addresses the following questions: 
- “What to retrieve”
- “When to retrieve”
- “How to use the retrieved information”

Over the last few years there has been a lot of research and innovation in the RAG space. **RAG systems can be split into 3 categories**:

- [Naive RAG](/posts/basic-rag/)
- Advanced RAG
- Modular RAG

See the comparison between all three paradigms of RAG - Naive RAG, Advanced RAG and Modular RAG below.

{{< figure src="/posts/rag/images/rag-evolution.png" attr="Comparison between the three paradigms of RAG ([Gao et al. 2024](https://arxiv.org/abs/2312.10997))" align=center target="_blank" >}}

---

## 1️⃣ Naive RAG

The Naive RAG pipeline consists of the below key phases:

1. **Data Indexing**
    1. **Data Loading:** This involves importing all the documents or information to be utilized.
    2. **Data Splitting:** Large documents are divided into smaller pieces, for instance, sections of no more than 500 characters each.
    3. **Data Embedding:** The data is converted into vector form using an embedding model, making it understandable for computers.
    4. **Data Storing:** These vector embeddings are saved in a vector database, allowing them to be easily searched.
2. **Retrieval**
When a user asks a question:
    1. The user's input is first transformed into a vector (query vector) using the same embedding model from the Data Indexing phase.
    2. This query vector is then matched against all vectors in the vector database to find the most similar ones (e.g., using the Euclidean distance metric) that might contain the answer to the user's question. This step is about identifying relevant knowledge chunks.
3. **Augmentation & Generation:** 
The LLM model takes the user's question and the relevant information retrieved from the vector database to create a response. This process combines the question with the identified data (augmentation) to generate an answer (generation).

### ✋ Problems with Naive RAG

Naive RAG faces challenges across all phases:

- **Retrieval** - failure to retrieve all relevant chunks or retrieving irrelevant chunks.
- **Augmentation** - challenges with integrating the context from retrieved chunks that may be disjointed or contain repetitive information.
- **Generation** - LLM may potentially generate answers that are not grounded in the provided context (retrieved chunks) or generate answers based on an irrelevant context that is retrieved.

---

## 2️⃣ Advanced RAG

Advanced RAG strategies have been developed to address the challenges faced by Naive RAG. Below is an overview of key Advanced RAG techniques.

RAG applications must efficiently retrieve relevant documents from the data source. But there are multiple challenges in each step.

1. How can we achieve accurate semantic representations of documents and queries?
2. What methods can align the semantic spaces of queries and documents (chunks)?
3. How can the retriever’s output be aligned with the preferences of the LLM?

Here I give an overview of pre-retrieval, retrieval and post-retrieval strategies:


### ➡️ Pre-Retrieval

How to optimize the data indexing?
- **Improve Data Quality** - remove irrelevant information, removing ambiguity in entities and terms, confirming factual accuracy, maintaining context, and updating outdated information.
- **Optimize Index Structure** - optimize chunk sizes to capture relevant context or add information from graph structure to capture relationships between entities.
- **Add Metadata** - add dates, chapters, subsections, purposes or any other relevant information into chunks as metadata to improve the data filtering


**Chunk Optimization** - when using external data sources / documents to build RAG pipeline, the initial step is break them down into smaller chunks to extract fine- grained features. Chunks are then embedded to represent their semantics. But embedding too large or too small text chunks may lead to sub-optimal outcome therefore we need to optimize chunk size for the types of documents we use in the RAG pipeline. 


📝 Summary of Key Pre-Retrieval Techniques:

**Sliding Window** - chunking method that uses overlap between the chunks.

**Auto-Merging Retrieval** - utilizes small text blocks during the initial search phase and subsequently provides larger related text blocks to the language model for processing.

**Abstract Embedding** - prioritizes Top-K retrieval based on document abstracts (or summaries), offering a comprehensive understanding of the entire document context.

**Metadata Filtering** - leverages document metadata to enhance the filtering process.

**Graph Indexing** - transforms entities and relationships into nodes and connections, significantly improving relevance.


### ➡️ Retrieval

Once the size of chunks is determined, the next step is to embed these chunks into the semantic space using an embedding model.

During the retrieval stage, the goal is to identify the most relevant chunks to query. This is done by calculating the similarity between the query and chunks. Here, we can optimize embedding models that are used to embed both the query and chunks.

**Domain Knowledge Fine-Tuning** - to ensure that an embedding model accurately captures domain-specific information of the RAG system, it is important to use domain-specific datasets for fine-tuning. The dataset for embedding model fine-tuning should contain: queries, a corpus and relevant documents.

**Similarity Metrics** - there are a number of different metrics to measure similarity between the vectors. The choise of the similarity metric is also an optimization problem. Vectori databases (ChromaDB, Pinecode, Weaviate...) support multiple similarity metrics. Here a few examples of different similarity metrics:
- Cosine Similarity
- Euclidean Distance (L2)
- Dot Product
- L2 Squared Distance
- Manhattan Distance


### ➡️ Post-Retrieval

After retrieving the context data (chunks) from a vector database, the next step is to merge the context with a query as an input into LLM. But some of the retrieved chunks may be repeated, noisy or contain irrelevant information. This may have an impact on how LLM processes the given context. 

Below I list a few strategies used to overcome these issues.

**Reranking** - rerank the retrieved information to prioritize the most relevant content first. LLMs often face performance declines when additional context is introduced, and reranking addresses this issue by reranking the retrieved chunks and identiying Top-K most relevant chunks that are then used as a context in LLM. Libraries such as [LlamaIndex](https://docs.llamaindex.ai/en/stable), [Langchain](https://python.langchain.com), HayStack offer different rerankers.


**Prompt Compression** - retrieved information might be noisy, its important to compress irrelevant context and reduce context length before presenting to LLM. Use Small Language Models to calculate prompt mutual information or perplexity to estimate element importance. Use summarization techniques when the context is long.

---

## 3️⃣ Modular RAG

Modular RAG integrates various modules and techniques from Adanced RAG to improve the overall RAG system. For example, incorporating a search module for similarity retrieval and applying a fine-tuning approach in the retriever. Modular RAG became a standard paradigm when building RAG applications. A few example of modules:

**Search Module** - in addition to retrieving context from vector database, search modules intergrates data from other sources such as search engines, tabular data, knowledge graphs etc.

**Memory Module** - adding memory component into RAG system where LLM can refer not only to the chunks retrieved from the vector database but also to the previous queries and answers that are stored in the systems memory.

**Fusion** - involves parallel vector searches of both original and expanded queries, intelligent reranking to optimize results, and pairing the best outcomes with new queries.

**Routing** - query routing decides the subsequent action to a user’s query for example summarization, searching specific databases, etc.




