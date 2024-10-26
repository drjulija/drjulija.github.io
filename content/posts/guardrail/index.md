---
title: "Hello"
summary: "Read about what is RAG and how to set up a Basic RAG Pipeline using Langchain, OpenAI LLM and ChromaDB. Sample code included."
date: 2024-01-07
series: ["Guardrail"]
weight: 1
aliases: ["/guardrail"]
tags: ["RAG", "LLMs", "Langchain", "OpenAI", "LlamaIndex", "ChromaDB"]
author: ["Dr Julija"]
cover:
    image: "/posts/guardrail/images/toxic-cover.png"  # image path/url
    alt: "Hello" # alt text
    caption: "Basic Rag Pipeline | 📔 DrJulija's Notebook | Follow my [Medium Blog](https://medium.com/p/938e4f6e03d1)" # display caption under cover
    relative: false # when using page bundles set this to true
---

## 📝 Introduction
I’ll share key insights from building a simple Guardrails model to classify toxic content. Guardrails play a crucial role in LLM-based applications by preventing models from generating harmful or undesirable content.

Recently, there has been significant research on leveraging large language models (LLMs) themselves to critique, judge, and classify harmful content using techniques like instruction-following (IF) and in-context learning (ICL). For example, the framework "Self-Criticism" proposed by [Tan, X. et. al., 2023](https://aclanthology.org/2023.emnlp-industry.62.pdf) allows LLMs to self-align to helpful, honest, and harmless (HHH) standards based on what they leant from extensive text corpus during training. 

Another example is Meta’s Llama Guard, a Llama2-7B model developed by [Inan et al., 2023](https://arxiv.org/pdf/2312.06674), which has been fine-tuned specifically for content safety classification.

In both approaches, LLMs are central to determining what qualifies as toxic content. However, this approach presents a few challenges:

1. Reliability – LLMs are not deterministic, meaning they can produce different outputs for the same input, which may impact consistency.
2. Efficiency – Deploying an LLM-based toxic content classifier in production can be costly and slow. For instance, using such classifier could add a delay of 2–4 seconds before displaying the final output to the end-user. This might negatively impact the end-user experience.

🤔 This made me wonder: do LLM-based toxic content classifiers truly outperform traditional neural network classifiers in accuracy?

The results were surprising!