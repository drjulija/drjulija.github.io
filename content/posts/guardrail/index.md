---
title: "How I Built Toxic Content Classifiers for LLM Guardrails"
summary: "In this post I compare two classifiers that I built for toxic content classification: LLM-based classifier and Feed Forward Neural Network classifier. I found surprising results."
date: 2024-10-07
series: ["Guardrail"]
weight: 1
aliases: ["/guardrail"]
tags: ["Guardrail", "LLMs", "Llama3", "LlamaGuard"]
author: ["Dr Julija"]
cover:
    image: "/posts/guardrail/images/toxic-cover.png"  # image path/url
    alt: "Hello" # alt text
    caption: "BAre LLM-based toxic content classifiers always better than old-school machine learning methods? | 📔 DrJulija's Notebook | Follow my [Medium Blog](https://medium.com/p/938e4f6e03d1)" # display caption under cover
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


## Background
This section provides an overview of Guardrails, their purpose, and current implementations.

### What are Guardrails?
Guardrails are filtering mechanisms in LLM-based applications that safeguard against generating toxic, harmful, or otherwise undesired content. They act as essential tools to mitigate risks associated with LLM use, such as ethical concerns, data biases, privacy issues, and overall robustness.

As LLMs become more widespread, the potential for misuse has grown, with risks ranging from spreading misinformation to facilitating criminal activities [Goldstein et al., 2023](https://arxiv.org/pdf/2301.04246).

In simple terms, a guardrail is an algorithm that reviews the inputs and outputs of LLMs and determines whether they meet safety standards.

For example, if a user’s input relates to child exploitation, a guardrail could either prevent the input from being processed by the LLM or adapt the output to ensure it remains harmless. In this way, guardrails intercept potentially harmful queries and help prevent models from responding inappropriately.

Depending on the application, guardrails can be customized to block various types of content, including offensive language, hate speech, hallucinations, or areas of high uncertainty. They also help ensure compliance with ethical guidelines and specific policies, such as fairness, privacy, or copyright protections [Dong, Y. et al. 2024](https://arxiv.org/html/2402.01822v1).

**Examples of Open-Source Guardrail Frameworks**

**1. Lama Guard** ([Inan et al., 2023](https://arxiv.org/pdf/2312.06674))

Llama Guard is a fine-tuned model that takes both input and output from an LLM and categorizes them based on user-specified criteria. While useful, its reliability can vary, as classification depends on the LLM’s understanding of the categories and predictive accuracy.

{{< figure src="/posts/guardrails/images/LlamaGuard.png" attr="Llama Guard Framework overview ([Dong, Y. et al. 2024](https://arxiv.org/html/2402.01822v1))" align=center target="_blank" >}}
