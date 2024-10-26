---
title: "How I Built Toxic Content Classifiers for LLM Guardrails"
summary: "In this post I compare two classifiers that I built for toxic content classification: LLM-based classifier and Feed Forward Neural Network classifier. I found surprising results."
date: 2024-09-25
series: ["Classifier"]
weight: 1
aliases: ["/toxic-content-classifier"]
tags: ["Neural Network", "LLMs", "Supervised Learning", "Toxic Content Classification", "Guardrails"]
author: ["Dr Julija"]
ShowToc: true #table of content
TocOpen: true #table of content open by default
cover:
    image: "/posts//toxic-content-classifier/images/toxic-cover.png" 
    alt: "Hello" # alt text
    caption: "Are LLM-based toxic content classifiers always better than "old-school" machine learning methods? | 📔 DrJulija's Notebook | Follow my [Medium Blog](https://medium.com/p/938e4f6e03d1)" # display caption under cover
    relative: false # when using page bundles set this to true
---


note: (CTRL + Shift + V)
"/posts/rag/images/rag-evolution-sketch.png" 

Plan
1. Overview - the goal of this work and what i found
2. Background - what is guardrail, current research and methods
3. Experiment overview and assumption
4. Methodology
    - Data
    - Classifiers: LLM and Neural Network
    - Results - evaluation
5. Limitations 
6. Conclution


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

{{< figure src="/posts/toxic-content-classifier/images/LlamaGuard.png" attr="Llama Guard Framework overview ([Dong, Y. et al. 2024](https://arxiv.org/html/2402.01822v1))" align=center target="_blank" >}}

**2. Nvidia NeMo** ([Rebedea et al., 2023](https://aclanthology.org/2023.emnlp-demo.40.pdf))

NeMo serves as an intermediary layer to enhance control and safety in LLM applications. When a customer’s prompt is received, NeMo converts it into an embedding vector, then applies a K-nearest neighbors (KNN) approach to match it with stored vectors representing standard user inputs (canonical forms). This retrieves the embeddings most similar to the prompt, which NeMo uses to guide output generation from the corresponding canonical form. Throughout this process, NeMo can use the LLM to produce safe responses as specified by the Colang program. Additionally, NeMo includes pre-built moderation tools, such as fact-checking, hallucination prevention, and content moderation, to further safeguard outputs.

{{< figure src="/posts/toxic-content-classifier/images/nemo.png" attr="Nvidia Nemo Guardrails Framework ([Dong, Y. et al. 2024](https://arxiv.org/html/2402.01822v1))" align=center target="_blank" >}}


## Objective
The goal of this work was to examine whether LLM-based toxic content classifiers genuinely surpass traditional neural network classifiers in terms of accuracy.

## Methodology
I evaluated the performance of three binary classifiers:
1. Llama3 7B with in-context-learning (ICL)
2. Llama Guard 3 7B 
3. Two Layer Neural Network - a feed-forward neural network trained on [Wikipedia Toxic Comments](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic) training dataset.

### Dataset
In this experiment, I used the [Wikipedia Toxic Comments](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic) dataset, which includes toxic comments and hate speech from Wikipedia, with each comment labeled by humans as either 1 ("toxic") or 0 ("safe").

To support training, validation, and testing, I divided the dataset into three balanced subsets:
1. Training Dataset - 25,868 samples
2. Validation Dataset - 6,582 samples
3. Test Dataset - 3,000 samples

The training and validation datasets were used to train a neural network classifier. For each sample, I generated embeddings using the [mGTE](https://arxiv.org/pdf/2407.19669) sentence embedding model developed by Alibaba Group, which is accessible [here](https://arxiv.org/pdf/2407.19669).

For evaluating the performance of all three classifiers, I used Test dataset.

### Classifiers

For the experiment, I set up 3 different classifiers: 


#### Llama3 7B with ICL
I used Llama3 model to classify toxic content using Test Dataset. Using in-context-learning the LLM is tasked to classify `user comment` as toxic or safe by returning 0 for "safe" or 1 for "toxic" content. If LLM can not return the answer or does not know it should return 2. I use similar prompt structure and toxic content categories as per [Inan et al., 2023](https://arxiv.org/pdf/2312.06674) paper. Below is an example of a prompt.


```python
"""
Task: Check if there is toxic content in 'User' comment according to our toxic content categories below. 
               
<BEGIN TOXIC CONTENT CATEGORIES>
Violence and Hate
Sexual Content
Guns & Illegal Weapons
Regulated or Controlled Substances
Suicide and Self Harm
Criminal Planning 
<END TOXIC CONTENT CATEGORIES>
               
<BEGIN USER COMMENT>
{USER_COMMENT}
<END USER COMMENT>
               
Provide your safety assesment for the above user comment. 
Only provide a single number:
1 if user comment is "Toxic" or 
0 if user comment is "Safe" 
Do not provide explanation or justification. If you can not provide the answer output number 2. You are not allowed to output non numeric answer. You can only output 0, 1 or 2."""
```
<!---
#### Llama Guard 3 7B
I used Llama Guard 3 model to classify toxic content using Test Dataset. Because Llama Guard is already fine-tuned to perform a classification task, I did not provide any prompt to the model. Llama Guard outputs "safe" for non-toxic content or "unsafe" for toxic content together with representing category (refer to the [paper](https://arxiv.org/pdf/2312.06674) for more details).
-->

#### Feed-forward Neural Network
I train a simple 2 layer neural network with the following architecture:
- Input layer: 1024
- Hidden layer 1: 100
- Hidden layer 2 (with dropout): 25
- Output: 2

I use Cross Entropy Loss and Stochastic Gradient Descent optimizaton. 

Below figure shows the training and validation loss for each epoch during training.

{{< figure src="/posts/toxic-content-classifier/images/nn_1024_100_25_loss.png" attr="Training and validation loss during Neural Network training" align=center target="_blank" >}}

After the training, the performance of the neural network was evaluated on Test Dataset.

Full code is accessible here.

## Evaluation

Here is the most interesting part.
{{< figure src="/posts/toxic-content-classifier/images/shock.gif" align=center target="_blank" >}}



## Limitations


## Conclution

## 🔗 Code
Can be found here
