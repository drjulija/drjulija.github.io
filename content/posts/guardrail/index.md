---
title: "LLM Guardrails: How I Built Toxic Content Classifier"
summary: "In this post I compare two classifiers that I built for toxic content classification: LLM-based classifier and Feed Forward Neural Network classifier. I found surprising results."
date: 2024-10-07
series: ["Guardrail"]
weight: 1
aliases: ["/guardrail"]
tags: ["Guardrails", "LLMs", "Llama3", "LlamaGuard"]
author: ["Dr Julija"]
cover:
    image: "/posts/guardrail/images/toxic-cover.png"  # image path/url
    alt: "Hello" # alt text
    caption: "Let's investigate if LLM-based toxic content classifiers are always better than old-school machine learning methods | 📔 DrJulija's Notebook | Follow my [Medium Blog](https://medium.com/p/938e4f6e03d1)" # display caption under cover
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

{{< figure src="/posts/guardrail/images/LlamaGuard.png" attr="Llama Guard Framework overview ([Dong, Y. et al. 2024](https://arxiv.org/html/2402.01822v1))" align=center target="_blank" >}}


**2. Nvidia NeMo** ([Rebedea et al., 2023](https://aclanthology.org/2023.emnlp-demo.40.pdf))

NeMo serves as an intermediary layer to enhance control and safety in LLM applications. When a customer’s prompt is received, NeMo converts it into an embedding vector, then applies a K-nearest neighbors (KNN) approach to match it with stored vectors representing standard user inputs (canonical forms). This retrieves the embeddings most similar to the prompt, which NeMo uses to guide output generation from the corresponding canonical form. Throughout this process, NeMo can use the LLM to produce safe responses as specified by the Colang program. Additionally, NeMo includes pre-built moderation tools, such as fact-checking, hallucination prevention, and content moderation, to further safeguard outputs.

{{< figure src="/posts/guardrail/images/nemo.png" attr="Nvidia Nemo Guardrails Framework ([Dong, Y. et al. 2024](https://arxiv.org/html/2402.01822v1))" align=center target="_blank" >}}


## Objective
The goal of this work was to examine whether LLM-based toxic content classifiers genuinely surpass traditional neural network classifiers in terms of accuracy and by how much.

## Methodology
I evaluated the performance of two binary classifiers:
1. Llama3 8B with in-context-learning (ICL)
2. Two Layer Neural Network - a feed-forward neural network trained on [Wikipedia Toxic Comments](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic) training dataset.

### Dataset
For this experiment, I used the [Wikipedia Toxic Comments](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic) dataset, which includes toxic comments and hate speech from Wikipedia, with each comment labeled by humans as either 1 ("toxic") or 0 ("safe").

To support training, validation, and testing, I divided the dataset into three balanced subsets that contains equal number of samples for each category 0 or 1:
1. Training Dataset - 25,868 samples
2. Validation Dataset - 6,582 samples
3. Test Dataset - 3,000 samples

The training and validation datasets were used to train a neural network classifier.

For evaluating the performance of all three classifiers, I used the Test dataset.

### Classifiers

For the experiment, I set up two different classifiers: 

#### Llama3 8B with ICL
I used Meta's Llama3 model to classify toxic content using Test Dataset. Using in-context-learning the LLM is tasked to classify `user comment` as toxic or safe by returning 0 for "safe" or 1 for "toxic" content. If LLM can not return the answer or does not know, it should return 2. I used similar prompt structure and toxic content categories as per [Inan et al., 2023](https://arxiv.org/pdf/2312.06674) paper. Below is an example of such prompt.


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


#### GPT 3.5 Turbo with ICL
I used OpenAI's GPT 3.5 Turbo model to classify toxic content using Test Dataset. Same as per above, I used in-context-learning technique where LLM is tasked to classify `user comment` as toxic or safe by returning 0 for "safe" or 1 for "toxic" content. If LLM can not return the answer or does not know, it should return 2. I used the same prompt (see above) in both Llama3 and GPT 3.5 classifiers.
-->

#### Feed-forward Neural Network
I train a simple 2 layer neural network with the following architecture:
- Input layer: 1024
- Hidden layer 1: 100
- Hidden layer 2 (with dropout): 25
- Output: 2

Below code shows the neural network architecture

```python
class NeuralNet(nn.Module):
    """
    Feed forward Neural Network architecture
    """
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(NeuralNet, self).__init__()
        
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size
        
        self.i2h = nn.Linear(self.input_size, self.hidden1)
        self.h2h = nn.Linear(self.hidden1, self.hidden2)
        self.h2o = nn.Linear(self.hidden2, self.output_size)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = F.relu(self.i2h(x))
        x = F.relu(self.dropout(self.h2h(x)))
        x = self.h2o(x)
        return x
```

For each data sample (user comment), I generated embeddings using the [mGTE](https://arxiv.org/pdf/2407.19669) sentence embedding model developed by Alibaba Group, which is accessible [here](https://arxiv.org/pdf/2407.19669).

I use Cross Entropy Loss and Stochastic Gradient Descent optimizaton. 

Below figure shows the training and validation loss for each epoch during training.

{{< figure src="/posts/guardrail/images/nn_1024_100_25_loss.png" attr="Training and validation loss during Neural Network training" align=center target="_blank" >}}

After the training, the performance of the neural network was evaluated on Test Dataset.

## Results

Here is the most interesting part.

{{< figure src="/posts/guardrail/images/shock.gif" align=center target="_blank" >}}


**Llama3 7B with ICL**

The LLM failed to classify 146 samples. I updated their labels to 1 assuming that we want a model with high recall score. To classify 3,000 test samples it took me more than an hour. 

Below is the summary of the model perfomance:
- Accuracy Score:  0.8
- Precision:  0.82
- Recall:  0.78
- F1 Score:  0.8

**Feed-forward Neural Network**

Neural network classified all 3,000 test samples and it took a few minutes.

Below is the summary of the model perfomance:
- Accuracy Score:  0.9
- Precision:  0.86
- Recall:  0.96
- F1 Score:  0.91


<!---
![alt-text-1](/posts/guardrail/images/cm.png "title-1")
-->

Below, are confusion matrices for both classifiers. 

{{< figure src="/posts/guardrail/images/cm.png" attr="Confusion matrices" align=center target="_blank" >}}

In this scenario, the neural network model significantly outperformed the LLM-based classifier across all evaluation metrics. 

For a toxic content classification system, achieving a high recall rate is essential to ensure maximum detection of harmful content. Consequently, the neural network—with a recall rate of 0.96—would be preferable to the LLM-based classifier, which achieved a recall rate of only 0.78.

Additionally, in a production environment, the neural network would offer faster processing speeds, taking less than a second per request, whereas the LLM requires approximately 2-3 seconds for each classification.

However, it is worth noting that the neural network may not generalize as effectively to novel, unseen content, where the LLM could potentially offer an advantage.


## Limitations and next steps
To implement the Llama3.1 8B model, I utilized the Ollama framework, a streamlined tool for running LLMs on local machines. Due to quantization, model performance may have been significantly affected. The next phase involves conducting the same experiment with the full Llama3.1 model on AWS Bedrock. Additionally, I plan to test the LlamaGuard model under similar conditions.

For further refinement, the neural network could be trained on a broader range of toxic content types, and alternative architectures and embedding models could be explored.

An additional experiment using OpenAI’s GPT-3.5 API was attempted; however, after approximately 200 requests, the API returned an error citing "repetitive patterns in the prompt."


## Conclusion
This study compared the performance of a large language model (Llama3 8B with in-context learning) and a traditional two layer neural network classifier for detecting toxic content. Results indicate that the feed-forward neural network significantly outperformed the LLM-based classifier across key metrics, achieving an accuracy score of 0.9 and a recall rate of 0.96, while the LLM-based classifier achieved an accuracy of 0.8 and a recall of 0.78. Furthermore, the neural network demonstrated faster inference times, making it a more suitable choice for real-time or high-volume production environments.

These findings highlight that for tasks requiring high recall in content moderation, traditional neural network classifiers may provide superior performance with a reduced computational burden. However, the neural network may be less capable of handling novel or more nuanced toxic content, where the LLM’s generalization capabilities could provide an advantage.

Future work will involve testing a non-quantized version of Llama3.1 on AWS Bedrock to assess any improvements in performance. Additionally, exploring the LlamaGuard model and experimenting with more diverse training data and alternative architectures will help refine the neural network classifier further. 

## 🔗 Code
Full code can be found [here](https://github.com/drjulija/guardrails/tree/main?tab=readme-ov-file)
