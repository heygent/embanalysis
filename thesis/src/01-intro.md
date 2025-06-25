# Introduction

This work started with a simple premise: why are LLMs bad at math?

This is not really a hard question to answer. Most of the LLMs to date are not built with that purpose in mind, and can rely on tool calling to give good answers to quantitative and numerical questions.

There is a tremendous investment in computing resources that is directed towards arithmetic operations that make up the inner workings of LLMs, computations that the LLMs themselves aren't capable of leveraging to answer arithmetic questions. It feels like witnessing a fundamental disconnection, where the LLM is segregated from the capabilities that make its own functioning possible.

Savant syndrome is a very rare disorder. It manifests primarily in people with autism spectrum disorders [@murray2010] or after traumatic episodes. The people affected by it possess extraordinary qualities in certain areas, like arts, music or mathematics, while usually showing significant impairment in others. One of the possible areas in which savants may show exceptional aptitude is calculation: calendrical savants are able to instantly know the day of the week of dates far in the future. Such a skill seems unlikely to be the product of algorithmic calculation, so alternative hypotheses emerged.

What I propose here is that the Savant condition can be seen as a parallel to the bridging of this capabilities gap in LLMs. In particular, what is taken in consideration here is the use of concrete representations as described in [@murray2010], where abstract numerical concepts are transformed into "highly accessible concrete representations" that can be directly manipulated rather than computed through algorithmic steps. This reification process—the conversion of abstract concepts into concrete entities—appears to provide savants with immediate access to numerical relationships that would otherwise require complex calculations.

This is not meant necessarily to give a comprehensive explanation of the phenomenon on an empirical basis, as that would be hard to establish from the basis of current knowledge about both savant cognition and neural network representations. Rather, it serves as a conceptual framework for exploring whether similar representational advantages can be induced in artificial systems.

This idea is explored in two ways:

- by a literature review, that is meant to clarify what can function as concrete representations in this context
- by an exploration of numerical embeddings, that is meant to show whether the learned representation of current language models already tends to conform to certain geometrical objects or structures. We show that there is remarkable structure and patterns in the learned representation of current LLMs.

![The dog is happy because the graphicx package has been included correctly.](src/res/dog.jpeg)

<!--
## About numerical representation in LLMs

Large Language Models, which as of today predominantly use the Transformer architecture, consist of <brief description of the transformer architecture that ties into tokenization> an initial layer of embeddings that contain the learned vector representation of individual tokens. Tokenization as a process has a significative inductive bias [@singh2024] that can lead to an improvement or worsening of performance in arithmetic tasks. While GPT-2 used to have a purely BPE frequency-based approach on number tokenization, which leads to the tokenization of the most statistically prevalent numbers <?>, modern models either tokenize digits separately (so as $'1234' \rightarrow [1, 2, 3, 4]$), or hardcode certain integer ranges (ex. 0-999) to be encoded as single tokens <?>.

The latter approach is what is taken into consideration into the analytical part of this work, as it allows examining what representation do LLMs use to represent the numbers in that range.

There have been proposed approaches in the literature that aim at maximizing the inductive bias in the representation by having embeddings that are computed based on the number to be represented. his fits very well with the idea of reification: the representation is no longer just a representation, but it has properties of the object that it represents. This can lead to symbolic representation that are directly fungible for the desired computations<?>.

<span class="free">
It's fascinating to observe that a case study of a Savant patient, DT [@murray2010], has been reported of having a mathematical landscape that has very similar characteristics:

- Has sequence-space synesthesia with a "mathematical landscape" containing numbers 0-9999
- Each number has specific colors, textures, sizes, and sometimes movements or sounds
- Prime numbers have special object properties that distinguish them from other numbers
- Arithmetic calculations happen automatically - solutions appear as part of his visual landscape without conscious effort
- fMRI studies showed that even unstructured number sequences had visual structure for DT
</span>

In [@mottron2006], the hypothesis is also that the capabilities of the savant might come from privileged access to lower-level perceptual processing systems that have been functionally re-dedicated to symbolic material processing. This suggests that mathematical savants may bypass high-level algorithmic reasoning entirely, instead leveraging perceptual mechanisms that can directly recognize patterns in numerical relationships - much like how we might instantly recognize a face without consciously processing its individual features. There are also arguably similar mechanisms already implemented in LLMs, although usually employed in the context of <?> gradient normalization, in the form of skip connections.

-->