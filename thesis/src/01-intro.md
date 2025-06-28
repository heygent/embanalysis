# Introduction

This work started with a simple premise: why are LLMs bad at math?

This is not really a hard question to answer. Most of the LLMs to date are not built
with that purpose in mind, and can rely on tool calling to give good answers to
quantitative and numerical questions.

There is a tremendous investment in computing resources that is directed towards
arithmetic operations that make up the inner workings of LLMs, computations that the
LLMs themselves aren't capable of leveraging to answer arithmetic questions. It feels
like witnessing a fundamental disconnection, where the LLM is segregated from the
capabilities that make its own functioning possible.

Savant syndrome is a very rare disorder. It manifests primarily in people with autism
spectrum disorders [@murray2010] or after traumatic episodes. The people affected by it
possess extraordinary qualities in certain areas, like arts, music or mathematics, while
usually showing significant impairment in others. One of the possible areas in which
savants may show exceptional aptitude is calculation: calendrical savants are able to
instantly know the day of the week of dates far in the future. These skills are unlikely
to be the product of algorithmic calculation [@cowan2009], so alternative hypotheses
emerged.

What I propose here is that the Savant condition can be seen as a parallel to the
bridging of this capabilities gap in LLMs. In particular, what is taken in consideration
here is the use of concrete representations as described in [@murray2010], where
abstract numerical concepts are transformed into "highly accessible concrete
representations" that can be directly manipulated rather than computed through
algorithmic steps. This reification process - the conversion of abstract concepts into
concrete entities - appears to provide savants with immediate access to numerical
relationships that would otherwise require complex calculations.

This is not meant necessarily to give a comprehensive explanation of the phenomenon on
an empirical basis, as that would be hard to establish from the basis of current
knowledge about both savant cognition and neural network representations. Rather, it
serves as a conceptual framework for exploring whether similar representational
advantages can be induced in artificial systems.

This idea is explored in two ways:

- by a literature review, that is meant to clarify what can function as concrete
  representations in this context
- by an exploration of numerical embeddings, that is meant to show whether the learned
  representation of current language models already tends to conform to certain
  geometrical objects or structures. We show that there is remarkable structure and
  patterns in the learned representation of current LLMs.


# The Transformer architecture and vector representations

## The inductive bias of Tokenization

Modern LLMs are built on the Transformer architecture [@vaswani2023], which operates by
converting input text into sequences of discrete tokens that are then mapped to
high-dimensional vector representations. This initial tokenization step creates an
inductive bias that shapes how the model processes information [@ali2024] [@singh2024], with
significant implications for the application of the numerical data to arithmetical
tasks.

The most used algorithm for tokenization is currently Byte-Pair Encoding, which, given a
fixed vocabulary size, starts with individual characters and iteratively merges the most
frequently occurring pairs of adjacent tokens until the vocabulary limit is reached.
This process naturally creates longer tokens for common substrings that appear
frequently in the training data. For numbers, this means that frequently occurring
numerical patterns like "100", "2020", or "999" might become single tokens, while less
common numbers get broken into smaller pieces. The result is an idiosyncratic and
unpredictable tokenization scheme where similar numbers can be tokenized completely
differently based purely on their frequency in the training corpus. While GPT-2 used to
have a purely BPE tokenizer, the successive iteration of GPT and generally more recent
models either tokenize digits separately (so as $'1234' \rightarrow [1, 2, 3, 4]$), or
tokenize clusters of 3 digits, encompassing the integers in the range 0-999.

![GPT-2 number tokenization. Each row represents 100 numbers, yellow squares mean that
the number is represented by a single token, purple ones by multiple
[@millidgeGpt2]](src/res/gpt2_unique_tokens.png){width=500px}

Most of the tokenizers right now do L2R (left-to-right) clustering, meaning that a number such
as $12345$ would be divided in two tokens, $123$ and $45$. It has been shown
[@singh2024] that this kind of clustering leads to a lesser arithmetic performance, as
the grouping doesn't match the positional system's <way of calculating?>.
An even more surprising development is that forcing the R2L token clustering of numbers
in models already trained with L2R clustering through the use of commas in the input
(ex. $12,345$) leads to big improvements in arithmetic performance [@millidge]. Despite
the model learning representations adapted to work with a L2R token clustering strategy,
forcing a R2L clustering at inference time shows substantial improvements in arithmetic
tasks, which means that despite being learned through an unfavorable tokenization
approach, the numeric representations retain the properties that allow for the
performance to improve when the clustering scheme is corrected.

There can be different hypotheses on why this might be, for example:

- Arithmetic operations would still work locally in the 0-999 range, which allows for a
  correct reading on them and possible generalization on a larger scale.
- The forced tokenization also happens in the data, as numbers are often separated by
  punctuation in clusters of 3 digits, right to left, for legibility reasons
  [@singh2024]

Still, we are left with the fact that the learned representations work better for a
tokenization strategy different from the one the model was trained for. At the very
least, the data being biased towards a R2L representation (in the form of using the
Arabic number system and adopting legibility rules that accommodate right to left
calculations) lead to embeddings that maintain that bias even when
learned in a L2R fashion.


| Model             | Strategy               |
|-------------------|------------------------|
| LLaMA 1 & 2 | single digit           |
| LLaMA 3           | L2R chunks of 3 digits |
| OLMo 2            | L2R chunks of 3 digits |
| GPT-2         | pure BPE |
| GPT-3.5/4         | L2R chunks of 3 digits |
| Claude 3/4        | R2L chunks of 3 digits |


: Language models with their respective tokenization strategy for numbers.

<!-- the problem I've come to in talking about this is that I want to put this as the
property of an optimal representation. I guess the thing is here I started talking about
it as a property of the data, but it would follow that if talking about a certain set of
data -->

## Reification as computed embeddings - xVal

There have been other, more comprehensive approaches to the improvement of the
representation of numeric values. xVal is a notable one, as its approach encompasses
real numbers beyond just integers and does away with learning different representation
for each number.

The idea is maximizing the inductive bias in the representation by having embeddings
that are computed based on the number to be represented. Numerical values
represented by a single embedding vector associated with the `[NUM]` special token.

This fits very well with the idea of reification: the embedding is no longer just a
representation, but it contains and has properties of the object it represents.

The model uses two separate heads for number and token predictions. If the token
head predicts a `[NUM]` token as the successor, the number head gets activated and
outputs a scalar. The rest of the weights in the transformer blocks are shared, allowing
the learning of representations that are useful for both discrete text
prediction and continuous numerical prediction. This means the model develops
number-aware internal representations throughout all its layers, not just at the output.
The shared weights force the model to learn features that work for both linguistic and
mathematical reasoning simultaneously.

The approach is shown to improve performance over a series of other techniques, mostly
using a standard notation to represent numbers.

## The search for better suited representation

A case study of a Savant patient, DT [@murray2010], has been reported of having a
mathematical landscape with the following characteristics:

- Has sequence-space synesthesia with a "mathematical landscape" containing numbers
  0-9999
- Each number has specific colors, textures, sizes, and sometimes movements or sounds
- Prime numbers have special object properties that distinguish them from other numbers
- Arithmetic calculations happen automatically - solutions appear as part of his visual
  landscape without conscious effort
- fMRI studies showed that even unstructured number sequences had visual structure for
  DT

Sequence-space synesthesia consists in the visualization of certain sequences in physical
space.

# Embeddings Analysis

The analytic part of this work consists in the search for structure in LLM numerical
embeddings.

As stated previously, many recent open source models use an L2R tokenization scheme that
performs better when the tokenization is forced into R2L. There are no large scale open
source models using R2L tokenization as of the time of writing, but the correction in
performance with the tokenization change could be a hint that the learned L2R
representation might still have similar properties to a R2L one.

We're looking for clues of mathematical properties being encoded in the embeddings. As
the results show,

To check whether there is comparable structure in representation in LLMs, a variety of
analysis 





# Paragraphs yet to contextualize - not a real section

In [@mottron2006], the hypothesis is also that the capabilities of the savant might come
from privileged access to lower-level perceptual processing systems that have been
functionally re-dedicated to symbolic material processing. This suggests that
mathematical savants may bypass high-level algorithmic reasoning entirely, instead
leveraging perceptual mechanisms that can directly recognize patterns in numerical
relationships - much like how we might instantly recognize a face without consciously
processing its individual features.
