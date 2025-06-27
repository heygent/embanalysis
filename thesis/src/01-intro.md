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


## The Transformer architecture and vector representations

### The inductive bias of Tokenization

Modern LLMs are built on the Transformer architecture [@vaswani2023], which operates by
converting input text into sequences of discrete tokens that are then mapped to
high-dimensional vector representations. This initial tokenization step creates an
inductive bias that shapes how the model processes information [@singh2024], with
significant implications for the application of the numerical data to arithmetical
tasks.

<BPE Tokenizer description>

While GPT-2 used to have a purely BPE frequency-based approach on number tokenization,
which leads to an uneven tokenization of numbers based on their frequency, modern
models either tokenize digits separately (so as $'1234' \rightarrow [1, 2, 3, 4]$), or
tokenize clusters of 3 digits, encompassing the numbers in the range 0-999.

![GPT-2 number tokenization. Each row represents 100 numbers, yellow squares mean that
the number is represented by a single token, purple ones by multiple. Image from
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

<the problem I've come to in talking about this is that I want to put this as the
property of an optimal representation. I guess the thing is here I started talking about
it as a property of the data, but it would follow that if talking about a certain set of
data>


| Model             | Strategy               |
|-------------------|------------------------|
| LLaMA 1 & 2 | single digit           |
| LLaMA 3           | L2R chunks of 3 digits |
| OLMo 2            | L2R chunks of 3 digits |
| GPT-2         | pure BPE |
| Claude 3          | R2L chunks of 3 digits |


: Language models with their respective tokenization strategy for numbers.

The latter approach is what is taken into consideration into the analytical part of this
work, as it allows examining what representation do LLMs use to represent the numbers in
that range.

There have been proposed approaches in the literature that aim at maximizing the
inductive bias in the representation by having embeddings that are computed based on the
number to be represented. This fits very well with the idea of reification: the
representation is no longer just a representation, but it has properties of the object
that it represents. This can lead to symbolic representation that are directly fungible
for the desired computations<?>.

<span class="free"> It's fascinating to observe that a case study of a Savant patient,
DT [@murray2010], has been reported of having a mathematical landscape that has very
similar characteristics:

- Has sequence-space synesthesia with a "mathematical landscape" containing numbers
  0-9999
- Each number has specific colors, textures, sizes, and sometimes movements or sounds
- Prime numbers have special object properties that distinguish them from other numbers
- Arithmetic calculations happen automatically - solutions appear as part of his visual
  landscape without conscious effort
- fMRI studies showed that even unstructured number sequences had visual structure for
  DT </span>

In [@mottron2006], the hypothesis is also that the capabilities of the savant might come
from privileged access to lower-level perceptual processing systems that have been
functionally re-dedicated to symbolic material processing. This suggests that
mathematical savants may bypass high-level algorithmic reasoning entirely, instead
leveraging perceptual mechanisms that can directly recognize patterns in numerical
relationships - much like how we might instantly recognize a face without consciously
processing its individual features. There are also arguably similar mechanisms already
implemented in LLMs, although usually employed in the context of <?> gradient
normalization, in the form of skip connections.

