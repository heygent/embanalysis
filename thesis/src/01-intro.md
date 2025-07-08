# Introduction

This thesis explores LLMs from the perspective of their embeddings, in
particular their numerical ones. There are several reasons why I came to be
interested in this topic, the first and naive one being that tokenization
schemes, as naively implemented with the BPE algorithm, would leave a lot of
space for improvement in numerical tasks, and it's interesting to explore how.

Better performance in LLMs has been sought through the lenses of scale, and
looking for emergent properties as training time and resources increase. There
are different reasons for this, one of them being The Bitter Lesson [@bitter-lesson], a
heuristic principle that states that general methods that better leverage computation
are better than methods that seek to use human domain-specific knowledge to inform
the implementation. This has been observed, for example, in the domain of chess,
where the strategies being put forward hard-coding human domain-specific knowledge
were ultimately beaten by deep search.

After the big success story of scaling in LLMs, the main reach has been towards
increasing model size and training on bigger datasets, and in unlocking the emergent
capabilities that would come along those. While this approach has given results, albeit
with some inconsistencies hard to reconcile from an epistemological perspective, such as
the difficulty of actually designing good benchmarks for those abilities that actually
verify they go beyond memorization [@skalse2023], this road would lead to the
monopolistic control of the best version of this tool to the actors that are able to get
access to the most amount of data.

If "Scale is all you Need" and LLM improvement is purely a game of resource
accumulation, we would be in a situation that could exacerbate the inequalities we're
living under at the present time, leading effectively to a crystallization of the power
structures that, as of today, have the biggest capacity for data collection. As the
state of the art gets better, the barriers of entry rise in terms of performance, training
data and hardware required to have a model that performs competitively. As such, it is
of primary importance to find strategies to break through the massive resource
requirements needed for AI training and performance, and find alternatives that
allow for a less resource-intensive development of the field.

What is investigated here are some possible improvements that don't come
directly from training, but from a better understanding of how, through learning, the
model weights that allow LLMs to function come to be. This comes through thought
experiments about human cognition, and anomalous cases of it (in particular, we'll take
a look at Savant syndrome and what it can tell us in the context of learning), as well
as recent research put forward about LLM capabilities in the specific field of math.

In particular, one thing I want to propose is that the representation of numbers can be
a gateway to better understanding LLM representations in general, as their object of
representation is the same as the object being represented (in both cases numbers),
allowing to look for the relations between the two using mathematical methods of
analysis.

# Background

## The inductive bias of Tokenization

Modern LLMs are mostly autoregressive models built on the Transformer architecture
[@vaswani2023]. Transformers are a deep learning architecture based on attention, a
mechanism that relates words in different positions in a sentence by computing weighted
relationships between all input tokens, allowing the model to capture long-range
dependencies and contextual relationships that sequential models like RNNs struggle
with. The first step in most Transformer models is tokenization, which operates by
converting input text into sequences of discrete tokens that are then mapped to
high-dimensional vector representations. This initial step creates an inductive bias
that shapes how the model processes information [@ali2024; @singh2024], with significant
implications for the application of numerical data to arithmetical tasks.

The most used algorithm for tokenization is currently Byte-Pair Encoding [@radford2019],
which, given a fixed vocabulary size, starts with individual characters and iteratively
merges the most frequently occurring pairs of adjacent tokens until the vocabulary limit
is reached. This process naturally creates longer tokens for common substrings that
appear frequently in the training data. For numbers, this means that frequently
occurring numerical patterns like "100", "2020", or "999" might become single tokens,
while less common numbers get broken into smaller pieces. The result is an idiosyncratic
and unpredictable tokenization scheme where similar numbers can be tokenized completely
differently based purely on their frequency in the training corpus. While GPT-2 used to
have a purely BPE tokenizer, the successive iteration of GPT and generally more recent
models either tokenize digits separately (so as $'1234' \rightarrow [1, 2, 3, 4]$), or
tokenize clusters of 3 digits, encompassing the integers in the range 0-999.

Most of the tokenizers right now do L2R (left-to-right) clustering [@millidge2023],
meaning that a number such as $12345$ would be divided in two tokens, $123$ and $45$. It
has been shown [@singh2024] that this kind of clustering leads to worse arithmetic
performance, as this brings misalignment in digit positions and, as a consequence, in
positional significance.

An even more surprising development is that forcing the R2L token clustering of numbers
in models already trained with L2R clustering through the use of commas in the input
(ex. $12,345$) leads to big improvements in arithmetic performance [@singh2024;
@millidge2024]. Despite the model learning representations adapted to work with a L2R
token clustering strategy, forcing a R2L clustering at inference time shows substantial
improvements in arithmetic tasks, which means that despite being learned through an
unfavorable tokenization approach, the numeric representations retain the properties
that allow for the performance to improve when the digit clustering scheme is corrected.

![GPT-4o tokenization of different numerical quantities, displaying the L2R clustering
and the comma trick to force R2L clustering.](res/gpt4o-tokenization.png)

There can be different hypotheses on why this might be, for example: arithmetic
operations would still work locally in the 0-999 range, which allows for a correct
reading on them and possible generalization on a larger scale; the forced tokenization
also happens in the data, as numbers are often separated by punctuation in clusters of 3
digits, right to left, for legibility reasons [@singh2024]; or, as it will be explored
later, an underlying self-correcting learning structure.

At the very least, the data being biased towards a R2L representation (in the form of
using the Arabic number system and adopting legibility rules that accommodate right to
left calculations) leads to embeddings that maintain that bias even when learned in a
L2R fashion. This can be a possible hint towards the optimality of certain
representations compared to others, given the resilience in preferring a certain
tokenization scheme over the one the model is trained on.


| **Model**           | **Strategy**           |
| ------------------- | ---------------------- |
| LLaMA 1 & 2         | Single digit           |
| LLaMA 3             | L2R chunks of 3 digits |
| OLMo 2              | L2R chunks of 3 digits |
| GPT-2               | Pure BPE               |
| GPT-3.5 / GPT-4     | L2R chunks of 3 digits |
| Claude 3 / Claude 4 | R2L chunks of 3 digits |



: Language models with their respective tokenization strategy for numbers.

## Strategies for mathematical improvements through embeddings

Beyond better tokenization, there have been other, more comprehensive approaches to the
improvement of the representation of numeric values. xVal is a notable one, as its
approach encompasses real numbers beyond just integers and does away with learning
different representation for each number.

The idea is maximizing the inductive bias in the representation by having embeddings
that are computed based on the number to be represented [@golkar2023]. Numerical values
represented by a single embedding vector associated with the `[NUM]` special token.

This fits very well with the idea of reification, which will be described
later[@murray2010]: the embedding, beyond its qualities as representational object,
becomes an entity with features that actively aid in the calculation process.

The model uses two separate heads for number and token predictions. If the token head
predicts a `[NUM]` token as the successor, the number head gets activated and outputs a
scalar. The rest of the weights in the transformer blocks are shared, allowing the
learning of representations that are useful for both discrete text prediction and
continuous numerical prediction. This means the model develops number-aware internal
representations throughout all its layers, not just at the output. The shared weights
force the model to learn features that work for both linguistic and mathematical
reasoning simultaneously.

The approach is shown to improve performance over a series of other techniques, mostly
using a standard notation to represent numbers [@golkar2023]. This work has been
inspired by the xVal paper, with one of its initial goals being to find good
representations for computed numerical embeddings.

There are several different approaches to improving math performance in LLMs that don't
necessarily come directly from training, but from a better understanding of how
representations work. Other approaches include giving models better positional
information about digits within numbers, such as Abacus Embeddings [@mcleish2024], which
encode each digit's position relative to the start of the number and can improve
arithmetic performance substantially.

While on one hand particular modes of numerical cognition can be explored through
explicitly reifying the representation (through the xVal approach), what the rest of the
analysis hinges on is whether LLMs develop such structures on their own, by looking at
their embeddings, as this could hypothetically inform us on how to build these
structures ourselves in a more direct way than training.


## Cognitive science - Savant syndrome and spatial representations

A case study of savant patient DT [@murray2010] reveals a mathematical cognitive
architecture with the following characteristics:

- has sequence-space synesthesia with a "mathematical landscape" containing numbers
  0-9999
- each number possesses specific colors, textures, sizes, and sometimes movements or
  sounds
- prime numbers have distinctive object properties that distinguish them from other
  numbers
- arithmetic calculations happen automatically
- solutions appear as part of his visual landscape without conscious effort
- fMRI studies showed that even unstructured number sequences had coherent visual
  structure for DT.

Murray argues that savants possess highly accessible concrete representations of
abstract concepts, for which she uses the term reification - the conversion of abstract
concepts into concrete, spatial entities that can be directly "inspected" rather than
computed.

Sequence-space synesthesia is the spontaneous visualization of numerical sequences
in organized spatial arrangements. The remarkable mathematical abilities of savants with
this condition suggest that their specialized perceptual representations confer
significant computational advantages over normal human numerical calculation abilities.

Given that the spatial arrangement confers advantages in numerical calculation to the
subject, we can pose the question: are there specific spatial arrangements that enable
advantageous numerical calculations, and are those present or replicable in LLMs?
The spatial idea is easily translatable from the perceptive sphere to the
representational one, by considering LLM embeddings.
This also requires the assumption that representational advantages can translate from
humans to LLMs. <expand>

According to [@huh2024] AI models, particularly deep networks, are converging. The
central hypothesis is that different models are converging toward a shared statistical
model of reality, akin to Plato's concept of an ideal reality. This representation is
termed the Platonic representation.

This convergence appears to be driven by several selective pressures: larger models have
more capacity to find optimal representations; models trained on more diverse tasks are
constrained to find solutions that work across multiple domains; and deep networks have
implicit biases toward simpler solutions [@huh2024].

For the investigation of numerical representations, this suggests that if there are
indeed optimal geometric structures for mathematical reasoning, different models might
naturally converge toward them during training. The shape suggested (the helix)
has properties on an information-theory basis that make its use as a learning geometry
more likely, in particular, its self-similarity can be an error-correcting property.

## Dimensionality reduction and Embedding Visualization

To visualize and analyze the high-dimensional embedding spaces that LLMs use for
numerical representations, we need techniques that make the underlying structure
evident. For this reason, we employ the following dimensionality reduction techniques:

- **SVD (Singular Value Decomposition)** is a fundamental matrix factorization that
  decomposes any matrix $A$ into three component matrices: $A = U\Sigma V^T$, where $U$
  and $V$ contain orthogonal vectors (left and right singular vectors respectively) and
  $\Sigma$ contains the singular values on its diagonal. SVD reveals the underlying
  structure of the matrix by identifying the principal directions of variation and their
  relative importance through the singular values.

- **PCA (Principal Component Analysis)** emerges as a specific application of SVD. By
  applying SVD to a centered data matrix (where each variable has been mean-centered),
  the right singular vectors $V$ become the principal components - the directions of
  maximum variance in the data. The singular values in $\Sigma$ are directly related to
  the eigenvalues of the covariance matrix.

- **t-SNE (t-Distributed Stochastic Neighbor Embedding)** [@maaten2008] converts
  similarities between data points in high-dimensional space into probabilities, then
  uses gradient descent to minimize the divergence between these probabilities and those
  of points in a low-dimensional embedding. It excels at preserving local neighborhood
  structure, making clusters very distinct in the visualization. However, t-SNE can
  distort global structure and distances between distant clusters become less
  meaningful, making it primarily useful for identifying local groupings in numerical
  embeddings.

- **UMAP (Uniform Manifold Approximation and Projection)** [@mcinnes2020] also preserves
  local structure like t-SNE, but additionally maintains more of the global structure
  through its foundation in topological data analysis. UMAP constructs a topological
  representation of the data in high dimensions, then uses stochastic gradient descent
  to optimize a low-dimensional representation to have similar topological properties.
  This makes it better suited for analyzing how models organize numerical concepts
  across different scales - both local clusters of similar numbers and global
  relationships between distant numerical regions.

## How current representational issues can help us better understand LLMs

Tokenization as a process is highly idiosyncratic, and the last "mechanical" step in the
LLM pipeline. This situation has been already identified as a problem by some authors
[@bitter-lesson-tokenization], and it's a passage that, while necessary to get LLM
performance to the point where it is today, given it allows the network to operate on a
token level instead of character, would probably be better replaced by learnable
approaches rather than being constrained by fixed pre-trained vocabularies that limit
the model capability to have a representation of the input adapted to the task at hand.
There are already alternative approaches being proposed [@pagnoni2024; @islam2022],
which manage to have better efficiency while addressing some of the problems rigid
training-time tokenization causes (like the strawberry problem). However, what the state
of the art offers now gives the opportunity for the exploration of representational
structures that have direct and easy associations with tokens, which we can catalogue in
sequences and analyze in a systematic way.

During the process of literature review for this thesis, and after the main
experimentation, an article was found touching on similar themes, in particular using
mechanistic techniques to get to the way LLMs perform addition, and in the process


recent research was also found demonstrating that LLMs use trigonometry
to do addition <?> [@kantamneni2025], representing numbers as a generalized helix which
is strongly causally implicated for addition and subtraction tasks. This provides
evidence that language models do indeed develop structured geometric representations for
numerical reasoning, supporting the hypothesis that analyzing these naturally emergent
structures could inform better representation design.
