
# Introduction

\epigraph{As above, so below. As within, so without. As the universe, so the
soul.}{\textit{Emerald Tablet, misattributed}}

This thesis takes a look at LLMs from the perspective of their embeddings, in
particular their numerical ones. There are several reasons why I came to be
interested in this topic, the first and naive one being that tokenization
schemes, as naively implemented with the BPE algorithm, would leave a lot of
space for improvement in numerical tasks, and it's interesting to explore how.

Better performance in LLMs has been sought through very large scaling. There
are different reasons for this, one of them being The Bitter Lesson [@bitter-lesson], a
heuristic principle that states that general methods that better leverage computation
are better than methods that seek to use human domain-specific knowledge to inform
the implementation. This has been observed, for example, in the domain of chess,
where the strategies being put forward hard-coding human domain-specific knowledge
were ultimately beaten by deep search.

Not only scaling is a more general approach to LLM improvement, it also comes with the
potential of unlocking emergent capabilities. This approach has given results through
time, albeit with some inconsistencies hard to reconcile from an epistemological
perspective, such as the difficulty of actually designing good benchmarks for those
abilities that actually verify they go beyond memorization [@skalse2023].

This work proposes to examine numerical embeddings as a lens for assessing semantic
understanding in LLMs. By analyzing the representational structure of numbers and their
constituent digits within the same embedding space, we can investigate whether models
develop coherent semantic relationships that transcend the symbolic-numeric boundary,
potentially revealing genuine numerical understanding rather than pattern memorization.

Along with this, we investigate the presence of structures that come to be
through learning numerical representation. Given that we can naturally arrange numbers
in a sequence, it comes natural to see what the disposition of those sequences form when
arranged in the space of LLM representations. By taking inspiration from the Savant mode
of human cognition, that comes through exploiting spatial arrangements as a means to
perform calculations, we look for similar arrangements in LLMs, take a look at relevant
research, and make hypotheses on why they come to be.


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

Most of the tokenizers currently use L2R (left-to-right) clustering [@millidge2023],
meaning that a number such as $12345$ would be divided in two tokens, $123$ and $45$. It
has been shown [@singh2024] that this kind of clustering leads to worse arithmetic
performance, as this brings misalignment in digit positions and, as a consequence, in
positional significance.

Even more surprisingly, forcing the R2L token clustering of numbers in models already
trained with L2R clustering through the use of commas in the input (ex. $12,345$) leads
to big improvements in arithmetic performance [@singh2024;
@millidge2024]. Despite the model learning representations adapted to work with a L2R
token clustering strategy, forcing a R2L clustering at inference time shows substantial
improvements in arithmetic tasks, which means that despite being learned through an
unfavorable tokenization approach, the numeric representations retain the properties
that allow for the performance to improve when the digit clustering scheme is corrected.

![GPT-4o tokenization of different numerical quantities, displaying the L2R clustering
and the comma trick to force R2L clustering.](res/gpt4o-tokenization.png)

This could be happening for different reasons, for example:

- Arithmetic operations would still work locally in the 0-999 range, which allows for a
  correct reading on them and possible generalization on a larger scale, which
  counteracts the unintuitive clustering scheme.

- The forced tokenization also happens in the data, as numbers are often separated by
  punctuation in clusters of 3 digits, right to left, for legibility reasons
  [@singh2024].

- There's a geometric bias towards the right mode of operation given by the structures
  that form in the space to compute mathematical operations.

At the very least, the data bias towards R2L representation (in the form of
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

\clearpage

## Strategies for mathematical improvements through embeddings

Beyond improving tokenization, there have been other, more comprehensive approaches to
the improvement of the representation of numeric values. xVal is a notable one, as its
approach encompasses real numbers beyond just integers and does away with learning
different representation for each number.

The idea is maximizing the inductive bias in the representation by having embeddings
that are computed based on the number to be represented [@golkar2023]. Numerical values
represented by a single embedding vector associated with the `[NUM]` special token, that
gets scaled on the basis of the numerical value to represent. There is an assumption
made so that this works: the semantics of magnitude work as they pass from the
represented object to the structure of the representation.

This fits very well with the idea of reification, which will be described
later [@murray2010]: the embedding, beyond its qualities as representational object,
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
using a standard notation to represent numbers [@golkar2023]. This very thesis has been
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


## Savant syndrome and spatial representations

Savant syndrome is a rare condition in which people show exceptional proclivity towards
certain specific activities, usually accompanied by great impairments in other areas of
their lives. Savants can have exceptional abilities in math, art, music and other
fields, as well as instant calculation abilities that don't seem to come through
algorithmic processing.

A case study of a Savant patient DT [@murray2010] reveals a mathematical cognitive
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
representational one, by considering LLM embeddings. If these come through because of
geometric properties of the structures, and going with the assumption that those
structures are replicable in the high-dimensional vector spaces we're working with, it
would follow that strict optimization through gradient descent could be a possible way
to make them come about.

## The Platonic Representation Hypothesis

According to [@huh2024] AI models, particularly deep networks, are converging. The
central hypothesis is that different models are converging toward a shared statistical
model of reality, akin to Plato's concept of an ideal reality. This representation is
termed the Platonic representation.

This convergence appears to be driven by several selective pressures: larger models have
more capacity to find optimal representations; models trained on more diverse tasks are
constrained to find solutions that work across multiple domains; and deep networks have
implicit biases toward simpler solutions.

For the investigation of numerical representations, this suggests that if there are
indeed optimal geometric structures for mathematical reasoning, different models might
naturally converge toward them during training. The shape suggested (the helix)
has properties on an information-theory basis that make its use as a learning geometry
more likely [@kantamneni2025]. In particular, their self-similarity can be a useful
error-correcting property.

## The Helix and its role in LLM addition

During the final process of literature review of this thesis, a paper was found
that recontextualized some of the findings seen here. [@kantamneni2025] has revealed
how mid-sized language models including GPT-J, Pythia-6.9B, and Llama3.1-8B employ a
helix to encode and manipulate numerical values during arithmetic operations. The helix
gets fit with the operands required, and through structural manipulation via the "Clock
algorithm" performs addition by rotating helical representations and reading out the
final answer. From an information-theoretic perspective, the authors demonstrate that
helical representations provide significant computational advantages over linear
encodings, offering built-in redundancy and error-correction properties. Even with
highly precise linear representations ($R^2 = 0.997$), linear addition achieves less than
20% accuracy while the helical approach achieves over 80%, suggesting the periodic
structure serves as an error-correcting mechanism analogous to how humans use decimal
digits rather than slide rules for precise calculation.

In looking at the numerical embeddings of OLMo and Llama models, we observe very similar
structures as the ones described in the paper, which gives more comprehensive
explanations on how the structures are employed to perform mathematical operations such
as addition, done from a mechanistic interpretability perspective. MI attempts to
explain the model workings through the reverse engineering of it, and going through the
motions of the network. The work presented here will limit itself to graphical
visualization of single-token embeddings and feature analysis, although the perspective
presented by Kantamneni et al certainly seems to give partial confirmation to the
findings here presented.

\clearpage

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

