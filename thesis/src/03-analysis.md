# Embeddings Analysis

The analytic part of this work consists in the search for structures in LLM
numerical embeddings.

As stated previously, recent open source models mostly employ an L2R
tokenization scheme. There are no large scale open source models using R2L
tokenization as of the time of writing, but the improvement in performance
observed when using R2L tokenization in L2R-trained models could be a hint that
L2R embedding representations still have similar properties to the R2L ones.

## Methodology

Two models are taken in consideration:

- OLMo-2-1124-7B is a model by AllenAI, which is favorable to research uses
thanks to the full disclosure of training data, code, logs and checkpoints

- Llama-3.2-1B-Instruct, due to being a small and manageable model to do analysis
with on limited hardware

For each of these, dimensionality reduction is applied through PCA, SVD, t-SNE and UMAP,
and the results are used to produce 2D and 3D visualization meant to show the geometric
structure that the representation assume in the space.

Then, an analysis of correlation with mathematical sequences is done: for each
dimension, we encode mathematical interesting sequences the embedding features might be
detecting for and look for correlations. We adopt the following encoding for sequences:

- direct, meaning we test the correlation directly with the sequence. We do this with
  plain integers in sequence (0, 1, 2, 3) and their logarithm (log(1), log(2), ...)

- one hot encoded, for the sequences where the growth would be too fast to have
  meaningful detection through direct encoding. We tried this for Fibonacci, triangular,
  and prime numbers

- Gaussian-smoothed one-hot encoded, same as last point but allowing a more gradual
  climb around the positions corresponding to the numbers by applying a Gaussian
  smoothing filter to the one-hot encoded vector.

- Fourier encoding, as $\sin\left(2 \pi  \cdot \frac{n_i}T\right)$ and $\cos\left(2 \pi
  \cdot \frac{n_i}T\right)$ where $n_i$ is the element of the sequence to be encoded and
  $T \in \{1, 2, 5, 10, 100\}$ is a collection of possible periods of encoding, as suggested
  in [@kantamneni2025] as a possible way for features to encode numeric characteristics.

## OLMo-2-1124-7B

### Linear analysis

![Principal components 1 and 2 of the OLMo model. Random embeddings sample for
comparison.](plots/OLMo-2-1124-7B_00_pca_components_gradient_v1.svg){#fig:olmo-pca}

In [@fig:olmo-pca], we plot the first two principal component of the numerical
embeddings, color-grading them on the basis of their numerical value.  The result shows
that numerical tokens do not occupy the embedding space randomly: they follow a
constrained path that preserves numerical relationships, suggesting that the model has
learned to encode ordering properties of the numbers within its representation. The
gradient is particularly smooth, suggesting that similar numbers maintain spatial
proximity in the reduced space.

![SVD for the two main components of the OLMo model, with random embeddings
sample for
comparison](plots/OLMo-2-1124-7B_02_svd_components_gradient_v1.svg){#fig:olmo-svd}

The relationship and similarities are even more clear in [@fig:olmo-svd], which, lacking
the data centering done in the PCA, shows a much more consistent geometric structure,
showing that the encoding of information likely happens in absolute distances rather
than just with relative positioning between data points. This will also inform the
strategies we use for reconstructing the datapoints with UMAP, as we'll do projections
that make use of Euclidean distance as well as cosine similarity as a metric.

![OLMo-2-1124-7B 3D visualization of SVD components .](plots/olmo-svd-3d.png){#fig:olmo-svd-3d width=50%}

![SVD coloring done by digit length and hundreds digit, highlighting the
clustering properties of the
embeddings.](plots/OLMo-2-1124-7B_03_svd_digit_visualizations_v1.svg){#fig:olmo-svd-digits}

Looking at [@fig:olmo-svd-digits], the fractal structure is strinking, and the
structure repeating within numbers of different digit lengths becomes really clear.
Models like Qwen-2.5 do away with tokenizing numbers outside the single digits from 0 to
9, and this picture can offer a compelling explanation on why that can be justified. In
fact, it seems like the encoding of higher digit quantities brings along a lot of
redundancy. On the other hand, as discussed for [@kantamneni2025], this same redundancy
could be used by the models an error correcting mechanism when it has to apply
numerical operations, leading to possibly better performance.


#### Explained variance

![OLMo PCA - explained variance
overview](plots/OLMo-2-1124-7B_01_pca_variance_overview_v1.svg){#fig:olmo-variance}

The explained variance by component plot ([@fig:olmo-variance], left) shows a sharp drop
within the first few components, meaning that the first principal components capture
dramatically more variance than subsequent ones. The cumulative explained variance
(right) shows that approximatively 600 principal components are needed to reach 90% of
explained variance.

By this we can conclude that the embeddings have a much lower intrinsic
dimensionality than their full 4096 dimensions, and that they lie on a
low-dimensional manifold in the full representation space. Only one-fifth of the
total embedding space is necessary to capture 90% of the variance, and, as described
earlier, the structures already encoded provide already a lot of redundancy.

### Non-linear analysis

![t-SNE visualization for OLMo
embeddings.](plots/OLMo-2-1124-7B_07_tsne_components_gradient_v1.svg){#fig:olmo-tsne}


| **Parameter**       | **Value** |
| ------------------- | --------- |
| perplexity          | 75        |
| max\_iter           | 3000      |
| learning\_rate      | 50        |
| early\_exaggeration | 20        |
| random\_state       | 42        |

: t-SNE hyperparameters for the presented plots. {#tbl-tsne-params}

The t-SNE visualization in [@fig:olmo-tsne] shows a distinctive branching pattern
emanating from a central region, with low numbers at the center and higher ones
radiating outward. The color progression follows these branches, indicating that
numerical sequences are preserved along each arm. The gradient seems also to transition
circularly; branches with gradually increasing numbers turn around the center
before abruptly getting back to the start. When interpreting the colors as
indicators of depth, it can look like a spiral from a top-down perspective, giving a
visual confirmation of what has been said about helical structures in [@kantamneni2025].

![UMAP visualization with cosine
distance](plots/OLMo-2-1124-7B_09_umap_cosine_components_gradient_v1.svg){#fig:olmo-umap-cosine}

![UMAP visualization with Euclidean
distance](plots/OLMo-2-1124-7B_11_umap_euclidean_components_gradient_v1.svg){#fig:olmo-umap-euclidean}

UMAP has been run using both Euclidean ([@fig:olmo-umap-euclidean]) and cosine distances
([@fig:olmo-umap-cosine]), since the SVD visualization has shown that absolute distances
can matter in this model. In the UMAP case we can observe a loss of shape similar to
what happened in the PCA and SVD case. While the structure is congruent when using
Euclidean distances, segregated clusters form when representing cosine similarity, with
their predominant criterion of division being the hundreds' digit. Using Euclidean
distances gives a picture similar to t-SNE, but projected and stretched and with more
dispersion for numbers close to zero. The spiral-like conformation is also notable here.

### Correlation with mathematical properties


| dimension | property  | encoding          | correlation          | P\_value                |
| --------- | --------- | ----------------- | --------------------:| -----------------------:|
| 514       | log       | direct            | -0.6728704322049941  | 8.446593249981852e-133  |
| 3085      | even      | direct            | -0.6099014911112753  | 6.384089936863827e-103  |
| 3085      | numbers   | direct            | -0.6099014911112753  | 6.384089936863827e-103  |
| 3085      | log       | direct            | -0.6065335539115024  | 1.6467293892949673e-101 |
| 514       | fibonacci | gauss             | 0.3743022442573753   | 1.304359393219689e-34   |
| 2538      | fibonacci | gauss             | 0.35111562249158407  | 2.1919842097713776e-30  |
| 514       | primes    | gauss             | 0.2635789635453773   | 2.3511422135536455e-17  |
| 695       | primes    | fourier\_cos\_T10 | -0.2206894374229878  | 1.698004543337325e-12   |
| 2538      | fibonacci | fourier\_cos\_T5  | -0.18335840673977102 | 5.200214605563535e-09   |


: Feature-sequence correlations in OLMo-2. {#tbl:olmo-correlations}

A lot of features ([@tbl:olmo-correlations; @fig:olmo-properties]) correlate very
significatively with magnitude, giving confirmation of the semantic connection between
the embedding value and the number represented. We find:

- Dimension 514 has a high correlation (0.37) with Fibonacci numbers using Gauss one-hot
  encoding, and with prime numbers with Gauss one-hot encoding (0.26).
- There are other dimensions are also correlated with Fibonacci and Primes, to a lesser
  extent, and Fourier encoding seems to show weaker ties than one-hot Gaussian encoding.

Although while these can be useful clues about the relation between the features and the
numerical properties they're correlated with, they do not explain by themselves how and
why the features are tied to those properties. The correlation with Fibonacci numbers in
particular is high enough to not resemble a pure coincidence, especially in such a
high-dimensional space. We can speculatively make some hypotheses:

- Features in the embedding might work as hierarchical detectors, with some of them
  being broad-scope, general detectors of numbers of interest (dimension 514), and
  others being more specific to certain properties (2538 to Fibonacci numbers).
- The correlation with the features is tied with the geometry of the embedding space.
  The Fibonacci numbers have ties to spiral structures (in particular golden spirals),
  which may have a relation to the self-similar structures observed.

Further study would be needed to untangle the relationship between mathematical
sequences and features, although having a dimension (514) with such a strong correlation
with Fibonacci numbers and such a low p-value doesn't seem dismissible over random
chance.

![Helical construction of embeddings used to do addition
[@kantamneni2025].](res/helical_structure.png){#fig:fibo-spiralB width=30%}



![Mathematical sequences against the number of associated strongly correlated embedding
dimensions.](plots/OLMo-2-1124-7B_strong_property_correlations_v1.svg){#fig:olmo-properties}

## Llama-3.2-1B-Instruct

### Linear analysis

![PCA visualization of Llama
embeddings.](plots/Llama-3.2-1B-Instruct_00_pca_components_gradient_v1.svg){#fig:llama-pca}

Differently from OLMo, in LLaMa PCA and SVD lead to very similar representations, which
might mean that LLaMa learned number embeddings are much closer to having a 0 mean.

The self-similar, recursive structure based on digit count is immediately visible
in the PCA plot.

The PCA projection shows a continuous, arc-shaped curved manifold, with smoother
transitions between numbers and a distinct separation with numbers close to 0.
As with what was seen with OLMo, it looks like the PCA centering might end up
destroying geometric relationships that are better preserved in the SVD
visualizations.

![SVD visualization of Llama
embeddings](plots/Llama-3.2-1B-Instruct_02_svd_components_gradient_v1.svg){#fig:llama-svd}

The SVD plot shows a remarkably linear arrangement - numbers form an almost
straight diagonal line from small (yellow) to large (purple) values. This linear
structure is much more pronounced than OLMo-2's curved SVD patterns, and it is a
unique shape rather than a recursive, recurring pattern.

![Llama SVD visualization by
digit](plots/Llama-3.2-1B-Instruct_03_svd_digit_visualizations_v1.svg){#fig:llama-svd-digits}

The digit-based coloring reveals clear but subtle clustering by mathematical
properties. Unlike OLMo-2's distinct spatial territories, Llama-3.2 shows
gradual transitions along the linear arrangement while maintaining digit-based
organization patterns.

#### Explained variance

![Llama PCA explained
variance.](plots/Llama-3.2-1B-Instruct_01_pca_variance_overview_v1.svg){#fig:llama-variance}

The explained variance plot reveals slightly higher information concentration
than OLMo-2.  Llama-3.2 reaches 90% explained variance with approximately 500
components compared to OLMo-2's 500 components. This suggests more efficient
numerical encoding in the smaller model.

### Non-Linear analysis

These nonlinear projections reveal dramatically different organizational
patterns from both the linear methods and from OLMo-2's structures.

![t-SNE structure in
Llama](plots/Llama-3.2-1B-Instruct_07_tsne_components_gradient_v1.svg){#fig:llama-tsne}

The t-SNE visualization is very unusual, and show continuous, winding structures
that might look like they had been uncoiled or unwound from a higher-dimensional
spiral arrangement. The mathematical progression follows these winding paths
smoothly. This can be informative, as for their particularly keen encoding of
the Fibonacci sequence, as will be shown successively.

![UMAP with cosine similarity in
Llama](plots/Llama-3.2-1B-Instruct_09_umap_cosine_components_gradient_v1.svg){#fig:llama-umap-cosine}

![Clustering in UMAP with cosine
similarity](plots/Llama-3.2-1B-Instruct_10_umap_cosine_digit_visualizations_v1.svg){#fig:llama-umap-cosine-digits}

![UMAP with Euclidean distance in
Llama](plots/Llama-3.2-1B-Instruct_11_umap_euclidean_components_gradient_v1.svg){#fig:llama-umap-euclidean}

The UMAP visualization is resembling the OLMo's one. It's also interesting to
see that changing the distance function to Euclidean doesn't have particular
effects, unlike the previous OLMo visualization.

### Correlation with mathematical properties

![Dimensions strongly correlated with properties in Llama
3.2](plots/Llama-3.2-1B-Instruct_13_strong_property_correlations_v1.svg){#fig:llama-properties}

In this case we see a lot more components directly encoding for digit_count, as
well as for parity. There are 12 strongly correlated components with primality
and 10 with being a Fibonacci number. There is still a big number of components
strongly correlated with the fibonacci_proximity, which would need further
analysis to fully establish whether their sensitivity to magnitude dominates
over the detection of Fibonacci numbers.

# Conclusions

This thesis investigated whether Large Language Models develop structured
numerical representations that might share organizational principles with the
specialized representations observed in mathematical savants. Through analysis
of numerical embeddings in OLMo-2-1124-7B and Llama-3.2-1B-Instruct, we found
evidence of systematic mathematical structure within learned representations.

## Key Findings

### Structured Numerical Embeddings

Our analysis revealed that numerical tokens are not randomly distributed in
embedding space but follow organized patterns:

**Geometric Organization**: Principal component analysis showed that numerical
embeddings lie on low-dimensional manifolds. OLMo-2 exhibited U-shaped curves
with recursive patterns across digit ranges, while Llama-3.2 displayed more
linear arrangements. Both models required only ~500 components to capture 90% of
variance, indicating substantial dimensionality reduction from the full
4096-dimensional space.

**Mathematical Property Correlations**: Multiple embedding dimensions showed
*significant
correlations with mathematical properties:

- Magnitude and digit count exhibited the strongest correlations (r > 0.67) -
Primality was encoded across multiple dimensions - Fibonacci number proximity
showed notable correlations, though potentially influenced by magnitude effects
- Binary properties like evenness were systematically represented

The consistent organization of numerical embeddings according to mathematical
properties suggests that neural language models might spontaneously develop
structured representations during training. This organization goes beyond simple
ordering, incorporating complex mathematical relationships like primality and
sequence membership.

## Limitations

The analysis was limited to two models and a specific set of mathematical
properties.  The relationship between Fibonacci proximity and magnitude
highlights the challenge of isolating specific property detectors from general
ordering mechanisms. Further investigation with controlled experiments would be
needed to establish causal relationships.

Whether these findings extend to larger models, different architectures, or
other mathematical domains remains to be determined. The observed structures may
reflect training data properties as much as emergent organizational principles.

## Future Directions

This work suggests several research directions: investigating mathematical
representations across model scales and architectures, exploring computed
embedding approaches that leverage discovered geometric structures, and
examining whether similar organizational principles apply to other domains where
specialized representations might be beneficial.

The findings contribute to understanding how neural language models organize
mathematical information and suggest that structured representations may emerge
naturally in systems trained on numerical data. While modest in scope, these
results provide a foundation for further investigation into the relationship
between representational structure and mathematical reasoning capabilities in
artificial systems.
