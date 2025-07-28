# Embeddings Analysis

## Methodology

The analytic part of this work consists in the search for structures in LLM
numerical embeddings. Two models are taken in consideration:

- OLMo-2-1124-7B [@olmo2025] is a model by AllenAI, which is favorable to research uses
  thanks to the full disclosure of training data, code, logs and checkpoints. This model
  was trained on 4.05T tokens using a two-stage curriculum: initial pretraining on 3.90T
  tokens of web-based data, followed by a specialized mid-training phase on 150B
  high-quality tokens including 10.7B synthetic mathematical data specifically designed
  to enhance numerical reasoning capabilities.

- Llama-3.2-1B-Instruct [@grattafiori2024], due to being a small and manageable model to
  do analysis with on limited hardware. This model derives from the Llama 3 family,
  which was trained on approximately 15T multilingual tokens with dynamic data mixing
  throughout training, including mathematical content integrated continuously rather
  than in separate phases.

Both models underwent single-pass training without data repetition, meaning their
numerical representations developed through one-time exposure to mathematical content.

For each of these, dimensionality reduction is applied through PCA, SVD, t-SNE and UMAP,
and the results are used to produce 2D and 3D visualization meant to show the geometric
structure that the representation assume in the space.

Then, an analysis of correlation with mathematical sequences is done: for each
dimension, we encode mathematically interesting sequences the embedding features might be
detecting for, and measure their Pearson correlation coefficient along with their
respective p-value. We adopt the following encoding for sequences:

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

\clearpage

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
proximity in the reduced space. It's also notable that in the top right of the curve,
there seems to be a smaller curve forming; it's better visible through the SVD
visualizations, but that part corresponds to single and double-digit integers
replicating the bigger overall structure of the lower curve.

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

Looking at [@fig:olmo-svd-digits], the self-similar structure repeating through
different integer lengths is striking. Models like Qwen-2.5 do away with tokenizing
numbers outside the single digits from 0 to 9, and this picture can offer a compelling
explanation on why that can be justified. In fact, it seems like the encoding of higher
digit quantities brings along a lot of redundancy. On the other hand, as discussed for
[@kantamneni2025], this same redundancy could be used by the models an error correcting
mechanism when it has to apply numerical operations, possibly leading to better
performance on calculation tasks.


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

: t-SNE hyperparameters for the presented plots. {#tbl:tsne-params}

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

| dimension | property  | encoding          | correlation |    p\_value |
| --------- | --------- | ----------------- | ----------: | ----------: |
| 514       | log       | direct            |    -0.67287 | 8.4465e-133 |
| 3085      | even      | direct            |    -0.60990 | 6.3840e-103 |
| 3085      | numbers   | direct            |    -0.60990 | 6.3840e-103 |
| 3085      | log       | direct            |    -0.60653 | 1.6467e-101 |
| 514       | fibonacci | gauss             |     0.37430 |  1.3043e-34 |
| 2538      | fibonacci | gauss             |     0.35112 |  2.1919e-30 |
| 514       | primes    | gauss             |     0.26358 |  2.3511e-17 |
| 695       | primes    | fourier\_cos\_t10 |    -0.22069 |  1.6980e-12 |
| 2538      | fibonacci | fourier\_cos\_t5  |    -0.18336 |  5.2002e-09 |


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

<!-- ![Helical construction of embeddings used to do addition
[@kantamneni2025].](res/helical_structure.png){#fig:fibo-spiralB width=30%} -->

![Mathematical sequences against the number of associated strongly correlated embedding
dimensions.](plots/OLMo-2-1124-7B_strong_property_correlations_v1.svg){#fig:olmo-properties
width=45%}

\clearpage

## Llama-3.2-1B-Instruct

### Linear analysis

![PCA visualization of Llama
embeddings.](plots/Llama-3.2-1B-Instruct_pca_components_gradient_digit_length_v1.svg){#fig:llama-pca}

The LLaMa PCA plot ([@fig:llama-pca]) shows a similar picture to the OLMo one: a curve
with smooth, gradual transitions between numeric quantities. The self-similar, recursive
structure based on digit count is immediately visible in the PCA plot, as there is a
much more striking division and a separation between numbers of different digit size.
It is remarkable how the replication of the same structures for different digit counts
stays present in different models.

![SVD visualization of Llama
embeddings compared to random sample.](plots/Llama-3.2-1B-Instruct_02_svd_components_gradient_v1.svg){#fig:llama-svd}

The SVD plot shows a linear arrangement - numbers form an almost straight diagonal line
from small to large values. However, this apparently linear picture changes drastically
once we also take into account the third component, showing a much more tridimensional
picture, looking like the curves observed in PCA after a diagonal rotation. The 3D
projection ([@fig:llama-svd-3d]) really shines here, showing how the same structure is
repeating at different distances and angles of rotation.

![Llama SVD visualizations of first and third
component.](plots/Llama-3.2-1B-Instruct_svd_components_gradient_0-2_v1.svg){#fig:llama-svd-digits}

![3D projection of LLaMa 3.2 embeddings after
SVD](plots/llama_svd_3d.png){#fig:llama-svd-3d}

#### Explained variance

![Llama PCA explained
variance.](plots/Llama-3.2-1B-Instruct_01_pca_variance_overview_v1.svg){#fig:llama-variance}

The explained variance plot reveals slightly higher information concentration
than OLMo-2.  Llama-3.2 reaches 90% explained variance with approximately 500
components compared to OLMo-2's 500 components. This suggests more efficient
numerical encoding, despite the smaller model size. A possible reason is the bigger
training dataset of LLaMa 3 [@grattafiori2024], having 15 trillion ingested during training against OLMo's
5 trillion [@olmo2025] could have lead to different convergence patterns.

\clearpage

### Non-Linear analysis

These nonlinear projections reveal dramatically different organizational
patterns from both the linear methods and from OLMo-2's structures.

![2D t-SNE structure in
Llama](plots/Llama-3.2-1B-Instruct_tsne_components_gradient_digit_v1.svg){#fig:llama-tsne}

The t-SNE visualization ([@fig:llama-tsne]) is very unusual, and show continuous,
winding structures that might look like they had been uncoiled or unwound from a
higher-dimensional spiral arrangement. The mathematical progression follows these
winding paths smoothly, and switching the coloring to highlight the hundreds digit
reveals that each filament clusters neatly for its hundreds' group. We can also see
another interesting phenomenon: some numbers divisible by 100 cluster between the teal
and green filaments.

![3D t-SNE structure in Llama](plots/llama_tsne_3d.png){#fig:llama-tsne-3d}

By applying the procedure in 3D, the helical structures become more visible. Interactive
visualizations are helpful in this context, as they give a sense of depth, but even in a
still frame the helices are clearly visible. With this visualization we get singular helices
clustered by the hundreds' digit, also due to setting a low perplexity value to let the
structures emerge locally.

![UMAP in Llama with cosine similarity.](plots/Llama-3.2-1B-Instruct_10_umap_cosine_digit_visualizations_v1.svg){#fig:llama-umap-cosine-digits}

![UMAP in Llama with Euclidean distance.](plots/Llama-3.2-1B-Instruct_11_umap_euclidean_components_gradient_v1.svg){#fig:llama-umap-euclidean}

The UMAP visualizations ([@fig:llama-umap-cosine-digits; @fig:llama-umap-euclidean])
resemble OLMo's ones. It's also interesting to see that changing the distance function
to Euclidean doesn't have particular effects, unlike the previous OLMo visualization.
This can be an indication that the numeric data is more centered towards the mean. Even
though the 2D visualization is interesting, the 3D one ([@fig:llama-umap-3d])
complements the picture we started seeing with t-SNE, by giving the idea that beyond
having dimensions that represent the spiral structure at a cluster-local level, there
might be a more global geometric phenomenon going on, making the whole structure
approximate a half-sphere.

![3D cosine similarity UMAP in Llama.](plots/llama_umap_3d.png){#fig:llama-umap-3d}

\clearpage

### Correlation with mathematical properties


| dimension | property  | encoding   | correlation |    p\_value |
| --------- | --------- | ---------- | ----------: | ----------: |
| 417       | log       | direct     |    -0.70445 | 9.1543e-151 |
| 1929      | log       | direct     |     0.69842 | 3.7225e-147 |
| 1511      | log       | direct     |    -0.69462 | 6.3413e-145 |
| 1601      | log       | direct     |     0.68892 | 1.2059e-141 |
| 1511      | numbers   | direct     |    -0.66539 | 7.2717e-129 |
| 1929      | fibonacci | gauss      |    -0.53600 |  1.8846e-75 |
| 1601      | fibonacci | gauss      |    -0.48667 |  1.3712e-60 |
| 417       | fibonacci | gauss      |     0.46017 |  1.4848e-53 |
| 1447      | fibonacci | gauss      |    -0.40412 |  1.4218e-40 |
| 1929      | triangular | gauss |    -0.43179 |  1.1163e-46 |
| 1601      | triangular | gauss |   -0.42298 |  1.1313e-44 |
| 881       | primes     | gauss |    -0.29401 |  2.1688e-21 |
| 665       | primes     | gauss |     0.28418 |  4.9509e-20 |

: Correlations between Llama embedding dimensions and mathematical sequences.

The picture being shown through the correlations is a very surprising one, as it
reinforces clearly what was observed previously through the OLMo model. The most
correlated properties are magnitude-related, with the logarithm and the plain numbers on
top. The maximum absolute correlation with Fibonacci numbers has soared to around 0.5,
making a very strong case for these dimensions to either be feature detectors for
certain sequences, or to incidentally have connections with the sequences due to
intrinsic geometric properties. The case for the second hypothesis is also reinforced by
how strikingly we were able to show a very interesting geometric landscape for this
model, and how this seemed to coincide with examining a model that was trained on a much
larger scale.

![Dimensions strongly correlated with properties in Llama
3.2](plots/Llama-3.2-1B-Instruct_strong_property_correlations_v1.svg){#fig:llama-properties
width=45%}

Taking a look at how many dimensions are correlated quantitatively
([@fig:llama-properties]), most of them are still related to magnitude, but the number
of dimensions related to mathematical properties is a lot higher. The rise in the
correlation coefficients with respect to the OLMo model might be because of scale and
the large quantity of additional tokens Llama was trained on, but further analysis would
be needed to establish a causal link, since this can be influenced by a lot of
confounding variables, such as data quality or training.

# Conclusions

With visualizations and the data, we were able to paint a picture of interesting
phenomena coming through the embeddings layers of two distinct LLMs, OLMo and Llama.
The most important difference between these two models that we were able to discuss is
the scale, and we have shown that with bigger scale come more visible patterns, in both
correlations with mathematical sequences and geometric structure. This seems to be in
line with the Platonic Representation hypothesis, as it is the fact that similar
geometric structures seem to appear in different models.

We have also shown that numerical embeddings lie on low-dimensional manifolds and have
semantic relationships with the symbols they represent, given the numerous and very
significant correlations there are between embedding dimensions and the magnitude of the
represented number (> 0.7). We have also shown features highly correlated with the
belonging of the represented number to certain numerical sequences, like Fibonacci (>
0.5), prime and triangular numbers. The fact that these relationships seem to strengthen
with the scale of the model seem to contribute to the case that these representations
may be converging.

## Limitations

The analysis was limited to two models and a specific set of mathematical
properties. Further investigation with controlled experiments would be
needed to establish causal relationships.

Whether these findings extend to larger models, different architectures, or
other mathematical domains remains to be determined. The observed structures may
reflect training data properties as much as emergent organizational principles.

## Future Directions

If it's true that numerical representations end up converging to certain geometric
dispositions, there's a lot we can follow this up with:

- Is it possible to model this mathematical landscape through mathematical definitions?
- Is there a way to see the final nature of the mathematical landscape numerical
  representations tend to form?
- Can the geometry adopted by these models teach us more about mathematics themselves?
- If the convergence is influenced by beneficial properties from an information-theory
  perspective, could the same structures influence human cognition?

There are a lot of immediate research directions to expand this work on, like
contrasting and comparing with more models, looking for correlations with more
mathematically interesting sequences, and working towards understanding the causal link
behind the high correlations. There's also creating new embedding schemes for numbers
inspired by these findings, and seeing if they can improve performance in a variety of
tasks.

There are a lot of possibilities to cover, that can lead to better understanding of
human and machine cognition. In its small scope, I hope this work can be convincing in
showing this is a worthwhile endeavor, so that after models learn everything about us,
we may look back into them to learn more about ourselves.
