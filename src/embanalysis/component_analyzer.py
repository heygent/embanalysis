import numpy as np
import pandas as pd

from sympy import isprime
from sympy.ntheory.primetest import is_square

from embanalysis.analyzer import EmbeddingsAnalyzer
from embanalysis.number_utils import (
    distance_to_nearest_prime,
    fibonacci_proximity,
    golden_ratio_resonance,
    is_fibonacci,
    squareness_score,
)


def make_properties_df(numbers):
    return pd.DataFrame(
        {
            "magnitude": np.log10(numbers + 1),
            "is_even": (numbers % 2 == 0).astype(int),
            "is_prime": np.array([isprime(n) for n in numbers], dtype=int),
            "prime_proximity": -distance_to_nearest_prime(numbers),
            "perfect_square": np.array([is_square(n) for n in numbers], dtype=int),
            "squareness": squareness_score(numbers),
            "is_fibonacci": np.array([is_fibonacci(n) for n in numbers], dtype=int),
            "fibonacci_proximity": fibonacci_proximity(numbers),
            "golden_ratio_resonance": golden_ratio_resonance(numbers),
            "digit_count": [len(str(n)) for n in numbers],
        }
    )


class ComponentAnalyzer:
    """
    A class to analyze components of embeddings.
    """

    def __init__(self, analyzer: EmbeddingsAnalyzer):
        self.analyzer = analyzer
