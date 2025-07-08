from collections.abc import Iterable
import functools
import numpy as np
from scipy.ndimage import gaussian_filter1d

from sympy.ntheory.generate import sieve

# properties = {
#     "magnitude": np.log10(numbers + 1),
#     "is_even": (numbers % 2 == 0).astype(int),
#     "is_prime": np.array([isprime(n) for n in numbers], dtype=int),
#     "prime_proximity": -distance_to_nearest_prime(numbers),
#     "perfect_square": np.array([is_square(n) for n in numbers], dtype=int),
#     "squareness": squareness_score(numbers),
#     "is_fibonacci": np.array([is_fibonacci(n) for n in numbers], dtype=int),
#     "fibonacci_proximity": fibonacci_proximity(numbers),
#     "golden_ratio_resonance": golden_ratio_resonance(numbers),
#     "digit_count": [len(str(n)) for n in numbers],
# }


def generate_triangular(max_value: int) -> np.ndarray:
    """Generate triangular numbers up to max_value."""
    triangular = []
    i = 0
    while i * (i + 1) // 2 < max_value:
        triangular.append(i * (i + 1) // 2)
        i += 1
    return np.array(triangular)


def generate_factorials(max_value: int) -> np.ndarray:
    """Generate factorial numbers up to max_value."""
    factorials = []
    factorial = 1
    i = 0
    while factorial <= max_value:
        factorials.append(factorial)
        i += 1
        if len(factorials) > 0:
            factorial *= i
    return np.array(factorials)


def generate_fibonacci(max_value: int) -> np.ndarray:
    """Generate Fibonacci numbers up to max_token."""
    fibs = [0, 1]
    while fibs[-1] < max_value:
        fibs.append(fibs[-1] + fibs[-2])
    return np.array(fibs)


def generate_primes(max_value: int) -> np.ndarray:
    return np.fromiter(sieve.primerange(0, max_value), dtype=int)


def direct_encoded_base_sequences(max_token: int = 1000):
    return {
        "numbers": np.arange(max_token),
        "sin": np.sin(np.arange(max_token)),
        "cos": np.cos(np.arange(max_token)),
        # "even": np.arange(0, max_token * 2, 2),
        "log": np.log10(np.arange(max_token) + 1),
    }


def binary_encoded_base_sequences(max_token: int = 1000):
    return {
        "primes": generate_primes(max_token),
        "fibonacci": generate_fibonacci(max_token),
        "triangular": generate_triangular(max_token),
        "factorials": generate_factorials(max_token),
    }


# Encoders


def one_hot_encode(sequence, size):
    """Binary encoding: 1 if number is in sequence, 0 otherwise."""
    return np.isin(np.arange(size), sequence).astype(int)


def one_hot_gaussian_smooth(binary, sigma=2.0):
    if len(binary) > 1:
        smooth = gaussian_filter1d(binary.astype(float), sigma=sigma)
        return smooth
    else:
        return binary.astype(float)


def fourier_encode(sequence, max_token: int, period: int):
    """
    Encode a sequence using Fourier features with a single period.

    Args:
        sequence: Array of values in the sequence (e.g., [1, 1, 2, 3, 5, 8, ...])
        max_token: Size of the output encoding
        period: Single period T for Fourier encoding

    Returns:
        tuple: (cos_encoding, sin_encoding) both of shape (max_token,)
    """
    # Create array to hold sequence values at their positions
    sequence_values = np.zeros(max_token)

    # Place sequence values at their corresponding positions
    for val in sequence:
        if val < max_token:
            sequence_values[val] = val

    # Compute Fourier components
    cos_component = np.cos(2 * np.pi * sequence_values / period)
    sin_component = np.sin(2 * np.pi * sequence_values / period)

    return cos_component, sin_component


@functools.cache
def make_sequences(max_token: int):
    return direct_encoded_base_sequences(max_token) | binary_encoded_base_sequences(
        max_token
    )


def make_encoded_sequences(
    max_token: int,
    sigma: float = 2.0,
    fourier_encoding: bool = False,
    fourier_periods: Iterable[int] = (1, 2, 5, 10, 100),
):
    encoded_sequences = {}

    direct_sequences = direct_encoded_base_sequences(max_token)
    for name, seq in direct_sequences.items():
        encoded_sequences[name, "direct"] = seq

    binary_sequences = binary_encoded_base_sequences(max_token)
    for name, seq in binary_sequences.items():
        one_hot = one_hot_encode(seq, max_token)
        encoded_sequences[name, "binary"] = one_hot
        encoded_sequences[name, "gauss"] = one_hot_gaussian_smooth(one_hot, sigma=sigma)
        if fourier_encoding:
            for period in fourier_periods:
                (
                    encoded_sequences[name, f"fourier_cos_T{period}"],
                    encoded_sequences[name, f"fourier_sin_T{period}"],
                ) = fourier_encode(seq, max_token, period)

    return encoded_sequences
