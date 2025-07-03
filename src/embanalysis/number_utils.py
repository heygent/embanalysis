import numpy as np
from sympy import nextprime, prevprime
from sympy.ntheory.primetest import is_square, isprime


def distance_to_nearest_prime(numbers):
    """Calculate distance to closest prime for each number"""
    distances = []

    for n in numbers:
        if n <= 1:
            distances.append(2 - n)  # Distance to first prime (2)
        elif isprime(n):
            distances.append(0)  # Already prime
        else:
            # Find previous and next primes
            prev_p = prevprime(n) if n > 2 else 2
            next_p = nextprime(n)

            # Distance to nearest
            dist_prev = n - prev_p
            dist_next = next_p - n
            distances.append(min(dist_prev, dist_next))

    return np.array(distances)


def squareness_score(numbers):
    """Continuous measure of how "square-like" a number is"""
    scores = []

    for n in numbers:
        if n <= 0:
            scores.append(0)
        else:
            sqrt_n = np.sqrt(n)
            # How close is sqrt to an integer?
            fractional_part = sqrt_n - int(sqrt_n)
            # Convert to score: 1.0 for perfect squares, decreasing with distance
            squareness = 1.0 - min(fractional_part, 1 - fractional_part) * 2
            scores.append(squareness)

    return np.array(scores)


def is_fibonacci(n):
    """Check if a number is a Fibonacci number"""
    return n >= 0 and (is_square(5 * n * n + 4) or is_square(5 * n * n - 4))


def fibonacci_proximity(numbers, max_fib=1000):
    """Distance to nearest Fibonacci number"""
    # Generate Fibonacci sequence up to reasonable limit
    fibs = [0, 1]
    while fibs[-1] < max_fib:
        fibs.append(fibs[-1] + fibs[-2])
    fibs = np.array(fibs)

    proximities = []
    for n in numbers:
        distances = np.abs(fibs - n)
        min_distance = np.min(distances)
        proximities.append(-min_distance)  # Negative so Fib numbers have highest values

    return np.array(proximities)


def golden_ratio_resonance(numbers):
    """How well a number fits golden ratio patterns"""
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    resonances = []
    for n in numbers:
        if n <= 0:
            resonances.append(0)
        else:
            # Check if n/phi^k is close to an integer for various k
            max_resonance = 0
            for k in range(-5, 6):  # Check various powers of phi
                ratio = n / (phi**k)
                closeness_to_integer = 1 - min(
                    ratio - int(ratio), int(ratio) + 1 - ratio
                )
                max_resonance = max(max_resonance, closeness_to_integer)

            resonances.append(max_resonance)

    return np.array(resonances)

def sequence_proximity(numbers, sequence):
    """Distance to nearest number in a given sequence"""
    sequence = np.array(sequence)
    proximities = []

    for n in numbers:
        distances = np.abs(sequence - n)
        min_distance = np.min(distances)
        proximities.append(-min_distance)  # Negative so closer numbers have higher values

    return np.array(proximities)