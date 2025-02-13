"""
Probability distribution functions for generating dungeons.
"""

import numpy as np
from dataclasses import dataclass
from operator import mul
from functools import reduce
from itertools import product
from collections import Counter


@dataclass
class Roll:
    n: int
    d: int
    c: int = 0  # constant


@dataclass
class LastRoomProbability:
    """
    Class mapping numbers of room to a probability of being the last.
    """

    # the minimum number of rooms this object accounts for
    min_rooms: int

    # the probability that room (i - min_rooms) is the last one
    p: np.ndarray

    def __getitem__(self, item):
        return self.p[item - self.min_rooms]

    @property
    def max_rooms(self):
        return self.min_rooms + len(self) - 1

    def __len__(self):
        return len(self.p)

    def quantize(self, quantum: float = 0.01):
        self.p = np.round(self.p / quantum) * quantum


@dataclass
class DiscreteDistribution:
    """
    Discrete probability distribution.
    """

    min_value: int
    p: np.array

    def __len__(self):
        return len(self.p)

    @property
    def max_value(self):
        return self.min_value + len(self) - 1

    def __getitem__(self, item):
        return self.p[item]


def compute_sum_probabilities(rolls: list[Roll] | Roll) -> DiscreteDistribution:
    """
    Compute the probabilities of all possible results of a dice roll.

    :return: a discrete distribution
    """
    if isinstance(rolls, Roll):
        rolls = [rolls]

    # number of combinations of each possible die, including repeatitions
    num_combinations_per_die_type = [roll.d**roll.n for roll in rolls]
    num_combinations = reduce(mul, num_combinations_per_die_type, 1)

    all_ranges = [range(1, roll.d + 1) for roll in rolls for i in range(roll.n)]
    cart_product = product(*all_ranges)

    possible_dice_sums = [sum(result) for result in cart_product]
    counter = Counter(possible_dice_sums)
    constant_sum = sum(roll.c for roll in rolls)

    min_value_without_constant = min(possible_dice_sums)
    p = np.zeros(len(counter), dtype=float)
    for value, count in counter.items():
        p[value - min_value_without_constant] = count / num_combinations

    min_value = min_value_without_constant + constant_sum
    dist = DiscreteDistribution(min_value, p)

    return dist


def divide_distribution(
    distribution: DiscreteDistribution, constant: float
) -> DiscreteDistribution:
    """
    Divides the distribution by a constant, splitting probability mass proportionally
    when the result is non-integer.

    This simulates the situation when the distribution stating the number of
    rooms in a dungeon is divided to account for the number of rooms in a
    subpath of it.

    Parameters:
        distribution: The original (discrete) probability distribution.
        constant (int or float): The constant to divide the values by.

    Returns:
        The new probability distribution after division.
    """
    min_value = distribution.min_value
    p = distribution.p

    # Calculate new min and max
    new_min = int(np.floor(min_value / constant))
    new_max = int(np.floor((min_value + len(p) - 1) / constant))
    num_bins = new_max - new_min + 1

    # Initialize the new distribution
    new_distribution = np.zeros(num_bins)

    # Compute the original values
    original_values = min_value + np.arange(len(p))

    # Compute the divided values
    divided_values = original_values / constant

    # Split probability mass proportionally
    for i, value in enumerate(divided_values):
        lower = int(np.floor(value))  # Lower integer
        upper = lower + 1  # Upper integer
        fraction = value - lower  # Fractional part

        # Distribute probability mass
        if new_min <= lower <= new_max:
            new_distribution[lower - new_min] += (1 - fraction) * p[i]

        if new_min <= upper <= new_max:
            new_distribution[upper - new_min] += fraction * p[i]

    # hack to fix imprecisions
    new_distribution = new_distribution / new_distribution.sum()

    return DiscreteDistribution(new_min, new_distribution)


def compute_last_room_probability(
    distribution: DiscreteDistribution,
) -> LastRoomProbability:
    """
    Calculate the probability of each position being the last,
    given that we've reached that position.

    Args:
      distribution: distribution probability of the total number of rooms

    Returns:
      an array of the conditional probabilities for each quantity of rooms.
    """
    p = distribution.p

    accumulated_probability_mass = p.cumsum()
    remaining_mass = 1 - accumulated_probability_mass
    remaining_mass = np.concatenate([[1], remaining_mass])
    conditional_probs = p / remaining_mass[:-1]
    new_p = np.array(conditional_probs)

    lrp = LastRoomProbability(distribution.min_value, new_p)
    return lrp


def compute_probabilities_after_split(
    distribution: DiscreteDistribution, min_split: float = 0.2, max_split: float = 0.8
) -> DiscreteDistribution:
    """
    Compute the probabilities of the total number of rooms down a path.

    This is supposed to be called at each bifurcation.

    :param distribution: the distribution probability of the total number of
        rooms in the dungeon
    :param min_split: minimum room budget for each path
    :param max_split: maximum room budget
    :return: a distribution probability over the total number of rooms down one
        of the paths. The other is supposed to continue after the first one is
        depleted.
    """
    # accumulate values before averaging -- from 0 rooms to the maximum
    accumulator = np.zeros(distribution.max_value, dtype=float)
    num_splits = 0

    new_min = 1e10

    # marginalize over the latent split percentage
    step = 0.05
    split = min_split
    while split <= max_split:
        num_splits += 1
        divider = 1 / split  # turn 0.2 into 5
        possible_dist = divide_distribution(distribution, divider)

        i = possible_dist.min_value
        j = possible_dist.max_value
        accumulator[i : j + 1] += possible_dist.p

        new_min = min(new_min, possible_dist.min_value)

        split += step

    accumulator = accumulator[new_min:]
    accumulator = accumulator[accumulator > 0]
    new_p = accumulator / num_splits
    assert np.isclose(new_p.sum(), 1)

    new_dist = DiscreteDistribution(new_min, new_p)

    return new_dist


def generate_last_room_probabilities(
    rolls: Roll | list[Roll], max_splits: int = 3
) -> list[LastRoomProbability]:
    """
    Generate the last room probabilities.

    It considers the given rolls to determine the original number of rooms in
    the dungeon. It takes into account up to the given number of splits.

    :param rolls: dice roll for the number of rooms
    :param max_splits: how many path splits there were
    :return: a list such that the i-th element has the last room probabilities
        when there were i splits.
    """
    last_dist = compute_sum_probabilities(rolls)
    last_room_probs = []

    for i in range(max_splits + 1):
        lrp = compute_last_room_probability(last_dist)
        lrp.quantize(0.01)
        last_room_probs.append(lrp)

        last_dist = compute_probabilities_after_split(last_dist)

    return last_room_probs
