from typing import Tuple
import argparse

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import statistics

from tqdm.auto import trange


def binary_rand(n: int) -> NDArray:
    """
    Generates random binary array of size n.

    Args:
        n (int): Size of the generated array

    Returns:
        Binary NumPy array of size n.
    """
    return np.random.randint(0, 2, (n,))


def measure(bits: NDArray, a_bases: NDArray, b_bases: NDArray) -> NDArray:
    """
    Simulates quantum measurement.
    Leaves bits where the sending base and the recieving base are equal, and
    generates replaces with random bits where the bases are different.

    Args:
        bits (NDArray): Binary array that represents the bits.
        a_bases (NDArray): Binary array that represents the sending bases.
        b_bases (NDArray): Binary array that represents the measurement bases.
    
    Returns:
        measured_bits: Binary array that represents the measured bits.
    """
    return np.where(a_bases == b_bases, bits, np.random.randint(0, 2, (bits.shape[0],)))


def sender(n_bits: int) -> Tuple[NDArray, NDArray]:
    """
    Generates two binary NumPy array (bits and bases) and returns then.

    Args:
        n_bits (int): Number of bits to sent, the length of the NumPy array.

    Returns:
        2 binary numpy array, representing the bits and the bases
    """
    return binary_rand(n_bits,), binary_rand(n_bits,)


def reciever(bits: NDArray, bases: NDArray):
    """
    Measures recived bits with random bases.
    
    Args:
        bits (NDArray): Binary array that represents the bits.
        bases (NDArray): Binary array that represents the sending bases.
    
    Returns:
        2 numpy arrays, that represent the measured bits and the bases they were measured in.
    """
    b_bases = binary_rand(bits.size)
    b_bits = measure(bits, bases, b_bases)
    return b_bits, b_bases


def accuracy(a_bits: NDArray, a_bases: NDArray, b_bits: NDArray, b_bases: NDArray,) -> float:
    """
    Calculate the transmision accuracy of quantum bits, i.e. the percentage of bits that have
    the same information with the same base.

    Args:
        a_bits (NDArray): Binary array that represents the sent bits.
        b_bits (NDArray): Binary array that represents the recieved bits.
        a_bases (NDArray): Binary array that represents the sending bases.
        b_bases (NDArray): Binary array that represents the measurement bases.

    Returns:
        The transmission accuracy of the bits.
    """
    n_bits = a_bits.size
    correct_bit = a_bits == b_bits
    correct_base = a_bases == b_bases
    
    a_bits = a_bits[correct_base]
    b_bits = b_bits[correct_base]
    correct_bit = (a_bits == b_bits)
    acc = correct_bit.sum() / n_bits

    return acc


def simulate(n_bits: int, eve: bool = False) -> float:
    """
    Simulates the full process of the BB84 protocol.

    Args: 
        n_bits (int): The number of bits to be transmittes in the simulation.
        eve (int): If True, simulates the transmision with presence of an evesdropper.

    Returns:
        The transmission accuracy of the bits.
    
    """
    a_bits, a_bases = sender(n_bits)    # Alice sends random bits with random bases

    if eve:
        e_bits, e_bases = reciever(a_bits, a_bases) # Eve measures bits in random bases and 
                                                    # trasmits accordingly.
    else:
        e_bits, e_bases = a_bits, a_bases

    b_bits, b_bases = reciever(e_bits, e_bases)

    return accuracy(a_bits, a_bases, b_bits, b_bases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', default=168, help='Number of bits transmitted in each simulation')
    parser.add_argument('--sims', default=10000, help='Number of simulations')
    parser.add_argument('--output', default=None, help='Output file')

    args = parser.parse_args()
    n_bits: int = int(args.bits)
    n_sims: int = int(args.sims)

    no_eve_acc = list()
    eve_acc = list()
    for _ in trange(n_sims):
        no_eve_acc.append(simulate(n_bits, False))
        eve_acc.append( simulate(n_bits, True))

    if args.output is not None:
        df = pd.DataFrame()
        df['eve'] = eve_acc
        df['no_eve'] = no_eve_acc
        df.to_csv(args.output)
    print(f'w/o eve:\t{statistics.mean(no_eve_acc):.5f}\nw/ eve:\t\t{statistics.mean(eve_acc):.5f}')
