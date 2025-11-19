import random
import os
SEED=10
# Set seed for reproducibility
random.seed(SEED)
TARGET_SEQ_LENGTH = 100  # Each sequence will be exactly this long

def normalize_lengths(num_parts, min_part_length=1):
    """
    Normalize a random list of part lengths to sum up to TARGET_SEQ_LENGTH.
    Ensures each part has at least length 1.
    """
    raw_lengths = [random.randint(1, 100) for _ in range(num_parts)]
    total = sum(raw_lengths)
    scaled = [max(1, round(x * TARGET_SEQ_LENGTH / total)) for x in raw_lengths]

    return scaled


def rand_digits(k):
    """Return a string of k random digits (1-9)"""
    return ''.join(random.choices('123456789', k=k))

def rand_letters(letter, k):
    """Return a string consisting of the given letter repeated k times"""
    return letter * k

def generate_example(is_positive):
    """
    Generate a single sequence with either the correct pattern (positive)
    or a misleading pattern (negative) of letter order.
    """
    # 5 digit parts and 4 letter parts = 9 segments
    part_lengths = normalize_lengths(9)
    d1, l1, d2, l2, d3, l3, d4, l4, d5 = part_lengths

    if is_positive:
        seq = (
            rand_digits(d1) +
            rand_letters('a', l1) +
            rand_digits(d2) +
            rand_letters('b', l2) +
            rand_digits(d3) +
            rand_letters('c', l3) +
            rand_digits(d4) +
            rand_letters('d', l4) +
            rand_digits(d5)
        )
    else:
        seq = (
            rand_digits(d1) +
            rand_letters('a', l1) +
            rand_digits(d2) +
            rand_letters('c', l2) +  # swapped b and c
            rand_digits(d3) +
            rand_letters('b', l3) +
            rand_digits(d4) +
            rand_letters('d', l4) +
            rand_digits(d5)
        )
    return seq

def write_examples(examples, name):
    """Write a list of sequences to a file named {name}.txt"""
    with open(f"{name}.txt", 'w') as f:
        for ex in examples:
            f.write(ex + '\n')

def generate_set():
    """Generate and write 500 positive and 500 negative examples"""
    pos = [generate_example(is_positive=True) for _ in range(500)]
    neg = [generate_example(is_positive=False) for _ in range(500)]

    write_examples(pos, "pos_examples")
    write_examples(neg, "neg_examples")

    print(f"Generated 500 positive and 500 negative examples of exactly {TARGET_SEQ_LENGTH} characters")

if __name__ == "__main__":
    generate_set()
