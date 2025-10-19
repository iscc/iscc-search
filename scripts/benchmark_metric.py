"""
Benchmark comparing usearch internal hamming metric against our custom NPHD metric for index
add/search methods.

Compares "Writes per second" and "Searches per second" with 1000, 100.000, entries in indexes
with 64-bit and 256-bit vectors. For all index sizes we add entries with a single batch call, we search
entries in batches of 100.
"""


def main():
    pass


if __name__ == "__main__":
    main()
