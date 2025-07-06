## Goals of the iscc-vdb project

- provide a specialized ISCC indexing and search library for python as an embedded binary vector database based
    on usearch.
- provide a command line tool for creating and managing indexes
- provide an optional REST api web service built with the blacksheep for indexing and querying ISCCs and
    SIMPRINTS

## Challenges

- Composite ISCC-CODEs need to be first decomposed into their individual units before indexing
- Comparing and matching ISCCs makes only sense if we compare ISCCs of the same MainType & SubType
- We need to manage a separate indexes per ISCC-UNIT MainType-SubType
- For granular matching capabilities with need to have additional indexes for SIMPRINTs
- For SIMPRINTS we need to somehow track to which asset and which section within that asset it belongs to

**ANNS for Variable-Length, Prefix-Aligned Bit Vectors**:

Our requirement is a specific, complex variant of the ANNS problem: performing efficient and scalable similarity
searches over a corpus of bit vectors characterized by **variable lengths**. Specifically, the vectors originate
from different ISCC collections, resulting in lengths of 64, 128, 192, or 256 bits. A critical structural
property defines this dataset: for any given input, shorter ISCCs generated are guaranteed to be **prefixes** of
their longer counterparts (matryoshka representation learning). The objective is to retrieve, for a query vector
of any valid length, the most similar vectors from the corpus, regardless of their length. Similarity must be
assessed based on the Hamming distance computed over the **common prefix length** shared between the query and
potential neighbors. This requirement renders standard ANNS methodologies, typically designed for fixed-length
vectors and conventional distance metrics inadequate. The inherent variability in vector length and the specific
prefix-based similarity definition necessitate a tailored approach.

## Ideas

**NPHD - A Custom Metric**:

```python
def iscc_nph_distance(a, b):
    # type: (bytes, bytes) -> dict
    """
    Calculate Normalized Prefix Hamming Distance (NPHD) between two bit vectors packed as byte strings
    (8-bits per byte).

    NPHD is defined as the Hamming distance of their common prefix, normalized by
    the length of that common prefix in bits.

    :param a: First byte string
    :param b: Second byte string
    :return: Dictionary with NPHD score and common prefix length
             {"distance": float, "common_prefix_bits": int}
    """
    common_bytes = min(len(a), len(b))
    common_bits = common_bytes * 8
    if common_bits == 0:
        return {"distance": 0.0 if (len(a) == 0 and len(b) == 0) else 1.0, "common_prefix_bits": 0}
    ba, bb = bitarray(), bitarray()
    ba.frombytes(a[:common_bytes])
    bb.frombytes(b[:common_bytes])
    hd = count_xor(ba, bb)
    return {"distance": hd / common_bits, "common_prefix_bits": common_bits}
```

**Important Properties of NPHD**:

- NPHD is a **valid metric** for prefix-compatible binary codes (satisfies all metric axioms)
- Unlike standard Hamming distance, NPHD correctly handles variable-length comparisons
- For prefix-compatible codes: 5 bits difference in 64 bits (7.8%) vs 5 bits in 256 bits (1.95%)
- The normalization ensures proportional distance measurement across different vector lengths

**Usearch Length Signalling**:

Usearch only supports fixed-dimensional vectors. To handle variable-length ISCCs (64, 128, 192, or 256 bits), we
encode length information into the vector itself. Our approach:

1. **Vector Format**: 264-bit vectors (33 bytes total)

    - First 8 bits (1 byte): Length signal indicating actual ISCC length in bytes
    - Next 256 bits (32 bytes): ISCC body (padded with zeros for shorter ISCCs)

2. **Usearch Configuration**:

    - `ndim=264` (number of bits, not bytes)
    - `dtype=ScalarKind.B1` (binary data type, 1 bit per dimension)
    - Custom NPHD metric via `CompiledMetric` with Numba

3. **Storage Overhead**:

    - 64-bit ISCC: 33 bytes stored vs 8 bytes actual (75.8% overhead)
    - 128-bit ISCC: 33 bytes stored vs 16 bytes actual (51.5% overhead)
    - 192-bit ISCC: 33 bytes stored vs 24 bytes actual (27.3% overhead)
    - 256-bit ISCC: 33 bytes stored vs 32 bytes actual (3.0% overhead)

    If most ISCCs are 256-bit, the average overhead approaches just 3%, making this a pragmatic solution.

4. **Implementation Example**:

```python
def prepare_iscc_for_index(iscc_bytes):
    """Prepare an ISCC for indexing with length signalling."""
    bit_array = np.zeros(264, dtype=np.uint8)

    # First 8 bits: length signal
    bit_array[0:8] = np.unpackbits(np.array([len(iscc_bytes)], dtype=np.uint8))

    # Next bits: actual ISCC data
    iscc_bits = np.unpackbits(np.array(list(iscc_bytes), dtype=np.uint8))
    bit_array[8:8+len(iscc_bits)] = iscc_bits

    # Pack into bytes for storage
    return np.packbits(bit_array)
```

## Example ISCC data

```
{
  "@context": "http://purl.org/iscc/context",
  "@type": "TextDigitalDocument",
  "$schema": "http://purl.org/iscc/schema",
  "iscc": "ISCC:KACZH265WE3KJOSRJT3OCVAFMMNYPEWWFTXNHEFX65YXQN4VEJVNKUQ",
  "name": "Economiche Gemme Poetiche",
  "description": "\"Economiche Gemme Poetiche\" è una raccolta di poesie brillanti, malinconiche, ironiche ...",
  "meta": "data:application/ld+json;base64,eyIkc2NoZW1hIjoiaHR0cHM6Ly9wdXJsLm9yZy9pc2NjL3NjaGVtYS9pc2JuLmpzb24",
  "creator": "Mario Sargeni",
  "keywords": [
    "poesia",
    "moderna",
    "contemporanea"
  ],
  "mode": "text",
  "filename": "9788832539868.epub",
  "filesize": 1055746,
  "mediatype": "application/epub+zip",
  "characters": 20116,
  "parts": [
    {
      "iscc": "ISCC:KEAZS3YHSYMWM2U2VZJ6MX73GJQNSDKNRAMMWCIXGI",
      "mode": "image",
      "filename": "ebook_image_121852_16730c2b28790320.jpg",
      "filesize": 139939,
      "mediatype": "image/jpeg",
      "width": 578,
      "height": 821,
      "generator": "iscc-sdk - v0.8.0",
      "thumbnail": "data:image/webp;base64,UklGRrYHAABXRUJQVlA4IKoHAAAQJQCdASpaAIAAPwFmqFArJSOis1ueuWAgCWxdg...",
      "datahash": "1e200d4d8818cb091732ddb152930644838a5dab0c3449759fe65be2e2be34a0fe74",
      "@type": "ImageObject",
      "units": [
        "ISCC:EEDZS3YHSYMWM2U2GPOQ4LBTZXKDI3QHSIMWM2U27HOQ4JBTZTKDJ4Y",
        "ISCC:GAD24U7GL75TEYGZ54AFE2KTU54UUG265UKXPK4PKPWYYQZUYN6KWMY",
        "ISCC:IADQ2TMIDDFQSFZS3WYVFEYGISBYUXNLBQ2ES5M74ZN6FYV6GSQP45A"
      ]
    }
  ],
  "features": [
    {
      "maintype": "content",
      "subtype": "text",
      "version": 0,
      "simprints": [
        "8IAnFvInk24iEkDGoxfPid4DLgKjoHcf9U4-_3zPEVk",
        "GH7W703iOzPEyhD295s0nrKPNujISF5YBbWDpGwiK1Q",
        "..."
      ],
      "offsets": [
        0,
        698,
        "..."
      ],
      "sizes": [
        698,
        469,
        "..."
      ]
    }
  ],
  "generator": "iscc-sdk - v0.8.0",
  "thumbnail": "data:image/webp;base64,UklGRkYGAABXRUJQVlA4IDoGAABwHwCdASpgAIAAPv1oqlArKqQis1p+uWAfiWgA1Efiw...",
  "metahash": "1e20d17598246a521e5ab47e808ed414a5dc191e0403d1e24aaf3df9dbb8618126e6",
  "datahash": "1e2071783795226ad5528224dddffa9d4831d799bdfb5a20b24947ea2e3c869d0eaf",
  "units": [
    "ISCC:AADZH265WE3KJOSR5K67QJEF5JHLF2REJJYVI4ZYKJ727JU2ZX2AHNQ",
    "ISCC:EADUZ5XBKQCWGG4HYIKX7CNPQMFTPTWEUCQLXFJWC25TKM645KYUSNQ",
    "ISCC:GADZFVRM53JZBN7XOOT3Y6FL372G2GY6PEKRY43JIJ6KV4GH5P7NN4A",
    "ISCC:IADXC6BXSURGVVKSQISN3X72TVEDDV4ZXX5VUIFSJFD6ULR4Q2OQ5LY"
  ],
  "isbn": "9788832539868",
  "imprint": "Passerino",
  "publisher": "Passerino",
  "suppliers_publisher_id": "3c434937-f6e8-4723-a4b2-4b54269f52e0",
  "country": "IT",
  "pubdate": "20190314",
  "md5": "124c664e112e68759695f71059d43143",
  "tdm_permitted": false,
  "bisac": [
    "POE000000"
  ],
  "thema": [
    "DC"
  ],
  "contributors": [
    {
      "name": "Mario Sargeni",
      "role": "A01",
      "sequence": 1
    }
  ]
}
```

## Key Insights and Clarifications

### 1. **NPHD is a Valid Metric**

NPHD satisfies all metric axioms (non-negativity, identity, symmetry, triangle inequality) when used with
prefix-compatible binary codes. This means it will work correctly with HNSW indexing without degrading search
quality.

### 2. **Binary Vectors in Usearch**

- Use `ndim` to specify the number of **bits**, not bytes
- Use `dtype=ScalarKind.B1` for binary data (1 bit per dimension)
- Data is packed 8 bits per byte using `np.packbits`/`np.unpackbits`
- For 264-bit vectors: `ndim=264`, which stores as 33 bytes

### 3. **Custom Metrics with Binary Data**

Usearch supports custom metrics for binary vectors through:

- Python: `CompiledMetric` with Numba-compiled functions
- C/C++: Functions matching `usearch_metric_t` signature
- Rust: `MetricFunction::B1X8Metric` implementations

Example implementation:

```python
from numba import cfunc, types, carray
from usearch.index import CompiledMetric, MetricKind, MetricSignature

@cfunc(types.float32(types.CPointer(types.uint8), types.CPointer(types.uint8)))
def nphd_metric(a_ptr, b_ptr):
    # Each vector is 33 bytes (264 bits / 8)
    a_bytes = carray(a_ptr, 33)
    b_bytes = carray(b_ptr, 33)

    # First byte contains length signal
    len_a = a_bytes[0]
    len_b = b_bytes[0]
    common_bytes = min(len_a, len_b)

    if common_bytes == 0:
        return 1.0 if (len_a != 0 or len_b != 0) else 0.0

    # Calculate Hamming distance over common prefix
    hamming_dist = 0
    for i in range(1, common_bytes + 1):
        xor_result = a_bytes[i] ^ b_bytes[i]
        while xor_result:  # Brian Kernighan's algorithm
            hamming_dist += 1
            xor_result &= xor_result - 1

    return hamming_dist / (common_bytes * 8.0)

metric = CompiledMetric(
    pointer=nphd_metric.address,
    kind=MetricKind.Unknown,
    signature=MetricSignature.ArrayArray
)
```

### 4. **Why Standard Hamming Distance Doesn't Work**

Standard Hamming distance treats all bit differences equally, regardless of vector length:

- 5 bits different in 64-bit vectors = distance of 5
- 5 bits different in 256-bit vectors = distance of 5

This fails to capture that 5/64 (7.8%) is more significant than 5/256 (1.95%). NPHD normalizes by the common
prefix length, providing proportional distance measurement.

### 5. **Storage Efficiency Considerations**

The length signalling approach with 33-byte vectors is efficient when:

- Most ISCCs are 256-bit (only 3% overhead)
- Simplicity is prioritized over absolute minimal storage
- You need a single index rather than managing multiple indices

For datasets with mostly shorter ISCCs, consider alternative approaches like separate indices per ISCC length.
