## About the ISCC

The **International Standard Content Code (ISCC)** is a data-driven (content-derived), multi-component,
similarity-preserving, **identifier and fingerprint for digital content** that uses free, transparent
open-source software to create standardized, decentralized **ISCC-CODEs** used for the identification of raw
data, text, image, audio video and other digital content across all domains concerned with producing, processing
and distributing digital information (science, journalism, books, music, film, etc.).

Digital content is dynamic, always in motion, and acted upon globally by a variety of entities with different
interests and requirements. Digital content continuously re-encodes, resizes, and re-compresses, changing its
data as it travels through a complex network of actors and systems.

An ISCC-CODE is the result of processing the digital content using a variety of algorithms. The generated
ISCC-CODE supports data integrity verification and preserves an estimate of data, content, semantic and metadata
similarity.

Organizations, individuals and machines may generate ISCCs for numerous kinds of digital assets and use them for
identification and management of those assets.

ISCCs are neither manually nor automatically assigned to digital media assets. Instead, ISCCs are derived from
media assets according to the procedures described by ISO 24138:2024. Unrelated parties can independently derive
the same ISCC from a given media asset.

ISCCs exclusively reference media assets without any implication about ownership. As such, ISCCs are not managed
authoritatively by any institution or entity.

The ISCC enables interoperability between different actors and systems using digital assets and supports
scenarios that require content deduplication, database synchronization and indexing, integrity verification,
timestamping, versioning, data provenance, similarity clustering, anomaly detection, usage tracking, allocation
of royalties, fact-checking and general digital asset management use-cases.

### ISCC Important Terms

- **ISCC** - Any ISCC-CODE, ISCC-UNIT or ISCC-ID
- **ISCC-HEADER** - Self describing header section of all ISCCs designating MainType, SubType, Version, Length
- **ISCC-BODY** - The actual payload of an ISCC, similarity preserving compact binary code, hash or timestamp
- **ISCC-UNIT** - An ISCC-HEADER + ISCC-BODY where the ISCC-BODY is calculated from algorithm
- **ISCC-CODE** - An ISCC-HEADER + ISCC-BODY where the ISCC-BODY is a sequence of multiple ISCC-UNIT BODYs
- **ISCC-ID** - A globally unique idendifier with an ISCC-BODY composed of a 52-bit microsecond timestamp and a
    12-bit server-id of the issuing ISCC-NOTARY server
- **SIMPRINT** - A headerless ISCC-UNIT that describes an individual section within larger content (granular
    feature)

### ISCC-UNITS

Here is a overview of the different ISCC-UNITs that can be part of an ISCC-CODE:

- The **Meta-Code** ISCC-UNIT encodes syntactic/lexical **metadata** similarity
- The **Semantic-Code** ISCC-UNITs encode semantic/conceptual **content** similarity
- The **Content-Codes** ISCC-UNITs encode perceptual/syntactic/structural **content** similarity
- The **Data-Code** ISCC-UNIT encode raw **data** similarity
- The **Instance-Code** ISCC-UNIT identifies **data** like a checksum or cryptographic hash

The Semantic-Code and Content-Code ISCC-UNITs have dedicated algorithms for the different content modalities
(Text, Image, Audio Video) An ISCC-CODE has at minimum a Data-Code and an Instance-Code in its composite
structure.

**NOTE**: While ISO 24138:2024 already denotes/reserves a MainType ID for "Semantic Codes", the standard does
not yet define algorithms for Semantic Codes but there are experimental implementations of Semantic-Codes and
SIMPRINTS for text (iscc-sct) and images (iscc-sci).

### ISCC Framework & Resources

The ISCC Framework consist of a collection of python libraries and applications published on GitHub:

- iscc/iscc-core - official python reference implementation of standardized low level codec and fingerprinting
    algorithms
- iscc/iscc-sdk - higher level content, detection, metadata extraction/embedding, content
    extraction/transformation, iscc code generation
- iscc/iscc-web - Rest api service for generating ISCC codes from media assets
- iscc/iscc-schema - ISCC metadata JSON schema and JSON-LD contexts
- iscc/iscc-crypto - cryptographic primitives for signing and timestamping iscc codes
- iscc/iscc-sct - library for generating semantic code text and granular semantic features
- iscc/iscc-sci - library for generating semantic code image
- iscc/iscc-ieps - Community driven specifications for the ISCC ecosystem

**Helpful Note**: These repositories are all available on deepwiki

### ISCC High-Level Goals

At a high level the ISCC Framework targets to enable developers to:

- Detect and analyse media content types
- Extract, embed and generate metadata for digital media content
- Extract, normalize and transform (OCR, transcribe ...) content (text, image, audio, video) from media files
- Create ISCC identifiers/fingerprints (ISCC-UNITs) for digital media content
- Index, store and search (by ISCC similarity) content and metadata
- Notarize content by signing and timestamping ISCCs and metadata (receiving ISCC-IDv1) and link to an ISCC
    Registry
- Register and search/discover metadata about ISCC-IDv1 identified content in ISCC registries using a common API

### Example data output from an ISCC generator

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

## About this project (iscc-vdb)

### Goals of `iscc-vdb`

- Provide a specialized ISCC indexing and search library for python as an embedded binary vector database based
    on usearch.
- Provide a command line tool for creating and managing indexes
- Provide an optional REST api web service built with the blacksheep for indexing and querying ISCCs and
    SIMPRINTS

### Challenges

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

## `ìscc-vdb` - Important Concepts & Ideas

### NPHD - A Custom Metric:

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

### Usearch Length Signalling:

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

### ISCC-IDs as 64-bit usearch vector keys:

*A globally unique, owned, and short identifier for digital assets.*

ISCC-IDs are timestamps that are minted and digitally signed by ISCC Notary Nodes. Each valid ISCC-ID is issued
as a verifiable credential, binding together the following critical components:

- **WHO**: The actor´s cryptographic public key and digital signature.
- **WHEN**: A timestamp (proof of existence), attested by the notary node.
- **WHERE**: A URL location where associated metadata/services can be discovered.
- **WHAT**: The digital content represented by an ISCC-CODE and a datahash.

The ISCC-IDv1 is a 64-bit identifier constructed from a timestamp and a server-id:

- First 52 bits: UTC time in microseconds since UNIX epoch (1970-01-01T00:00:00Z)
- Last 12 bits: ID of the timestamping server (0-4095)

With this structure:

- A single server can issue up to 1 million timestamps per second until the year 2112
- The system supports up to 4096 timestamp servers (IDs 0-4095)
- Timestamps are globally unique and support total ordering in both integer and base32hex forms
- The theoretical maximum throughput is ~4 billion unique timestamps per second

If the ID space becomes crowded, it can be extended by introducing additional REALMS via ISCC-HEADER SUBTYPEs.

**Minting Authority**:

ISCC-IDv1s are minted and digitally signed by authoritative ISCC Notary Servers in a federated system. A valid
ISCC-IDv1 is guanteed to be bound to an owner represented by a cryptographic public key. The rules by which
ISCC-IDv1 can be verified and resolved are defined by the `ISCC Notary Protocol` (IEP-0011 - TBD).

**Timestamp Requirements**:

Timestamp minting requires:

- A time source with at least microsecond precision
- Strictly monotonic (always increasing) integer timestamps
- Measures to prevent front-running of actual time

**Server ID Reservations**:

Server-ID `0` is reserved for sandbox/testing purposes. An ISCC-IDv1 with Server-ID 0:

- Makes no promises about uniqueness
- Is not authoritative
- Should not be used in production systems

**Technical Format**:

The ISCC-IDv1 has the following format:

- Scheme Prefix: `ISCC:`
- Base32-Encoded concatenation of:
    - 16-bit header:
        - MAINTYPE = "0110" (ISCC-ID)
        - SUBTYPE = "0000" (REALM, configurable via realm_id)
        - VERSION = "0001" (V1)
        - LENGTH = "0001" (64-bit)
    - 52-bit timestamp: Microseconds since 1970-01-01T00:00:00Z
    - 12-bit server-id: The Time Server ID (0-4095)

The 64-bit ISCC-BODY of the ISCC-IDv1 is the ideal candidate to be used as 64-bit integer keys for and ISCC
similarity index implementation based on usearc.

**ISCC-IDv1 Example**:

- Canonical: ISCC:MAIGHFEDREDPPIAB
- Human Readable Representation: ID-REALM_0-V1-641751832209682298-1
- Decoded μs timestamp: 1751832209682298
- Decoded Server-ID: 1
- Multiformat base32hex: vpg0m0433ii1oi1nnk00g

### Usearch Multi-key mode for SIMPRINT indexing:

We are planning to index simprints using usearch multi-key moded.

**Behavior Differences**:

The multi setting significantly affects how retrieval operations work:

**Single-key mode (multi=False)**:

- get() returns a single 2D array where each row corresponds to a key lib.cpp:931-938

**Multi-key mode (multi=True)**:

- get() returns a tuple where each element can contain multiple vectors for that key lib.cpp:913-930. The index
    tracks how many vectors are stored per key using index.count(key) lib.cpp:918

**Property Access**:

You can check if an index supports multi-value entries through the multi property index.py:1295-1301 , which is
exposed from the compiled index lib.cpp:1202 .

**Notes** The multi parameter must be set during index construction and cannot be changed afterward. It affects
the internal data structures and serialization format of the index. When using multi-key mode, retrieval
operations may return variable-length results depending on how many vectors were stored under each key.

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
