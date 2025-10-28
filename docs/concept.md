# ISCC-SEARCH Concept

## Introduction

The International Standard Content Code (ISCC) is an open source content identification system. ISCCs come in
multiple different flavors but share a common format and structure. Every ISCC is self-describing and starts
with variable-length ISCC-HEADER (minimum 2 bytes) that identifies the MainType, SubType, Version, and Length of
an ISCC. The ISCC-HEADER is followed by an ISCC-BODY which is the actual payload that identifies a given digital
asset. We are distinguishing three broad categories of ISCCs: ISCC-UNITs, ISCC-CODEs, and ISCC-IDs:

ISCC-CODEs are multi-component fingerprints derived from digital content. The individual components of an
ISCC-CODE are called ISCC-UNITs. Each ISCC-UNIT uses a different algorithm such that in combination an ISCC-CODE
identifies a digital asset using a multi-faceted approach. Some ISCC-UNIT algorithms are media-type specific
such as text, image, audio or video others can work with any bitstream.

All algorithms, except for one, produce ISCC-UNITs that are similarity preserving binary codes (bit-vectors)
that can match similar content based on hamming distance. The one exception is the Instance-Code, which is a
cryptographic hash (blake3).

ISCC-CODEs and UNITs are not to be confused with persistent identifiers (PIDs). The sole purpose of these ISCCs
is to serve as standardized and reproducible and machine readable descriptors or fingerprints for digital
assets.

This is where the ISCC-ID comes into play. ISCC-IDs are PIDs that record who declared what content at what time
and where to find related metadata and services. ISCC-IDs issued by a distributed network of ISCC-HUBs and can
be decoded to a microsecond timestamp and an identifier of the issuing HUB.

But how are ISCC-IDs different from other PIDs? The difference is that traditional PIDs are unidirectional. They
resolve PIDs to metadata, content, and services. The ISCC-ID is bidirectional such that you can discover PID(s)
for some digital content by means of generating an ISCC-CODE and searching for ISCC-IDs issued for the same
similar content.

ISCC-SEARCH implements custom indexing techniques that are tailored to the structure of the ISCC and enable web
scale content based reverse search.

### The NPHD Distance Metric

ISCC-SEARCH uses **Normalized Prefix Hamming Distance (NPHD)** as its core similarity metric. Unlike standard
Hamming distance, NPHD handles variable-length codes by:

1. **Prefix alignment**: Compares only the common prefix length of two codes
2. **Length normalization**: Divides bit differences by common prefix length
3. **Metric properties**: Satisfies all metric axioms (non-negativity, identity, symmetry, triangle inequality)

This enables meaningful similarity comparison between:

- Short 64-bit codes from ISCC-CODEs (embedded in composite codes)
- Extended 256-bit ISCC-UNITs (standalone, high-precision)
- Mixed-length codes in the same search operation

Standard Hamming distance doesn't work here because it treats all bit differences equally regardless of vector
length, producing incorrect similarity scores when comparing codes of different lengths.

## The ISCC-SEARCH Solution

ISCC-SEARCH addresses the challenge of reverse content discovery at web scale. Traditional approaches to
similarity search struggle with:

- Variable-length codes (64-256 bits) that must be matched efficiently
- Multiple unit types requiring specialized indexing strategies
- Prefix compatibility requirements for shorter/longer codes
- Need for both exact (INSTANCE) and similarity (other units) matching

ISCC-SEARCH implements custom indexing techniques optimized for ISCC structure, providing:

- Unified interface across multiple storage backends
- Bidirectional prefix matching for variable-length codes
- Parallel search across unit-specific indexes with aggregated ranking
- Sub-millisecond query performance at scale

## ISCC Indexing Requirements

An ISCC index provides a unified interface for searching and matching ISCCs. This project provides multiple
index types, each making different tradeoffs in the solution space.

Generally an ISCC index has to manage multiple internal indexes because only ISCC-UNITs of the same type can be
searched and matched against each other. The input for a search can be an ISCC-CODE or a set of extended
ISCC-UNITs that belong to a given ISCC-CODE. The engine will than dispatch multiple searches against the
internal UNIT-specific indexes in parallel. The results are than aggregated and ranked before being returned to
the client.

Another special requirement for an ISCC index is the capability to match ISCC-UNITs of variable lengths against
each other. An ISCC-UNIT may be extracted from an ISCC-CODE with a length of only 64-bit while an individual
ISCC-UNIT may come with a 256-bit ISCC-BODY. The ISCC system guarantees that longer versions of ISCC-UNITs of
the same type are extensions of their shorter counterparts such that their common prefix is compatible. This
means that, while the statistical probability of unintended collisions is higher with shorter codes, we can
still match and compare short ISCC-UNITs against long ISCC-UNITS.

## Index Types

### LMDB Index

The LMDB Index is an embeddable, memory efficient, and durable index that uses local storage and supports
incremental updates. It matches ISCC-UNITs based on common prefixes supporting variable length UNIT matches. It
can only match near duplicates where two different but similar assets happen to produce identical ISCC-UNITs or
unit-prefixes.

### Usearch Index

The Usearch Index an embeddable index that provides fast and scalable similarity search for ISCCs. It makes use
of the Hierarchical Navigable Small World (HNSW) algorithm for efficiently finding approximate nearest neighbors
by building a hierarchical, multi-layered graph index. As such it can also match and rank ISCCs with
non-identical but similar codes. By default Usearch only supports indexes with fixed-length vectors. To overcome
that restriction we store ISCC-UNITs with a one byte length prefix and fixed-length padding. To support matching
those variable-length ISCC-UNITs we implemented as specialized distance metric called "Normalized Prefix Hamming
Distance (NPHD)" that accounts for the custom storage format. In read-only (view) mode the Usearch index is can
be efficiently memory mapped and support indexes that are larger than RAM. Usearch support incremental index
building but only if the entire index is loaded into memory.

### Postgres Index

The Postgres index uses the PGvector extenstion to support efficient similarity search over ISCC-UNITs in
hamming space using its HNSW implementation. ISCC-UNITs are indexed in separate tables. The Postgres index
stores all ISCC-UNITs using their short 64-bit representation does not support variable length indexing.
