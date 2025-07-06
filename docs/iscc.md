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

# The ISCC Framework

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

## At a high level the ISCC Framework targets to enable developers to:

- Detect and analyse media content types
- Extract, embed and generate metadata for digital media content
- Extract, normalize and transform (OCR, transcribe ...) content (text, image, audio, video) from media files
- Create ISCC identifiers/fingerprints (ISCC-UNITs) for digital media content
- Index, store and search (by ISCC similarity) content and metadata
- Notarize content by signing and timestamping ISCCs and metadata (receiving ISCC-IDv1) and link to an ISCC
    Registry
- Register and search/discover metadata about ISCC-IDv1 identified content in ISCC registries using a common API
