# ISCC Simprint Index for Granular Content Matching

A Simprint is a typed similarity-preserving hash of a chunk of content situated in a larger content asset.

Simprint Datamodel:
- iscc_id (str): ISCC persistent identifier of the content asset to which the chunk belongs
- type (str): Type of the simprint (example: 'CONTENT_TEXT_V0')
- simprint (str|bytes): base64 encoded similarity hash bit-vector (up to 256 bits)
- offset (int): byte offset of the chunk in the content asset
- size (int): size of the chunk in bytes

Chunk:
    """A fully qualified chunk of content within a larger content asset."""
    iscc_id: str - Persistent Content Identifier
    offset: int - Byte offset of the chunk within the content asset
    size: int - Size of the chunk in bytes
    content: bytes - Content of the chunk

Indexes:

iscc-id:offset:size -> content
simprint -> list[iscc-id:offset:size]

## SimprintIndex

An LMDB-based typed inverted index that maps individual simprints to Chunks.


## Input Data

Object structure for a sin
- iscc_id
-
