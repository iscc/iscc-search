---
icon: lucide/rocket
description: Step-by-step tutorial to install iscc-search, create an index, add ISCC codes, and run your first similarity search.
---

# Getting started

By the end of this tutorial you will have a working iscc-search installation, an index with ISCC codes, and
search results showing similar content.

## Prerequisites

- Python 3.10 or later
- `uv` or `pip` for package installation

## Install iscc-search

=== "uv"

    ```bash
    uv add iscc-search
    ```

=== "pip"

    ```bash
    pip install iscc-search
    ```

Verify the installation:

```bash
iscc-search version
```

## What is ISCC?

ISCC (International Standard Content Code, ISO 24138) is a content fingerprinting system for digital media.
It generates short codes from content - text, images, audio, video - that preserve similarity. Two
documents with overlapping content produce ISCC codes that are close in Hamming distance. iscc-search
exploits this property to find similar content across large collections.

For a deeper explanation, see the [ISCC primer](../explanation/iscc-primer.md).

## Create an index

An index stores ISCC codes and enables similarity search. Start with the `memory://` backend - it keeps
everything in RAM and requires no setup.

=== "Python"

    ```python
    import os

    os.environ["ISCC_SEARCH_INDEX_URI"] = "memory://"

    from iscc_search.options import get_index
    from iscc_search.schema import IsccIndex

    index = get_index()
    index.create_index(IsccIndex(name="myindex"))
    ```

=== "CLI"

    ```bash
    iscc-search index add myindex --local
    iscc-search index use myindex
    ```

=== "REST API"

    Start the server first:

    ```bash
    ISCC_SEARCH_INDEX_URI=memory:// iscc-search serve --dev
    ```

    Then create an index:

    ```bash
    curl -X POST http://localhost:8000/indexes \
      -H "Content-Type: application/json" \
      -d '{"name": "myindex"}'
    ```

## Add ISCC codes

Each asset you add contains an ISCC-CODE - a composite fingerprint that encodes multiple similarity
dimensions (content, data, instance). The index decomposes the code into individual units and indexes each
one for search.

=== "Python"

    ```python
    from iscc_search.schema import IsccEntry

    assets = [
        IsccEntry(
            iscc_code="ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY"
        ),
        IsccEntry(
            iscc_code="ISCC:KEC6CAS5WCRSL4AE"
        ),
    ]
    results = index.add_assets("myindex", assets)

    for r in results:
        print(f"{r.iscc_id}  status={r.status}")
    ```

=== "CLI"

    Create a JSON file `asset.json`:

    ```json
    {
      "iscc_code": "ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY"
    }
    ```

    Add it to the active index:

    ```bash
    iscc-search add asset.json
    ```

=== "REST API"

    ```bash
    curl -X POST http://localhost:8000/indexes/myindex/assets \
      -H "Content-Type: application/json" \
      -d '[{"iscc_code": "ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY"}]'
    ```

Each asset gets an auto-generated ISCC-ID (a unique identifier) if you do not provide one. The `status`
field in the result tells you whether the asset was `created` or `updated`.

## Search for similar content

Pass an ISCC-CODE as a query. The engine compares it against all indexed codes and returns ranked matches.

=== "Python"

    ```python
    from iscc_search.schema import IsccQuery

    query = IsccQuery(
        iscc_code="ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY"
    )
    results = index.search_assets("myindex", query, limit=10)

    for match in results.global_matches:
        print(f"{match.iscc_id}  score={match.score}")

    index.close()
    ```

=== "CLI"

    ```bash
    iscc-search search "ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY" --limit 10
    ```

=== "REST API"

    ```bash
    curl -X POST http://localhost:8000/indexes/myindex/search \
      -H "Content-Type: application/json" \
      -d '{"iscc_code": "ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY"}'
    ```

The `score` field ranges from 0.0 to 1.0. A score of 1.0 means the codes are identical. Scores above 0.75
(the default threshold) indicate strong similarity.

!!! tip
    You can also search using a GET request with a query parameter:
    `GET /indexes/myindex/search?iscc_code=ISCC:KACYPXW445FTYNJ3CYSXHAFJMA2HUWULUNRFE3BLHRSCXYH2M5AEGQY`

## Try a persistent backend

The `memory://` backend loses data when the process exits. For persistent storage, use `lmdb://` which
stores indexes on disk using LMDB (Lightning Memory-Mapped Database).

=== "Python"

    ```python
    import os

    os.environ["ISCC_SEARCH_INDEX_URI"] = "lmdb:///tmp/iscc-data"

    from iscc_search.options import get_index
    from iscc_search.schema import IsccIndex

    index = get_index()
    index.create_index(IsccIndex(name="persistent"))

    # Add assets and search as before...
    # Data survives restarts.

    index.close()
    ```

=== "CLI"

    ```bash
    iscc-search index add persistent --local --path /tmp/iscc-data
    iscc-search index use persistent
    ```

!!! note
    For production workloads with large collections, use the `usearch://` backend. It adds HNSW
    (Hierarchical Navigable Small World) graph indexing for fast approximate nearest neighbor search.
    See the [index backends guide](../howto/index-backends.md).

## Next steps

- [Index backends](../howto/index-backends.md) - configure memory, LMDB, and usearch backends
- [REST API](../howto/rest-api.md) - run the API server and use all endpoints
- [CLI reference](../howto/cli.md) - full command-line usage
- [ISCC primer](../explanation/iscc-primer.md) - how ISCC content codes work
