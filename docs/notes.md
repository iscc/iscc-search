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
their longer counterparts. The objective is to retrieve, for a query vector of any valid length, the most
similar vectors from the corpus, regardless of their length. Similarity must be assessed based on the Hamming
distance computed over the **common prefix length** shared between the query and potential neighbors. This
requirement renders standard ANNS methodologies, typically designed for fixed-length vectors and conventional
distance metrics inadequate. The inherent variability in vector length and the specific prefix-based similarity
definition necessitate a tailored approach.

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

**Usearch Length Signalling**:

As it seems usearch (as most vector indexes) only supports equi-dimensional vectors. Based on the assumption
that HNSW index construction in Usearch solely depends on the metric we could encode length information into the
vector itself and have metric implementation use it as a signal for the NPHD calculation. Given the maximum
length of 256-bit for an ISCC-UNIT body we could instantiate an 264-dimensional usearch index and dedicate the
first byte as signal of the actual code length (in number of bytes). Then for example given a 128-bit ISCC-UNIT
we would construct the bit-vector by seting the first byte to 16, add the 16 bytes from the ISCC-UNIT body and
pad the remaining 16 byte with zero bytes. Our custom NPHD metric could then infer the common_bytes of two
usearch vectors based on the signal in the first byte. While this is not very storage efficient it could be a
workable first implementation. If storage becomes a probelem we can still work on the much larger challenge of a
storage and access efficient datastructure for variable length bit-vectors.

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
