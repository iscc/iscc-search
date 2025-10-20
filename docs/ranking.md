# Search Result Ranking

## Problem Statement

IsccLookupIndex search results are currently sorted by total score (sum of matched bits across unit types).
However, **score ties are common** due to fixed ISCC bit lengths (64, 128, 192, 256). When multiple results have
identical scores, the current implementation returns them in non-deterministic order (dict iteration order).

### Example Scenario

Query with multiple units (META, CONTENT, DATA, INSTANCE) returns:

```python
[
    {
        "iscc_id": "ISCC:AAA...",
        "score": 256,
        "matches": {
            "INSTANCE_NONE_V0": 64,
            "DATA_NONE_V0": 64,
            "CONTENT_TEXT_V0": 64,
            "META_NONE_V0": 64
        }
    },
    {
        "iscc_id": "ISCC:BBB...",
        "score": 256,
        "matches": {
            "DATA_NONE_V0": 128,
            "CONTENT_TEXT_V0": 128
        }
    },
    {
        "iscc_id": "ISCC:CCC...",
        "score": 256,
        "matches": {
            "META_NONE_V0": 256
        }
    },
]
```

All three have identical total scores (256), but represent different similarity patterns:

- **AAA**: Matches on all unit types (INSTANCE present → strongest signal)
- **BBB**: Matches on DATA and CONTENT (strong structural and perceptual similarity)
- **CCC**: Matches only on META (weakest - just metadata similarity)

## Proposed Solution: Secondary Ranking by Unit Type Priority

Add a **secondary sort criterion** based on unit type semantic weight when scores are tied.

### Unit Type Hierarchy

ISCC unit types have a natural ordering from concrete to abstract similarity:

| Priority | Unit Type | Similarity Strength | Meaning                              |
| -------- | --------- | ------------------- | ------------------------------------ |
| 1        | INSTANCE  | Strongest           | Byte-identical binary data           |
| 2        | DATA      | Very Strong         | Same data structure/encoding         |
| 3        | CONTENT   | Strong              | Perceptual/feature similarity        |
| 4        | SEMANTIC  | Moderate            | Conceptual/semantic relatedness      |
| 5        | META      | Weakest             | Metadata similarity (title, creator) |

### Rationale

1. **INSTANCE** matches indicate exact duplicates (strongest evidence of similarity)
2. **DATA** matches indicate structural similarity (very strong evidence)
3. **CONTENT** matches indicate perceptual similarity (strong evidence for human perception)
4. **SEMANTIC** matches indicate conceptual relatedness (moderate evidence)
5. **META** matches only indicate organizational similarity (weakest evidence)

## Use Cases

Different applications benefit from different priority orderings:

### Concrete Priority (INSTANCE → META)

**Use cases:**

- Deduplication and integrity checking
- Finding exact or near-exact copies
- Content authenticity verification

**Example:**

```python
results = idx.search(query, limit=100, priority="concrete")

# Tied results sorted as:
# 1. Exact binary duplicate (INSTANCE)
# 2. Similar encoding/structure (DATA)
# 3. Just metadata matches (META)
```

### Abstract Priority (META → INSTANCE)

**Use cases:**

- Content discovery and recommendation
- Thematic collection building
- Finding conceptually related works

**Example:**

```python
results = idx.search(query, limit=100, priority="abstract")

# Tied results sorted as:
# 1. Topically related content (SEMANTIC)
# 2. Perceptually similar (CONTENT)
# 3. Exact duplicates (INSTANCE) - less interesting for discovery
```

## API Design

### Option 1: Search Parameter (Recommended)

```python
def search(self, iscc_items, limit=100, priority=None):
    # type: (IsccItemDict | list[IsccItemDict], int, str | None) -> list[IsccLookupResultDict]
    """
    Search via bidirectional prefix matching with aggregated scoring.

    :param iscc_items: Single IsccItemDict or list
    :param limit: Max matches per query (default: 100)
    :param priority: Secondary sort - "concrete", "abstract", or None (default: None)
    :return: One IsccLookupResultDict per query item
    """
```

**Pros:**

- Simple, clear API
- Flexible per-query
- Backward compatible (default=None preserves current behavior)

**Cons:**

- Must specify on every search call

### Option 2: Index-Level Configuration

```python
def __init__(self, path, realm_id=0, lmdb_options=None, default_priority=None):
    """
    :param default_priority: Default secondary sort - "concrete", "abstract", or None
    """
```

**Pros:**

- Set once, applies to all searches
- Cleaner search() calls

**Cons:**

- Less flexible (can't change per query)
- More complex if need to override

### Option 3: Hybrid

Allow both index-level default and per-query override:

```python
idx = IsccLookupIndex(path, default_priority="concrete")
results = idx.search(query, priority="abstract")  # Override default
```

## Implementation Details

### Sort Key Design

```python
# Primary sort: highest score first (unchanged)
# Secondary sort: unit type priority (new)

lookup_matches.sort(
    key=lambda x: (
        -x["score"],                           # Primary: descending score
        self._get_unit_type_priority(x, priority)  # Secondary: ascending priority
    )
)
```

### Priority Calculation

When a match contains multiple unit types, use **minimum priority** (strongest evidence wins):

```python
match = {
    "score": 256,
    "matches": {
        "META_NONE_V0": 64,      # Priority: 5 (weak)
        "INSTANCE_NONE_V0": 192  # Priority: 1 (strong)
    }
}

# Priority = min(5, 1) = 1 (INSTANCE priority dominates)
```

**Rationale:** The strongest match signal should determine ranking position. A result with both weak and strong
matches should rank with strong-match results.

**Alternatives considered:**

- `max(priorities)`: Weakest match dominates (counterintuitive)
- `avg(priorities)`: Dilutes strong signals (problematic)
- `weighted_avg(priorities, bits)`: Complex, unclear semantics

### Main Type Extraction

Use MainType from unit_type string (handles all subtypes automatically):

```python
"CONTENT_TEXT_V0" → "CONTENT" → Priority 3
"CONTENT_IMAGE_V0" → "CONTENT" → Priority 3
"SEMANTIC_TEXT_V0" → "SEMANTIC" → Priority 4
```

## Implementation Sketch

```python
def _get_unit_type_priority(self, match, priority):
    # type: (IsccLookupMatchDict, str) -> int
    """
    Calculate secondary sort priority based on unit type semantic weight.

    Returns best (minimum) priority among matched unit types.
    Concrete order: INSTANCE(1) → DATA(2) → CONTENT(3) → SEMANTIC(4) → META(5)
    Abstract order: reversed

    :param match: Match result with unit_type scores
    :param priority: "concrete" or "abstract"
    :return: Priority value (lower = higher priority)
    """
    MAIN_TYPE_PRIORITY = {
        "INSTANCE": 1,
        "DATA": 2,
        "CONTENT": 3,
        "SEMANTIC": 4,
        "META": 5,
    }

    priorities = []
    for unit_type in match["matches"].keys():
        main_type = unit_type.split("_")[0]
        base_priority = MAIN_TYPE_PRIORITY.get(main_type, 999)

        if priority == "abstract":
            priorities.append(6 - base_priority)  # Reverse: META(1) ... INSTANCE(5)
        else:  # "concrete"
            priorities.append(base_priority)

    return min(priorities) if priorities else 999
```

## Open Questions

1. **Default priority:** Should default be `None` (preserve current behavior) or `"concrete"` (most common use
    case)?

2. **Heapq integration:** If implementing heapq.nlargest() optimization, how does secondary sort interact with
    heap operations?

3. **Custom priority orders:** Should we allow custom priority mappings, or keep it simple with just
    "concrete"/"abstract"?

4. **Unknown unit types:** How should we handle future unit types not in MAIN_TYPE_PRIORITY dict? (Currently:
    priority=999, sorts last)

5. **Tertiary sort:** If both score and unit type priority tie, should we add a third criterion (e.g.,
    lexicographic ISCC-ID order for determinism)?

## Performance Optimization: Heap-Based Sorting

### Current Implementation

```python
# Build all matches into list
lookup_matches = []
for iscc_id, unit_type_scores in matches.items():
    total_score = sum(unit_type_scores.values())
    match_dict = IsccLookupMatchDict(...)
    lookup_matches.append(match_dict)

# Sort entire list, then slice to limit
lookup_matches.sort(key=lambda x: x["score"], reverse=True)
lookup_matches = lookup_matches[:limit]
```

**Complexity:** O(n log n) where n = total number of matches

### Proposed: Heap-Based Selection

```python
import heapq

# Build matches (same as before)
lookup_matches = [...]

# Use heap to select top k elements
lookup_matches = heapq.nlargest(
    limit,
    lookup_matches,
    key=lambda x: x["score"]  # Or tuple for secondary sort
)
# Result already in descending order
```

**Complexity:** O(n log k) where k = limit

### Performance Analysis

**When heapq helps:**

- Large result sets: n >> k (e.g., 10,000 matches, limit=100)
- Speedup: ~10x when n = 10,000 and k = 100
- Avoids sorting thousands of results that will be discarded

**When heapq doesn't help:**

- Small result sets: n ≈ k (e.g., 50 matches, limit=100)
- Overhead: Heap operations slightly slower than sort for small n
- Negligible difference in practice

**Recommendation:** Always use heapq.nlargest() since:

- Worst case (n ≈ k): minimal overhead
- Best case (n >> k): significant speedup
- Cleaner API (returns top k directly)

### Secondary Sort Integration

Heapq supports tuple-based sort keys for secondary ranking:

```python
# With secondary ranking
lookup_matches = heapq.nlargest(
    limit,
    lookup_matches,
    key=lambda x: (
        x["score"],                            # Primary (heapq auto-reverses for nlargest)
        -self._get_unit_type_priority(x, priority)  # Secondary (negate for correct order)
    )
)
```

**Note:** heapq.nlargest() automatically handles descending order for the primary key, but secondary keys need
manual negation to achieve ascending priority order.

### Complexity Comparison

| Scenario         | Current O(n log n) | Heapq O(n log k) | Speedup |
| ---------------- | ------------------ | ---------------- | ------- |
| n=100, k=10      | ~664 ops           | ~230 ops         | 2.9x    |
| n=1,000, k=10    | ~9,966 ops         | ~2,300 ops       | 4.3x    |
| n=10,000, k=100  | ~132,877 ops       | ~69,657 ops      | 1.9x    |
| n=100,000, k=100 | ~1,660,964 ops     | ~766,441 ops     | 2.2x    |

## Performance Impact of Secondary Ranking

**Expected:** Negligible

- Secondary sort key adds minimal computation (string split, dict lookup)
- Sort complexity unchanged: O(n log n) or O(n log k) with heapq
- No additional LMDB operations or memory overhead
- Overhead per comparison: ~1-2 microseconds (string split + dict lookup)

## Testing Requirements

1. **Score ties with different unit types:** Verify priority ordering works
2. **Mixed unit type matches:** Verify min() logic (strongest signal wins)
3. **Concrete vs abstract:** Verify ordering reverses correctly
4. **None priority:** Verify backward compatibility (current behavior unchanged)
5. **Unknown unit types:** Verify graceful handling (sort last)
6. **Edge cases:** Empty matches, single result, all same priority

## Related Improvements

This proposal synergizes with:

1. **Heap-based sorting optimization**: Both features affect the sorting step and can be implemented together
2. **Pagination support**: Stable, deterministic sort enables consistent paging across requests
3. **Score explanation API**: Could expose unit type priorities in match metadata for transparency
4. **Relevance tuning**: Secondary ranking provides foundation for weighted scoring experiments

## References

- ISCC specification: https://iscc.codes/
- ISCC unit types: https://core.iscc.codes/units/
- Original discussion: [link to this conversation]

## Status

**Proposed** - Awaiting discussion and decision on:

- API design (search parameter vs index config vs hybrid)
- Default priority value (None vs "concrete")
- Integration with heapq optimization
