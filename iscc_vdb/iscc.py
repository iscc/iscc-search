"""
Scalable multi-index ANNS search for ISCC-CODEs.

The IsccIndex class is a wrapper around multiple UnitIndex instances.
The IsccIndex accepts ISCC-IDs as keys and ISCC-CODEs as "vectors". ISCC-CODEs are decomposed
into ISCC-UNITs and indexed in separate UnitIndexes. UnitIndexes are created lazy on demand.

Basic usage:

```python
idx = CodeIndex()
idx.add(iscc_ids, iscc_codes)
idx.get(iscc_id)
idx.search(iscc_code)
```
"""

# import os
# from iscc_vdb.unit import UnitIndex
#
#
# class IsccIndex:
#     def __init__(self, path=None, view=False):
#         # type: (os.PathLike, bool) -> None
#         """
#         :param path: Where to store the index
#         :param view: If True, memory-map the index instead of loading
#         """
#         self.indexes = {}
#         self.path = path
#         if path is not None and os.path.exists(path):
#             if view:
#                 self.view(path)
#             else:
#                 self.load(path)
