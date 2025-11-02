# LMDB Conversation

# DeepWiki Q&A with Code Context for Repository: jnwatson/py-lmdb

## Q1

lmdb allows for storing duplicate keys with different values. How does that compare to storing as single key with a
serialized list of values?

### Answer

You're asking about LMDB's `MDB_DUPSORT` feature, which allows storing multiple values for the same key, versus the
alternative of storing a single key with a serialized list of values. py-lmdb:1094-1097

## Key Differences

**Storage and Performance**

With `MDB_DUPSORT`, duplicate values are stored as a sorted sub-database where each value becomes a key in the
sub-database. py-lmdb:7599-7607 This means:

- Values are stored separately and sorted
- Individual values can be accessed, added, or removed without deserializing the entire list
- LMDB handles the B-tree structure for you

With a serialized list, you must:

- Deserialize the entire list to access any value
- Serialize and write the entire list for any modification
- Manage your own serialization format

**API Operations**

`MDB_DUPSORT` provides specialized cursor operations for working with duplicates: py-lmdb:1829-1844

- `iternext_dup()` - iterate through values for current key
- `iternext_nodup()` - iterate through keys, skipping duplicates py-lmdb:1846-1863
- `first_dup()`, `last_dup()` - navigate to first/last value py-lmdb:1943-1954 py-lmdb:1970-1981
- `set_key_dup()` - seek to specific key/value pair py-lmdb:2081-2093

**Write Behavior**

When using `MDB_DUPSORT`, the `dupdata` parameter controls whether duplicate values are allowed: py-lmdb:1545-1548

- `dupdata=True` (default) - adds the value if it doesn't exist
- `dupdata=False` - returns `False` if key already has that value

The implementation converts single values to sub-databases automatically when a second value is added. py-lmdb:
6811-6848

**Memory Efficiency**

For fixed-size values, `MDB_DUPFIXED` can be combined with `MDB_DUPSORT` for additional optimizations, enabling bulk
operations like `MDB_GET_MULTIPLE` and `MDB_NEXT_MULTIPLE`. py-lmdb:1102-1108

## Notes

The choice depends on your access patterns. Use `MDB_DUPSORT` when you need to:

- Access individual values without reading all of them
- Add/remove values incrementally
- Iterate through values efficiently
- Leverage LMDB's sorted storage

Use serialized lists when:

- You always need all values together
- The list is small
- You want simpler code without cursor operations

The py-lmdb changelog shows that duplicate support was significantly enhanced in v0.79 with methods like `next_dup()`,
`set_key_dup()`, etc. py-lmdb:461-468

Wiki pages you might want to explore:

- [Architecture (jnwatson/py-lmdb)](/wiki/jnwatson/py-lmdb#4)
- [LMDB Integration (jnwatson/py-lmdb)](/wiki/jnwatson/py-lmdb#4.2)

### Citations

**File:** lib/lmdb.h (L1094-1097)

```text
	 *	<li>#MDB_DUPSORT
	 *		Duplicate keys may be used in the database. (Or, from another perspective,
	 *		keys may have multiple data items, stored in sorted order.) By default
	 *		keys must be unique and may have only a single data item.
```

**File:** lib/lmdb.h (L1102-1108)

```text
	 *	<li>#MDB_DUPFIXED
	 *		This flag may only be used in combination with #MDB_DUPSORT. This option
	 *		tells the library that the data items for this database are all the same
	 *		size, which allows further optimizations in storage and retrieval. When
	 *		all data items are the same size, the #MDB_GET_MULTIPLE, #MDB_NEXT_MULTIPLE
	 *		and #MDB_PREV_MULTIPLE cursor operations may be used to retrieve multiple
	 *		items at once.
```

**File:** lib/mdb.c (L6811-6848)

```c
			/* Was a single item before, must convert now */
			if (!F_ISSET(leaf->mn_flags, F_DUPDATA)) {
				MDB_cmp_func *dcmp;
				/* Just overwrite the current item */
				if (flags == MDB_CURRENT)
					goto current;
				dcmp = mc->mc_dbx->md_dcmp;
#if UINT_MAX < SIZE_MAX
				if (dcmp == mdb_cmp_int && olddata.mv_size == sizeof(size_t))
					dcmp = mdb_cmp_clong;
#endif
				/* does data match? */
				if (!dcmp(data, &olddata)) {
					if (flags & (MDB_NODUPDATA|MDB_APPENDDUP))
						return MDB_KEYEXIST;
					/* overwrite it */
					goto current;
				}

				/* Back up original data item */
				dkey.mv_size = olddata.mv_size;
				dkey.mv_data = memcpy(fp+1, olddata.mv_data, olddata.mv_size);

				/* Make sub-page header for the dup items, with dummy body */
				MP_FLAGS(fp) = P_LEAF|P_DIRTY|P_SUBP;
				MP_LOWER(fp) = (PAGEHDRSZ-PAGEBASE);
				xdata.mv_size = PAGEHDRSZ + dkey.mv_size + data->mv_size;
				if (mc->mc_db->md_flags & MDB_DUPFIXED) {
					MP_FLAGS(fp) |= P_LEAF2;
					fp->mp_pad = data->mv_size;
					xdata.mv_size += 2 * data->mv_size;	/* leave space for 2 more */
				} else {
					xdata.mv_size += 2 * (sizeof(indx_t) + NODESIZE) +
						(dkey.mv_size & 1) + (data->mv_size & 1);
				}
				MP_UPPER(fp) = xdata.mv_size - PAGEBASE;
				olddata.mv_size = xdata.mv_size; /* pretend olddata is fp */
			} else if (leaf->mn_flags & F_SUBDATA) {
```

**File:** lib/mdb.c (L7599-7607)

```c
/** Initial setup of a sorted-dups cursor.
 * Sorted duplicates are implemented as a sub-database for the given key.
 * The duplicate data items are actually keys of the sub-database.
 * Operations on the duplicate data items are performed using a sub-cursor
 * initialized when the sub-database is first accessed. This function does
 * the preliminary setup of the sub-cursor, filling in the fields that
 * depend only on the parent DB.
 * @param[in] mc The main cursor whose sorted-dups cursor is to be initialized.
 */
```

**File:** lmdb/cffi.py (L1545-1548)

```python
            `dupdata`:
If
``False`` and database
was
opened
with `dupsort=True`, will return
``False`` if the
key
already
has
that
value.In
other
words, this
only
affects
the
return value.
```

**File:** lmdb/cffi.py (L1829-1844)

```python
    def iternext_dup(self, keys=False, values=True):
    """Return a forward iterator that yields the current value
    ("duplicate") of the current key before calling :py:meth:`next_dup`,
    repeating until the last value of the current key is reached.

    Only meaningful for databases opened with `dupsort=True`.

    .. code-block:: python

        if not cursor.set_key("foo"):
            print("No values found for 'foo'")
        else:
            for idx, data in enumerate(cursor.iternext_dup()):
                print("%d'th value for 'foo': %s" % (idx, data))
    """
    return self._iter(_lib.MDB_NEXT_DUP, keys, values)
```

**File:** lmdb/cffi.py (L1846-1863)

```python
    def iternext_nodup(self, keys=True, values=False):
    """Return a forward iterator that yields the current value
    ("duplicate") of the current key before calling :py:meth:`next_nodup`,
    repeating until the end of the database is reached.

    Only meaningful for databases opened with `dupsort=True`.

    If the cursor is not yet positioned, it is moved to the first key in
    the database, otherwise iteration proceeds from the current position.

    .. code-block:: python

        for key in cursor.iternext_nodup():
            print("Key '%s' has %d values" % (key, cursor.count()))
    """
    if not self._valid:
        self.first()
    return self._iter(_lib.MDB_NEXT_NODUP, keys, values)
```

**File:** lmdb/cffi.py (L1943-1954)

```python
    def first_dup(self):
    """Move to the first value ("duplicate") for the current key, returning
    ``True`` on success or ``False`` if the database is empty.

    Only meaningful for databases opened with `dupsort=True`.

    Equivalent to `mdb_cursor_get()
    <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
    with `MDB_FIRST_DUP
    <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
    """
    return self._cursor_get(_lib.MDB_FIRST_DUP)
```

**File:** lmdb/cffi.py (L1970-1981)

```python
    def last_dup(self):
    """Move to the last value ("duplicate") for the current key, returning
    ``True`` on success or ``False`` if the database is empty.

    Only meaningful for databases opened with `dupsort=True`.

    Equivalent to `mdb_cursor_get()
    <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
    with `MDB_LAST_DUP
    <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
    """
    return self._cursor_get(_lib.MDB_LAST_DUP)
```

**File:** lmdb/cffi.py (L2081-2093)

```python
    def set_key_dup(self, key, value):
    """Seek exactly to `(key, value)`, returning ``True`` on success or
    ``False`` if the exact key and value was not found. It is an error
    to :py:meth:`set_key` the empty bytestring.

    Only meaningful for databases opened with `dupsort=True`.

    Equivalent to `mdb_cursor_get()
    <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
    with `MDB_GET_BOTH
    <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
    """
    return self._cursor_get_kv(_lib.MDB_GET_BOTH, key, value)
```

**File:** ChangeLog (L461-468)

```text
* Cursor.replace() and Cursor.pop() updated for MDB_DUPSORT databases. For
  pop(), the first data item is popped and returned. For replace(), the first
  data item is returned, and all duplicates for the key are replaced.

* Implement remaining Cursor methods necessary for working with MDB_DUPSORT
  databases: next_dup(), next_nodup(), prev_dup(), prev_nodup(), first_dup(),
  last_dup(), set_key_dup(), set_range_dup(), iternext_dup(),
  iternext_nodup(), iterprev_dup(), iterprev_nodup().
```

## Q2

are keys stored multiple times with dupsort or only once? which method is more storage efficient with fixed size keys
and values?

### Answer

You're asking whether keys are stored multiple times when using `MDB_DUPSORT`, and which approach is more
storage-efficient for fixed-size keys and values.

## Key Storage with MDB_DUPSORT

With `MDB_DUPSORT`, **keys are stored only once**. py-lmdb:7599-7606 The duplicate values are stored as a sub-database
where the key appears once in the main database, and the values become keys in a sub-database. py-lmdb:7056-7061

When you add a second value for an existing key, LMDB converts the single value into a sub-page or sub-database
structure. py-lmdb:6811-6847 The original key remains stored once in the parent database, pointing to the sub-database
that contains all the duplicate values.

## Storage Efficiency Comparison

For **fixed-size keys and values**, `MDB_DUPSORT` with `MDB_DUPFIXED` is more storage-efficient than serialized
lists:<cite />

**MDB_DUPSORT + MDB_DUPFIXED advantages:**

- Key stored once, not repeated for each value<cite />
- Fixed-size values stored contiguously without per-item overhead py-lmdb:1026-1030
- Uses `P_LEAF2` pages which omit node headers and store keys contiguously py-lmdb:800-840
- Enables bulk operations (`MDB_GET_MULTIPLE`, `MDB_NEXT_MULTIPLE`) that retrieve multiple values in a single call
  py-lmdb:1102-1108

**Serialized list approach:**

- Key stored once<cite />
- Entire serialized list stored as a single value<cite />
- Overhead from serialization format (length prefixes, delimiters, etc.)<cite />
- Must read/write entire list for any modification<cite />

The `MDB_DUPFIXED` optimization is specifically designed for this use case - when all duplicate values are the same
size, LMDB packs them efficiently without per-value metadata. py-lmdb:318-320

## Notes

The storage efficiency advantage of `MDB_DUPSORT` + `MDB_DUPFIXED` increases with the number of duplicate values per
key, since the key overhead is amortized across all values. For very small numbers of duplicates (1-2 values), the
difference may be negligible, but for larger sets, the space savings become significant.

Wiki pages you might want to explore:

- [Architecture (jnwatson/py-lmdb)](/wiki/jnwatson/py-lmdb#4)
- [LMDB Integration (jnwatson/py-lmdb)](/wiki/jnwatson/py-lmdb#4.2)

### Citations

**File:** lib/mdb.c (L800-840)

```c
	  + (((MDB_PIDLOCK) != 0) << 16)))
/** @} */

/** Common header for all page types. The page type depends on #mp_flags.
 *
 * #P_BRANCH and #P_LEAF pages have unsorted '#MDB_node's at the end, with
 * sorted #mp_ptrs[] entries referring to them. Exception: #P_LEAF2 pages
 * omit mp_ptrs and pack sorted #MDB_DUPFIXED values after the page header.
 *
 * #P_OVERFLOW records occupy one or more contiguous pages where only the
 * first has a page header. They hold the real data of #F_BIGDATA nodes.
 *
 * #P_SUBP sub-pages are small leaf "pages" with duplicate data.
 * A node with flag #F_DUPDATA but not #F_SUBDATA contains a sub-page.
 * (Duplicate data can also go in sub-databases, which use normal pages.)
 *
 * #P_META pages contain #MDB_meta, the start point of an LMDB snapshot.
 *
 * Each non-metapage up to #MDB_meta.%mm_last_pg is reachable exactly once
 * in the snapshot: Either used by a database or listed in a freeDB record.
 */
typedef struct MDB_page {
#define	mp_pgno	mp_p.p_pgno
#define	mp_next	mp_p.p_next
	union {
		pgno_t		p_pgno;	/**< page number */
		struct MDB_page *p_next; /**< for in-memory list of freed pages */
	} mp_p;
	uint16_t	mp_pad;			/**< key size if this is a LEAF2 page */
/**	@defgroup mdb_page	Page Flags
 *	@ingroup internal
 *	Flags for the page headers.
 *	@{
 */
#define	P_BRANCH	 0x01		/**< branch page */
#define	P_LEAF		 0x02		/**< leaf page */
#define	P_OVERFLOW	 0x04		/**< overflow page */
#define	P_META		 0x08		/**< meta page */
#define	P_DIRTY		 0x10		/**< dirty page, also set for #P_SUBP pages */
#define	P_LEAF2		 0x20		/**< for #MDB_DUPFIXED records */
#define	P_SUBP		 0x40		/**< for #MDB_DUPSORT sub-pages */
```

**File:** lib/mdb.c (L1026-1030)

```c
	/** The address of a key in a LEAF2 page.
	 *	LEAF2 pages are used for #MDB_DUPFIXED sorted-duplicate sub-DBs.
	 *	There are no node headers, keys are stored contiguously.
	 */
#define LEAF2KEY(p, i, ks)	((char *)(p) + PAGEHDRSZ + ((i)*(ks)))
```

**File:** lib/mdb.c (L6811-6847)

```c
			/* Was a single item before, must convert now */
			if (!F_ISSET(leaf->mn_flags, F_DUPDATA)) {
				MDB_cmp_func *dcmp;
				/* Just overwrite the current item */
				if (flags == MDB_CURRENT)
					goto current;
				dcmp = mc->mc_dbx->md_dcmp;
#if UINT_MAX < SIZE_MAX
				if (dcmp == mdb_cmp_int && olddata.mv_size == sizeof(size_t))
					dcmp = mdb_cmp_clong;
#endif
				/* does data match? */
				if (!dcmp(data, &olddata)) {
					if (flags & (MDB_NODUPDATA|MDB_APPENDDUP))
						return MDB_KEYEXIST;
					/* overwrite it */
					goto current;
				}

				/* Back up original data item */
				dkey.mv_size = olddata.mv_size;
				dkey.mv_data = memcpy(fp+1, olddata.mv_data, olddata.mv_size);

				/* Make sub-page header for the dup items, with dummy body */
				MP_FLAGS(fp) = P_LEAF|P_DIRTY|P_SUBP;
				MP_LOWER(fp) = (PAGEHDRSZ-PAGEBASE);
				xdata.mv_size = PAGEHDRSZ + dkey.mv_size + data->mv_size;
				if (mc->mc_db->md_flags & MDB_DUPFIXED) {
					MP_FLAGS(fp) |= P_LEAF2;
					fp->mp_pad = data->mv_size;
					xdata.mv_size += 2 * data->mv_size;	/* leave space for 2 more */
				} else {
					xdata.mv_size += 2 * (sizeof(indx_t) + NODESIZE) +
						(dkey.mv_size & 1) + (data->mv_size & 1);
				}
				MP_UPPER(fp) = xdata.mv_size - PAGEBASE;
				olddata.mv_size = xdata.mv_size; /* pretend olddata is fp */
```

**File:** lib/mdb.c (L7056-7061)

```c
	if (rc == MDB_SUCCESS) {
		/* Now store the actual data in the child DB. Note that we're
		 * storing the user data in the keys field, so there are strict
		 * size limits on dupdata. The actual data fields of the child
		 * DB are all zero size.
		 */
```

**File:** lib/mdb.c (L7599-7606)

```c
/** Initial setup of a sorted-dups cursor.
 * Sorted duplicates are implemented as a sub-database for the given key.
 * The duplicate data items are actually keys of the sub-database.
 * Operations on the duplicate data items are performed using a sub-cursor
 * initialized when the sub-database is first accessed. This function does
 * the preliminary setup of the sub-cursor, filling in the fields that
 * depend only on the parent DB.
 * @param[in] mc The main cursor whose sorted-dups cursor is to be initialized.
```

**File:** lib/lmdb.h (L318-320)

```text
	/** with #MDB_DUPSORT, sorted dup items have fixed size */
#define MDB_DUPFIXED	0x10
	/** with #MDB_DUPSORT, dups are #MDB_INTEGERKEY-style integers */
```

**File:** lib/lmdb.h (L1102-1108)

```text
	 *	<li>#MDB_DUPFIXED
	 *		This flag may only be used in combination with #MDB_DUPSORT. This option
	 *		tells the library that the data items for this database are all the same
	 *		size, which allows further optimizations in storage and retrieval. When
	 *		all data items are the same size, the #MDB_GET_MULTIPLE, #MDB_NEXT_MULTIPLE
	 *		and #MDB_PREV_MULTIPLE cursor operations may be used to retrieve multiple
	 *		items at once.
```

## Q3

how does lmdb know keys and values are fixed sizeo I have to indicate that? give me a python example

### Answer

You're asking how LMDB knows that keys and values are fixed-size, and whether you need to indicate this when opening a
database in py-lmdb.

## Indicating Fixed-Size Keys and Values

Yes, you must explicitly indicate fixed-size keys and values when opening a database using flags: py-lmdb:1098-1108

**For fixed-size keys:**

- Use `MDB_INTEGERKEY` flag - tells LMDB that all keys are unsigned integers or `size_t` of the same size py-lmdb:
  1098-1101

**For fixed-size duplicate values:**

- Use `MDB_DUPFIXED` flag (requires `MDB_DUPSORT`) - tells LMDB that all duplicate values for each key are the same
  size py-lmdb:1102-1108
- Use `MDB_INTEGERDUP` flag - tells LMDB that duplicate values are integers (implies both `MDB_DUPSORT` and
  `MDB_DUPFIXED`) py-lmdb:1109-1111

## Python Example

Here's how to open a database with fixed-size keys and values in py-lmdb: py-lmdb:1102-1194

```python
import lmdb


# Example 1: Fixed-size integer keys with duplicate fixed-size values
env = lmdb.open('/tmp/mydb', max_dbs=5)

# Open database with integerkey=True and dupfixed=True
db = env.open_db(
    b'my_fixed_db',
    integerkey=True,  # Keys are fixed-size integers
    dupsort=True,  # Allow duplicate keys
    dupfixed=True  # All duplicate values are same size
)

# Example 2: Using integerdup for integer duplicate values
db2 = env.open_db(
    b'my_int_dup_db',
    dupsort=True,
    integerdup=True  # Duplicate values are integers (implies dupfixed)
)

# Example 3: Just fixed-size duplicate values (not integers)
db3 = env.open_db(
    b'my_dupfixed_db',
    dupsort=True,
    dupfixed=True  # All values same size, but not necessarily integers
)
```

When you set these flags, LMDB uses optimized storage: py-lmdb:1026-1030

- `P_LEAF2` pages store keys/values contiguously without node headers py-lmdb:839-840
- The `md_pad` field stores the fixed size py-lmdb:1041-1041
- Enables bulk operations like `MDB_GET_MULTIPLE` and `MDB_NEXT_MULTIPLE` py-lmdb:2125-2131

## Notes

The flags are set when opening the database and are **persistent** - once set, they cannot be changed for that
database. py-lmdb:1137-1142 If you try to open an existing database with different flags, they will be ignored. You can
check the current flags using `db.flags()`. py-lmdb:1014-1035

Wiki pages you might want to explore:

- [Architecture (jnwatson/py-lmdb)](/wiki/jnwatson/py-lmdb#4)
- [LMDB Integration (jnwatson/py-lmdb)](/wiki/jnwatson/py-lmdb#4.2)

### Citations

**File:** lib/lmdb.h (L1098-1108)

```text
	 *	<li>#MDB_INTEGERKEY
	 *		Keys are binary integers in native byte order, either unsigned int
	 *		or size_t, and will be sorted as such.
	 *		The keys must all be of the same size.
	 *	<li>#MDB_DUPFIXED
	 *		This flag may only be used in combination with #MDB_DUPSORT. This option
	 *		tells the library that the data items for this database are all the same
	 *		size, which allows further optimizations in storage and retrieval. When
	 *		all data items are the same size, the #MDB_GET_MULTIPLE, #MDB_NEXT_MULTIPLE
	 *		and #MDB_PREV_MULTIPLE cursor operations may be used to retrieve multiple
	 *		items at once.
```

**File:** lib/lmdb.h (L1109-1111)

```text
	 *	<li>#MDB_INTEGERDUP
	 *		This option specifies that duplicate data items are binary integers,
	 *		similar to #MDB_INTEGERKEY keys.
```

**File:** lmdb/cffi.py (L1102-1194)

```python
    def open_db(self, key=None, txn=None, reverse_key=False, dupsort=False,
                create=True, integerkey=False, integerdup=False,
                dupfixed=False):
    """
    Open a database, returning an instance of :py:class:`_Database`. Repeat
    :py:meth:`Environment.open_db` calls for the same name will return the
    same handle. As a special case, the main database is always open.

    Equivalent to `mdb_dbi_open()
    <http://lmdb.tech/doc/group__mdb.html#gac08cad5b096925642ca359a6d6f0562a>`_

    Named databases are implemented by *storing a special descriptor in the
    main database*. All databases in an environment *share the same file*.
    Because the descriptor is present in the main database, attempts to
    create a named database will fail if a key matching the database's name
    already exists. Furthermore *the key is visible to lookups and
    enumerations*. If your main database keyspace conflicts with the names
    you use for named databases, then move the contents of your main
    database to another named database.

    ::

        >>> env = lmdb.open('/tmp/test', max_dbs=2)
        >>> with env.begin(write=True) as txn:
        ...     txn.put('somename', 'somedata')

        >>> # Error: database cannot share name of existing key!
        >>> subdb = env.open_db('somename')

    A newly created database will not exist if the transaction that created
    it aborted, nor if another process deleted it. The handle resides in
    the shared environment, it is not owned by the current transaction or
    process. Only one thread should call this function; it is not
    mutex-protected in a read-only transaction.

    The `dupsort`, `integerkey`, `integerdup`, and `dupfixed` parameters are
    ignored if the database already exists.  The state of those settings are
    persistent and immutable per database.  See :py:meth:`_Database.flags`
    to view the state of those options for an opened database.  A consequence
    of the immutability of these flags is that the default non-named database
    will never have these flags set.

    Preexisting transactions, other than the current transaction and any
    parents, must not use the new handle, nor must their children.

        `key`:
            Bytestring database name. If ``None``, indicates the main
            database should be returned, otherwise indicates a named
            database should be created inside the main database.

            In other words, *a key representing the database will be
            visible in the main database, and the database name cannot
            conflict with any existing key.*

        `txn`:
            Transaction used to create the database if it does not exist.
            If unspecified, a temporarily write transaction is used. Do not
            call :py:meth:`open_db` from inside an existing transaction
            without supplying it here. Note the passed transaction must
            have `write=True`.

        `reverse_key`:
            If ``True``, keys are compared from right to left (e.g. DNS
            names).

        `dupsort`:
            Duplicate keys may be used in the database. (Or, from another
            perspective, keys may have multiple data items, stored in
            sorted order.) By default keys must be unique and may have only
            a single data item.

        `create`:
            If ``True``, create the database if it doesn't exist, otherwise
            raise an exception.

        `integerkey`:
            If ``True``, indicates keys in the database are C unsigned
            or ``size_t`` integers encoded in native byte order. Keys must
            all be either unsigned or ``size_t``, they cannot be mixed in a
            single database.

        `integerdup`:
            If ``True``, values in the
            database are C unsigned or ``size_t`` integers encoded in
            native byte order.  Implies `dupsort` and `dupfixed` are
            ``True``.

        `dupfixed`:
            If ``True``, values for each key
            in database are of fixed size, allowing each additional
            duplicate value for a key to be stored without a header
            indicating its size.  Implies `dupsort` is ``True``.
    """
```

**File:** lib/mdb.c (L839-840)

```c
#define	P_LEAF2		 0x20		/**< for #MDB_DUPFIXED records */
#define	P_SUBP		 0x40		/**< for #MDB_DUPSORT sub-pages */
```

**File:** lib/mdb.c (L1026-1030)

```c
	/** The address of a key in a LEAF2 page.
	 *	LEAF2 pages are used for #MDB_DUPFIXED sorted-duplicate sub-DBs.
	 *	There are no node headers, keys are stored contiguously.
	 */
#define LEAF2KEY(p, i, ks)	((char *)(p) + PAGEHDRSZ + ((i)*(ks)))
```

**File:** lib/mdb.c (L1041-1041)

```c
	uint32_t	md_pad;		/**< also ksize for LEAF2 pages */
```

**File:** lmdb/cpython.c (L1014-1035)

```c
static PyObject *
db_flags(DbObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *dct;
    unsigned int f;

    if (args) {
        Py_ssize_t size = PyTuple_GET_SIZE(args);
        if(size > 1) {
            return type_error("too many positional arguments.");
        }
    }

    dct = PyDict_New();
    f = self->flags;
    PyDict_SetItemString(dct, "reverse_key", py_bool(f & MDB_REVERSEKEY));
    PyDict_SetItemString(dct, "dupsort", py_bool(f & MDB_DUPSORT));
    PyDict_SetItemString(dct, "integerkey", py_bool(f & MDB_INTEGERKEY));
    PyDict_SetItemString(dct, "integerdup", py_bool(f & MDB_INTEGERDUP));
    PyDict_SetItemString(dct, "dupfixed", py_bool(f & MDB_DUPFIXED));
    return dct;
}
```

**File:** lmdb/cpython.c (L2125-2131)

```c
    if(arg.dupfixed_bytes) {
        get_op = MDB_GET_MULTIPLE;
        next_op = MDB_NEXT_MULTIPLE;
    } else {
        get_op = MDB_GET_CURRENT;
        next_op = MDB_NEXT_DUP;
    }
```

## Q4

what exactly do we gain by using integer keys and/or values?

### Answer

You're asking about the specific benefits of using `MDB_INTEGERKEY` and `MDB_INTEGERDUP` flags in LMDB, which tell LMDB
that keys and/or values are fixed-size integers.

## Performance Benefits

**Faster Comparisons**

When you use `MDB_INTEGERKEY` or `MDB_INTEGERDUP`, LMDB uses optimized integer comparison functions instead of
byte-by-byte memory comparison. py-lmdb:9841-9855

The comparison functions are selected based on the flags: py-lmdb:9846-9854

- `MDB_INTEGERKEY` → uses `mdb_cmp_cint` (integer comparison)
- `MDB_INTEGERDUP` → uses `mdb_cmp_int` or `mdb_cmp_cint` depending on whether `MDB_DUPFIXED` is also set

Integer comparisons (`mdb_cmp_long` and similar) directly compare numeric values rather than doing memcmp: py-lmdb:
5257-5263

This is significantly faster than byte-by-byte string comparison, especially for larger integers.

## Storage Benefits

**Compact Storage with LEAF2 Pages**

When using `MDB_INTEGERKEY` or `MDB_INTEGERDUP` with `MDB_DUPFIXED`, LMDB uses `P_LEAF2` pages that store keys/values
contiguously without node headers. py-lmdb:1026-1030

The `md_pad` field stores the fixed size of each integer, allowing LMDB to calculate offsets directly without storing
per-item metadata. py-lmdb:1040-1041

**Bulk Operations**

With fixed-size integers (`MDB_DUPFIXED` + `MDB_INTEGERDUP`), you can use bulk retrieval operations like
`MDB_GET_MULTIPLE` and `MDB_NEXT_MULTIPLE` to fetch multiple values in a single call. py-lmdb:1102-1108

This is exposed in py-lmdb through cursor operations that can retrieve batches of fixed-size values efficiently.
py-lmdb:2179-2227

## Notes

The integer flags are most beneficial when:

- You're storing numeric IDs or timestamps as keys
- You have many duplicate values per key (with `MDB_INTEGERDUP`)
- You need fast sorted iteration over numeric ranges

The flags require that all keys (or values) are the same size - either `unsigned int` or `size_t` in native byte order.
py-lmdb:1098-1101 You cannot mix different integer sizes in the same database.

Wiki pages you might want to explore:

- [Architecture (jnwatson/py-lmdb)](/wiki/jnwatson/py-lmdb#4)
- [LMDB Integration (jnwatson/py-lmdb)](/wiki/jnwatson/py-lmdb#4.2)

### Citations

**File:** lib/mdb.c (L1026-1030)

```c
	/** The address of a key in a LEAF2 page.
	 *	LEAF2 pages are used for #MDB_DUPFIXED sorted-duplicate sub-DBs.
	 *	There are no node headers, keys are stored contiguously.
	 */
#define LEAF2KEY(p, i, ks)	((char *)(p) + PAGEHDRSZ + ((i)*(ks)))
```

**File:** lib/mdb.c (L1040-1041)

```c
typedef struct MDB_db {
	uint32_t	md_pad;		/**< also ksize for LEAF2 pages */
```

**File:** lib/mdb.c (L5257-5263)

```c
/** Compare two items pointing at aligned size_t's */
static int
mdb_cmp_long(const MDB_val *a, const MDB_val *b)
{
	return (*(size_t *)a->mv_data < *(size_t *)b->mv_data) ? -1 :
		*(size_t *)a->mv_data > *(size_t *)b->mv_data;
}
```

**File:** lib/mdb.c (L9841-9855)

```c
static void
mdb_default_cmp(MDB_txn *txn, MDB_dbi dbi)
{
	uint16_t f = txn->mt_dbs[dbi].md_flags;

	txn->mt_dbxs[dbi].md_cmp =
		(f & MDB_REVERSEKEY) ? mdb_cmp_memnr :
		(f & MDB_INTEGERKEY) ? mdb_cmp_cint  : mdb_cmp_memn;

	txn->mt_dbxs[dbi].md_dcmp =
		!(f & MDB_DUPSORT) ? 0 :
		((f & MDB_INTEGERDUP)
		 ? ((f & MDB_DUPFIXED)   ? mdb_cmp_int   : mdb_cmp_cint)
		 : ((f & MDB_REVERSEDUP) ? mdb_cmp_memnr : mdb_cmp_memn));
}
```

**File:** lib/lmdb.h (L1098-1101)

```text
	 *	<li>#MDB_INTEGERKEY
	 *		Keys are binary integers in native byte order, either unsigned int
	 *		or size_t, and will be sorted as such.
	 *		The keys must all be of the same size.
```

**File:** lib/lmdb.h (L1102-1108)

```text
	 *	<li>#MDB_DUPFIXED
	 *		This flag may only be used in combination with #MDB_DUPSORT. This option
	 *		tells the library that the data items for this database are all the same
	 *		size, which allows further optimizations in storage and retrieval. When
	 *		all data items are the same size, the #MDB_GET_MULTIPLE, #MDB_NEXT_MULTIPLE
	 *		and #MDB_PREV_MULTIPLE cursor operations may be used to retrieve multiple
	 *		items at once.
```

**File:** lmdb/cpython.c (L2179-2227)

```c
                    /* dupfixed, MDB_GET_MULTIPLE returns batch, iterate values */
                    int items = (int) self->val.mv_size/val_size;
                    if (first) {
                        key_size = (size_t) self->key.mv_size;
                        item_size = key_size + val_size;
                        if (arg.keyfixed) { /* Init structured array buffer */
                            buffer = malloc(buffer_size * item_size);
                        }
                        first = false;
                    }

                    for(i=0; i<items; i++) {
                        char *val_data = (char *) self->val.mv_data + (i * val_size);
                        if (arg.keyfixed) {
                            /* Add to array buffer */
                            char *k, *v;
                            if (buffer_pos >= buffer_size) { // Grow buffer
                                buffer_size = buffer_size * 2;
                                buffer = realloc(buffer, buffer_size * item_size);
                            }
                            k = buffer + (buffer_pos * item_size);
                            v = k + key_size;
                            memcpy(k, (char *) self->key.mv_data, key_size);
                            memcpy(v, val_data, val_size);

                            buffer_pos++;
                        } else {
                            /* Add to list of tuples */
                            if(as_buffer) {
                                val = PyMemoryView_FromMemory(
                                    val_data, (size_t) arg.dupfixed_bytes, PyBUF_READ);
                            } else {
                                val = PyBytes_FromStringAndSize(
                                    val_data, (size_t) arg.dupfixed_bytes);
                            }
                            tup = PyTuple_New(2);
                            if (tup && key && val) {
                                Py_INCREF(key); // Hold key in loop
                                PyTuple_SET_ITEM(tup, 0, key);
                                PyTuple_SET_ITEM(tup, 1, val);
                                PyList_Append(pylist, tup);
                                Py_DECREF(tup);
                            } else {
                                Py_DECREF(val);
                                Py_DECREF(tup);
                            }
                        }
                    }
                    Py_DECREF(key); // Release key
```

## Q5

how many bits can those integers have

### Answer

You're asking about the bit size of integers that can be used with `MDB_INTEGERKEY` and `MDB_INTEGERDUP` flags in LMDB.

## Integer Size Options

LMDB supports two integer sizes for keys and values: py-lmdb:315-317

1. **`unsigned int`** - typically 32 bits (4 bytes)
2. **`size_t`** - platform-dependent, either 32 bits or 64 bits

The documentation specifies that keys must be "either unsigned int or size_t" in native byte order, and all keys (or
values) in a database must be the same size. py-lmdb:1098-1101

## Platform-Specific Behavior

The actual size used depends on your platform:<cite />

- **32-bit systems**: `size_t` is 4 bytes (32 bits)
- **64-bit systems**: `size_t` is 8 bytes (64 bits)

LMDB automatically selects the appropriate comparison function based on the actual size detected at runtime. py-lmdb:
1813-1817 When `size_t` equals 8 bytes, it uses `mdb_cmp_clong` for aligned comparisons or `mdb_cmp_long` for unaligned
ones. py-lmdb:5257-5263

For `unsigned int`, the comparison uses `mdb_cmp_int` which handles 32-bit integers. py-lmdb:5270-5275

## Notes

You cannot use arbitrary integer sizes - only `unsigned int` (32-bit) or `size_t` (32 or 64-bit depending on platform).
All integers in the same database must be the same size. If you need different integer sizes, you would need to use
separate databases or encode them as fixed-size byte strings without the `MDB_INTEGERKEY`/`MDB_INTEGERDUP` flags.

Wiki pages you might want to explore:

- [LMDB Integration (jnwatson/py-lmdb)](/wiki/jnwatson/py-lmdb#4.2)

### Citations

**File:** lib/lmdb.h (L315-317)

```text
	/** numeric keys in native byte order: either unsigned int or size_t.
	 *  The keys must all be of the same size. */
#define MDB_INTEGERKEY	0x08
```

**File:** lib/lmdb.h (L1098-1101)

```text
	 *	<li>#MDB_INTEGERKEY
	 *		Keys are binary integers in native byte order, either unsigned int
	 *		or size_t, and will be sorted as such.
	 *		The keys must all be of the same size.
```

**File:** lib/mdb.c (L1813-1817)

```c
#if UINT_MAX < SIZE_MAX
	if (dcmp == mdb_cmp_int && a->mv_size == sizeof(size_t))
		dcmp = mdb_cmp_clong;
#endif
	return dcmp(a, b);
```

**File:** lib/mdb.c (L5257-5263)

```c
/** Compare two items pointing at aligned size_t's */
static int
mdb_cmp_long(const MDB_val *a, const MDB_val *b)
{
	return (*(size_t *)a->mv_data < *(size_t *)b->mv_data) ? -1 :
		*(size_t *)a->mv_data > *(size_t *)b->mv_data;
}
```

**File:** lib/mdb.c (L5270-5275)

```c
static int
mdb_cmp_int(const MDB_val *a, const MDB_val *b)
{
	return (*(unsigned int *)a->mv_data < *(unsigned int *)b->mv_data) ? -1 :
		*(unsigned int *)a->mv_data > *(unsigned int *)b->mv_data;
}
```

## Q6

when storin multiple value for a given key using dupsort, how can i effieciently retrieve all values for that key

### Answer

You're asking about efficiently retrieving all duplicate values for a given key when using `MDB_DUPSORT` in LMDB.

## Efficient Retrieval Methods

There are several ways to retrieve all values for a key with `MDB_DUPSORT`, depending on your needs:

### 1. Using `iternext_dup()` Iterator

The most straightforward approach is to position the cursor at the key and iterate through its duplicates: py-lmdb:
1829-1844

```python
import lmdb


env = lmdb.open('/tmp/mydb')
db = env.open_db(b'mydb', dupsort=True)

with env.begin() as txn:
    cursor = txn.cursor(db)

    # Position at the key
    if cursor.set_key(b'my_key'):
        # Iterate through all values for this key
        for value in cursor.iternext_dup():
            print(value)
```

This uses `MDB_NEXT_DUP` internally to move through duplicate values. py-lmdb:1844-1844

### 2. Using `getmulti()` for Batch Retrieval

For better performance, especially with fixed-size values, use `getmulti()` which can retrieve multiple values
efficiently: py-lmdb:2103-2134

```python
# For variable-size values
with env.begin() as txn:
    cursor = txn.cursor(db)
    results = cursor.getmulti([b'key1', b'key2'], dupdata=True)
    for key, value in results:
        print(f"{key}: {value}")
```

### 3. Bulk Operations with `MDB_DUPFIXED`

If all duplicate values are fixed-size, you can use `MDB_GET_MULTIPLE` and `MDB_NEXT_MULTIPLE` for maximum efficiency:
py-lmdb:373-384

```python
# Open database with dupfixed flag
db = env.open_db(b'mydb', dupsort=True, dupfixed=True)

with env.begin() as txn:
    cursor = txn.cursor(db)
    # Retrieve all values in batches
    results = cursor.getmulti([b'my_key'], dupdata=True, dupfixed_bytes=8)
```

The `dupfixed_bytes` parameter tells `getmulti()` to use `MDB_GET_MULTIPLE` which retrieves up to a page of duplicate
values at once. py-lmdb:2142-2147

The implementation fetches multiple values per cursor operation, significantly reducing overhead: py-lmdb:2124-2131

### 4. Manual Cursor Navigation

You can also manually navigate using cursor methods: py-lmdb:1943-1954

```python
with env.begin() as txn:
    cursor = txn.cursor(db)

    # Position at first duplicate
    if cursor.set_key(b'my_key'):
        cursor.first_dup()  # Move to first value

        # Collect all values
        values = []
        while True:
            values.append(cursor.value())
            if not cursor.next_dup():  # Move to next duplicate
                break
```

## Performance Comparison

- **`iternext_dup()`**: Simple, good for small numbers of duplicates
- **`getmulti()` without `dupfixed_bytes`**: Uses `MDB_NEXT_DUP`, one value per operation py-lmdb:2145-2147
- **`getmulti()` with `dupfixed_bytes`**: Uses `MDB_GET_MULTIPLE`/`MDB_NEXT_MULTIPLE`, retrieves page-sized batches
  py-lmdb:6459-6495 - **most efficient for fixed-size values**

## Notes

The bulk operations (`MDB_GET_MULTIPLE`/`MDB_NEXT_MULTIPLE`) are only available when the database is opened with
`MDB_DUPFIXED`, meaning all duplicate values must be the same size. py-lmdb:6464-6467 This was added in py-lmdb v1.1.0
with the `getmulti()` method. py-lmdb:97-98

Wiki pages you might want to explore:

- [Architecture (jnwatson/py-lmdb)](/wiki/jnwatson/py-lmdb#4)
- [LMDB Integration (jnwatson/py-lmdb)](/wiki/jnwatson/py-lmdb#4.2)

### Citations

**File:** lmdb/cffi.py (L1829-1844)

```python
    def iternext_dup(self, keys=False, values=True):
    """Return a forward iterator that yields the current value
    ("duplicate") of the current key before calling :py:meth:`next_dup`,
    repeating until the last value of the current key is reached.

    Only meaningful for databases opened with `dupsort=True`.

    .. code-block:: python

        if not cursor.set_key("foo"):
            print("No values found for 'foo'")
        else:
            for idx, data in enumerate(cursor.iternext_dup()):
                print("%d'th value for 'foo': %s" % (idx, data))
    """
    return self._iter(_lib.MDB_NEXT_DUP, keys, values)
```

**File:** lmdb/cffi.py (L1943-1954)

```python
    def first_dup(self):
    """Move to the first value ("duplicate") for the current key, returning
    ``True`` on success or ``False`` if the database is empty.

    Only meaningful for databases opened with `dupsort=True`.

    Equivalent to `mdb_cursor_get()
    <http://lmdb.tech/doc/group__mdb.html#ga48df35fb102536b32dfbb801a47b4cb0>`_
    with `MDB_FIRST_DUP
    <http://lmdb.tech/doc/group__mdb.html#ga1206b2af8b95e7f6b0ef6b28708c9127>`_
    """
    return self._cursor_get(_lib.MDB_FIRST_DUP)
```

**File:** lmdb/cffi.py (L2103-2134)

```python
    def getmulti(self, keys, dupdata=False, dupfixed_bytes=None, keyfixed=False):
    """Returns an iterable of `(key, value)` 2-tuples containing results
    for each key in the iterable `keys`.

        `keys`:
            Iterable to read keys from.

        `dupdata`:
            If ``True`` and database was opened with `dupsort=True`, read
            all duplicate values for each matching key.

        `dupfixed_bytes`:
            If database was opened with `dupsort=True` and `dupfixed=True`,
            accepts the size of each value, in bytes, and applies an
            optimization reducing the number of database lookups.

        `keyfixed`:
            If `dupfixed_bytes` is set and database key size is fixed,
            setting keyfixed=True will result in this function returning
            a memoryview to the results as a structured array of bytes.
            The structured array can be instantiated by passing the
            memoryview buffer to NumPy:

            .. code-block:: python

                key_bytes, val_bytes = 4, 8
                dtype = np.dtype([(f'S{key_bytes}', f'S{val_bytes}}')])
                arr = np.frombuffer(
                    cur.getmulti(keys, dupdata=True, dupfixed_bytes=val_bytes, keyfixed=True)
                )

    """
```

**File:** lmdb/cffi.py (L2142-2147)

```python
        if dupfixed_bytes:
    get_op = _lib.MDB_GET_MULTIPLE
    next_op = _lib.MDB_NEXT_MULTIPLE
else:
    get_op = _lib.MDB_GET_CURRENT
    next_op = _lib.MDB_NEXT_DUP
```

**File:** lib/lmdb.h (L373-384)

```text
	MDB_GET_MULTIPLE,		/**< Return up to a page of duplicate data items
								from current cursor position. Move cursor to prepare
								for #MDB_NEXT_MULTIPLE. Only for #MDB_DUPFIXED */
	MDB_LAST,				/**< Position at last key/data item */
	MDB_LAST_DUP,			/**< Position at last data item of current key.
								Only for #MDB_DUPSORT */
	MDB_NEXT,				/**< Position at next data item */
	MDB_NEXT_DUP,			/**< Position at next data item of current key.
								Only for #MDB_DUPSORT */
	MDB_NEXT_MULTIPLE,		/**< Return up to a page of duplicate data items
								from next cursor position. Move cursor to prepare
								for #MDB_NEXT_MULTIPLE. Only for #MDB_DUPFIXED */
```

**File:** lmdb/cpython.c (L2124-2131)

```c
    /* Choose ops */
    if(arg.dupfixed_bytes) {
        get_op = MDB_GET_MULTIPLE;
        next_op = MDB_NEXT_MULTIPLE;
    } else {
        get_op = MDB_GET_CURRENT;
        next_op = MDB_NEXT_DUP;
    }
```

**File:** lib/mdb.c (L6459-6495)

```c
	case MDB_GET_MULTIPLE:
		if (data == NULL || !(mc->mc_flags & C_INITIALIZED)) {
			rc = EINVAL;
			break;
		}
		if (!(mc->mc_db->md_flags & MDB_DUPFIXED)) {
			rc = MDB_INCOMPATIBLE;
			break;
		}
		rc = MDB_SUCCESS;
		if (!(mc->mc_xcursor->mx_cursor.mc_flags & C_INITIALIZED) ||
			(mc->mc_xcursor->mx_cursor.mc_flags & C_EOF))
			break;
		goto fetchm;
	case MDB_NEXT_MULTIPLE:
		if (data == NULL) {
			rc = EINVAL;
			break;
		}
		if (!(mc->mc_db->md_flags & MDB_DUPFIXED)) {
			rc = MDB_INCOMPATIBLE;
			break;
		}
		rc = mdb_cursor_next(mc, key, data, MDB_NEXT_DUP);
		if (rc == MDB_SUCCESS) {
			if (mc->mc_xcursor->mx_cursor.mc_flags & C_INITIALIZED) {
				MDB_cursor *mx;
fetchm:
				mx = &mc->mc_xcursor->mx_cursor;
				data->mv_size = NUMKEYS(mx->mc_pg[mx->mc_top]) *
					mx->mc_db->md_pad;
				data->mv_data = METADATA(mx->mc_pg[mx->mc_top]);
				mx->mc_ki[mx->mc_top] = NUMKEYS(mx->mc_pg[mx->mc_top])-1;
			} else {
				rc = MDB_NOTFOUND;
			}
		}
```

**File:** ChangeLog (L97-98)

```text
* Add a Cursor.getmulti method with optional buffer support.  Contributed by
  Will Thompson <willsthompson@gmail.com>.
```
