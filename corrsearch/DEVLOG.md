## NOTE: 02/10/2021.
In `ObjectState`: It appears that `hash(frozenset(...))`
does not always yield the same result even if the `attributes.items()`
are the same. But I cannot reproduce this by myself. I observed
this when using pickling pgmpy's DiscreteFactor. Recommendation:
Do not return `_hashcode` for the `__hash__()` function. Override
this function in your child class.
