# Py-Polygon-BoolTools
Performs boolean operations on polygons (intersection(clipping) / union / difference)

Only the intersection method is implemented right now, but extending to union and difference operations is as simple as a very simple decision change in the algorithm.

- Can't have overlapping vertices at the moment.
- Overlapping edges are ignored (zero intersection area cases)
