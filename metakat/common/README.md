# Common MetaKat components

`metakat.common.models` owns the general internal representations shared by
otherwise independent MetaKat processing packages:

- `BoundingBox` is MetaKat's axis-aligned rectangle model;
- `PageDimensions` represents a validated positive page canvas;
- `DetectionEvidence` associates detected text and confidence with a MetaKat
  bounding box and source page key.

These models are independent from similarly shaped objects in lower-level
libraries. Engines explicitly copy library geometry into MetaKat models when
data crosses that boundary.
