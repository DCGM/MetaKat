# Page-number processing

Page-number processing is split into two engines:

- a core engine that detects and aligns printed page-number regions;
- a bind engine that writes the selected page number and its geometry mapping
  to the MetaKat representation.

Each engine is supplied as a directory containing a
`metakat_engine_config.json` file. The initial implementations use the names
`page_number_core_engine_yolo` and `page_number_bind_engine_base`.
