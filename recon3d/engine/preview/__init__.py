"""Preview renders: stills, turntables, wireframe/normal/UV/coverage sheets.

The stage is optional and never fails a run; ``render_preview`` and friends are
imported lazily by the pipeline.  The explicit package marker keeps the module in
installed distributions (a namespace directory works from a checkout but not from
a wheel).
"""
