# Pole-top development v2

This deterministic development stage consumes the pinned crossarm-association v2 records and produces an unscored search region only when one guarded crossarm is associated unambiguously with the upper endpoint of one upright pole.

It abstains on 26 of 27 UK development images and emits one geometry candidate. The candidate is centred on the guarded crossarm and sized from the crossarm extent, pole width and image scale. It is not an annotated physical component, model detection, confidence or accuracy result. UK v3 was not accessed, and no model inference ran.

The result SHA-256 is `ef371bd66e45df17ddbdbba133ad76fde03be98d72aa63e5c121b78fb4419beb`. The integrated English comparison is served under `report/crossarm_association_v2/index.html`.
