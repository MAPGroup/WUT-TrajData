# Public AIS Samples

This directory contains small public sample files derived from the anonymized release data in `release_anonymized/`.

These files are **not** the full datasets. They are lightweight samples intended for:

- demonstrating the public data format
- testing benchmark code
- quick integration checks

## Source

All sample files are derived from `release_anonymized/`, which has already been processed with:

- vessel ID remapping
- global timestamp shifting
- longitude/latitude preservation

No additional anonymization is applied at the sample-building stage.

## Files

- `raw_csj_sample.csv`: anonymized raw AIS sample from the CSJ area
- `raw_zs_sample.csv`: anonymized raw AIS sample from the ZS area
- `clean_csj_sample.csv`: anonymized cleaned-and-segmented AIS sample from the CSJ area
- `clean_zs_sample.csv`: anonymized cleaned-and-segmented AIS sample from the ZS area

## Unified Schema

All sample files use exactly the same schema:

```text
vessel_id,timestamp,lon,lat,sog,cog
```

Field meanings:

- `vessel_id`: public sample vessel identifier in the form `vessel_000001`
- `timestamp`: formatted as `YYYY-MM-DD HH:MM:SS`
- `lon`: longitude
- `lat`: latitude
- `sog`: speed over ground
- `cog`: course over ground

## Notes

- `timestamp` was already globally shifted during the `release_anonymized` stage. In this sample layer, timestamps are only reformatted into a unified representation.
- `lon` and `lat` preserve the spatial trajectory geometry.
- `raw_*_sample` files correspond to anonymized raw AIS records.
- `clean_*_sample` files correspond to anonymized cleaned AIS trajectories that were also split at gaps longer than 30 minutes.
