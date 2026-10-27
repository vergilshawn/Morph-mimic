# Third-party assets and data

Robot models and motion data retain the rights and licenses of their respective
authors. This repository does not assign a new license to third-party assets.

- **G1**: robot files were migrated from the local `supermimic` project. Preserve
  [MJLAB_LICENSE](assets/g1/MJLAB_LICENSE), which contains the Apache License 2.0,
  and the separate [LAFAN1 retargeting license](assets/g1/LAFAN1_Retargeting_Dataset_LICENSE)
  and [dataset README](assets/g1/LAFAN1_Retargeting_Dataset_README.md).
  Their presence does not relicense unrelated files. Per-file source paths and
  hashes are recorded in [the G1 manifest](assets/g1/manifest.json).
- **Mojia**: robot descriptions, meshes and control settings were migrated from
  the local `supermimic` project. No dedicated Mojia asset license was found in
  the migrated files. Preserve the original ownership and licensing terms;
  [the Mojia manifest](assets/mojia/manifest.json) records the file provenance.
- **External motion datasets**: downloaded archives, motion arrays, retargeted
  samples and generated contact labels are not included in Git. Consult
  [the data guide](docs/DATA_SOURCES.md) and
  [the download inventory](datasets/external/README.md) for the upstream sources.
  Check each source's terms before redistributing its data or derivatives.
- **Research papers**: local copies of papers and source-inspection snapshots
  are not included. The documentation links to their original publications.

No repository-wide license has been selected for the original project code.
