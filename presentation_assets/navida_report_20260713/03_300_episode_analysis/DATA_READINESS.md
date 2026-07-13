# HM3D NaVIDA 13-Scene Data Readiness

## Status

- Scope: HM3D engineering protocol, not an R2R/RxR benchmark.
- Train split: 300 episodes across 10 scenes, 30 episodes per scene.
- Validation split: 60 episodes across 3 scene-disjoint scenes, 20 episodes per scene.
- Scene overlap: none.
- Train Oracle success: 300 / 300.
- Train atomic state-action samples: 20,165.
- Mixed training samples: 3,000 total, with 1,500 VLN and 1,500 IDS samples.
- Qwen training: not started by this data-preparation run.

## Route Coverage

Each split contains equal requested-profile quotas:

| Profile | Train | Validation | Meaning |
|---|---:|---:|---|
| standard | 60 | 12 | Unconstrained valid indoor route |
| long_route | 60 | 12 | Geodesic route at least 8 m |
| multi_turn | 60 | 12 | At least two route direction changes of 30 degrees or more |
| low_clearance | 60 | 12 | At least 70% of sampled route points within 0.65 m of a navmesh obstacle |
| reorientation | 60 | 12 | Initial heading deliberately faces away from the route |

`low_clearance` is a geometry-based proxy for narrow areas. It is not a
semantic corridor or room label.

## State-Action Coverage

| Data | VLN rows | Episodes | Scenes | Nominal cells | Core cells | STOP episodes | Recovery rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 20,165 | 300 | 10 | 59 / 80 | 20 / 20 | 300 | 7,289 |
| Mixed VLN half | 1,500 | 300 | 10 | 54 / 80 | 20 / 20 | 300 | 426 |

Oracle action counts are 10,898 move-forward, 4,500 turn-left, 4,467
turn-right, and 300 STOP actions. Recovery coverage contains 7,288
no-progress/reorientation rows and one naturally observed collision row. No
deliberate bad-action collision demonstrations were injected into VLN labels.

## Artifacts

- Dataset report: `datasets/dataset_report.json`
- Train episodes: `datasets/train/train.json.gz`
- Validation episodes: `datasets/val/val.json.gz`
- Oracle manifest: `oracle/collect_20260713_135247/manifest.jsonl`
- Oracle episode results: `oracle/collect_20260713_135247/episodes.csv`
- Oracle coverage: `coverage/oracle_20260713_135528/coverage.md`
- Mixed manifest: `mixed/build_20260713_135434/manifest.jsonl`
- Mixed summary: `mixed/build_20260713_135434/summary.json`
- Mixed coverage: `coverage/mixed_20260713_135529/coverage.md`
- Pipeline state: `pipeline_state.json`

## HM3D Citation

```bibtex
@inproceedings{ramakrishnan2021hm3d,
  title={Habitat-Matterport 3D Dataset ({HM}3D): 1000 Large-scale 3D Environments for Embodied {AI}},
  author={Santhosh Kumar Ramakrishnan and Aaron Gokaslan and Erik Wijmans and Oleksandr Maksymets and Alexander Clegg and John M Turner and Eric Undersander and Wojciech Galuba and Andrew Westbury and Angel X Chang and Manolis Savva and Yili Zhao and Dhruv Batra},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2021},
  url={https://arxiv.org/abs/2109.08238}
}
```

The citation is also embedded in the metadata of both generated episode files
and in `datasets/dataset_report.json`.
