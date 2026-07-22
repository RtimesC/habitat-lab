# Habitat VLN Experiments

This directory is the single home for the Habitat vision-and-language navigation
experiments.

The runtime module boundaries and simulator-to-robot extension points are described
in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Naming

- **VLN** names the task: vision-and-language navigation.
- **VLM** names the model family used as a policy, such as Qwen-VL.

So the package is `habitat_vln`, while policy implementations and prompt
templates live under `habitat_vln/policies/`.

## Entrypoints

- `habitat_vln_nav.py`: main Habitat-Lab command entrypoint for standard VLN
  episodes, frame export, video export, Success/SPL metrics, and trajectory CSV
  logging.
- `legacy/qwen_habitat_sim_nav.py`: archived direct-`habitat_sim` Qwen runner.
  It remains available for historical comparison but is not a framework entrypoint.
- `python -m habitat_vln.data.<tool>`: dataset checks, generation, collection,
  coverage analysis, and training-record construction.
- `python -m habitat_vln.training.<tool>`: Qwen and NaVIDA QLoRA training.
- `python -m habitat_vln.evaluation.<tool>`: offline and closed-loop evaluation.
- `python -m habitat_vln.pipelines.hm3d_navida_pipeline`: staged HM3D workflow.

## Standard VLN Run

```bash
conda run -n habitat_vlm python habitat_vln/habitat_vln_nav.py \
  --dataset-split val_seen \
  --num-episodes 1 \
  --max-steps 80
```

The default task config is `benchmark/nav/vln_r2r.yaml`, which expects R2R data
under `data/datasets/vln/mp3d/r2r/v1/{split}/{split}.json.gz` and Matterport3D
scenes under `data/scene_datasets/`. Use `--dataset-path` and `--scenes-dir` to
override those paths.

## Data Setup

Download the R2R VLN episode data:

```bash
mkdir -p data/datasets/vln/mp3d/r2r/v1
wget -O /tmp/vln_r2r_mp3d_v1.zip \
  https://dl.fbaipublicfiles.com/habitat/data/datasets/vln/mp3d/r2r/v1/vln_r2r_mp3d_v1.zip
unzip -q /tmp/vln_r2r_mp3d_v1.zip -d data/datasets/vln/mp3d/r2r/v1
```

Expected files include:

```text
data/datasets/vln/mp3d/r2r/v1/train/train.json.gz
data/datasets/vln/mp3d/r2r/v1/val_seen/val_seen.json.gz
data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz
```

Download Matterport3D scenes separately through the Matterport3D access process,
then use their `download_mp.py` script for Habitat assets:

```bash
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

Expected scene files look like:

```text
data/scene_datasets/mp3d/{scene_id}/{scene_id}.glb
```

Quick checks:

```bash
test -f data/datasets/vln/mp3d/r2r/v1/val_seen/val_seen.json.gz
find data/scene_datasets/mp3d -mindepth 2 -maxdepth 2 -name '*.glb' | head
conda run -n habitat_vlm python -m habitat_vln.data.check_vln_data --splits val_seen
```

## HM3D Smoke Test

HM3D cannot replace MP3D for the R2R benchmark because R2R episodes reference
MP3D scene ids, start positions, and goal positions. It is still useful for a
pipeline smoke test with PointNav Success/SPL.

Download a small HM3D example scene:

```bash
conda run -n habitat_vlm python -m habitat_sim.utils.datasets_download \
  --uids hm3d_example_habitat \
  --data-path data/ \
  --no-replace
```

If the GitHub media download is flaky, resume the tar download manually and then
rerun the downloader command above so it can extract/link the files:

```bash
curl -L --retry 8 --retry-delay 3 --continue-at - \
  -o data/hm3d-example-habitat-v0.2.tar \
  https://media.githubusercontent.com/media/matterport/habitat-matterport-3dresearch/main/example/hm3d-example-habitat-v0.2.tar
```

Generate a tiny HM3D PointNav dataset from the downloaded scene:

```bash
conda run -n habitat_vlm python -m habitat_vln.data.generate_hm3d_pointnav_smoke \
  --episodes 20 \
  --min-distance 2 \
  --max-distance 6
```

Run the same VLM navigation loop against the generated dataset:

```bash
conda run -n habitat_vlm python habitat_vln/habitat_vln_nav.py \
  --task-config benchmark/nav/pointnav/pointnav_hm3d.yaml \
  --dataset-path data/datasets/pointnav/hm3d_smoke/v1/{split}/{split}.json.gz \
  --dataset-split val \
  --instruction "Navigate to the target location and stop when you reach it." \
  --mock-policy \
  --num-episodes 1 \
  --max-steps 80
```

Check the non-VLM baselines before loading Qwen:

```bash
conda run -n habitat_vlm python -m habitat_vln.evaluation.evaluate_pointnav_policies \
  --policies oracle geometric \
  --num-episodes 20 \
  --max-steps 80 \
  --width 224 \
  --height 224
```

On the generated 2-6 m smoke set, the oracle should be near 100% success. The
geometric controller is only a simple baseline and is expected to be lower.

Remove `--mock-policy` when the Qwen environment is ready.
On an 8GB GPU, prefer the 4-bit path:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n habitat_vlm python habitat_vln/habitat_vln_nav.py \
  --task-config benchmark/nav/pointnav/pointnav_hm3d.yaml \
  --dataset-path data/datasets/pointnav/hm3d_smoke/v1/{split}/{split}.json.gz \
  --dataset-split val \
  --instruction "Navigate to the target location and stop when you reach it." \
  --num-episodes 1 \
  --max-steps 80 \
  --width 224 \
  --height 224 \
  --max-new-tokens 16 \
  --advisor-fallback follow_goal \
  --load-in-4bit
```

This loads the Qwen model. If you want to require an existing local Hugging Face
cache and avoid downloading model files, prepend
`TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`.

The runner adds the previous action, collision flag, distance-to-goal change,
and depth summary to the VLM prompt. It also enables an anti-stuck guard by
default, which overrides repeated same-direction turns with `move_forward` only
when the target is roughly ahead and center depth is safe.
Pass `--disable-anti-stuck` to inspect the raw model policy.

## Simulation frequencies

The navigation runner has two timing modes:

- `layered`: fresh RGB/depth and geometric control at 5 Hz, with a new Qwen
  advisory inference submitted at 0.5 Hz in a background worker. The latest
  completed Qwen advice is reused between model updates while the controller
  continues to use the newest observation.
- `joint`: the model receives the latest visual observation and replans at 1 Hz.
  This is the default for direct controller and NaVIDA runs.

`--frequency-mode auto` is the default. It selects `layered` for
`--qwen-role advisor` and `joint` for `--qwen-role controller`. The explicit
commands are:

```bash
# Layered visual/controller + slower Qwen advisor.
conda run -n habitat_vlm python habitat_vln/habitat_vln_nav.py \
  ... \
  --qwen-role advisor \
  --frequency-mode layered \
  --vision-hz 5 \
  --inference-hz 0.5

# Joint visual-plus-inference model loop.
conda run -n habitat_vlm python habitat_vln/habitat_vln_nav.py \
  ... \
  --qwen-role controller \
  --frequency-mode joint \
  --joint-hz 1
```

The runner sleeps so a loop never runs faster than its requested wall-clock
rate. A model inference that itself takes longer than the requested period will
make the measured rate slower; it will never skip safety checks to catch up.
Use `--no-rate-limit` only for fast tests: the same inference schedule is then
preserved in logical simulator time without wall-clock sleeps.
Saved video defaults to the active visual frequency (5 FPS in layered mode and
1 FPS in joint mode); `--video-fps` can override playback speed only.

For videos intended for visual inspection, use the native 4:3 viewing preset:

```bash
--width 640 --height 480 --hfov 90
```

Avoid `224x224` except for quick plumbing checks: it reduces scene detail and
leaves too little horizontal space for status text. Video duration is determined
by `max_steps / active_visual_hz`. For approximately 10 seconds, use
`--max-steps 50` in layered 5 Hz mode or `--max-steps 10` in joint 1 Hz mode.
The status overlay uses three short lines, resolution-aware text sizing, and a
dark translucent background so actions are not clipped on small frames.

`trajectory.csv` records `frequency_mode`, target frequencies, logical and
wall time, whether inference started or completed on the step, inference
duration, and cached decision age. Joint NaVIDA runs keep one action from each
generated chunk, so every 1 Hz tick replans from the latest image.

## Outputs

Runtime artifacts go under `habitat_vln/outputs/`, which is ignored by git.
Videos are transcoded to browser-compatible H.264 with `ffmpeg`, so they can be
previewed in VS Code. If `ffmpeg` or `libx264` is unavailable, the runtime keeps
the original `mp4v` video and prints a warning instead of discarding it.

## QLoRA Training Pipeline

The training pipeline learns the four executable Habitat actions from Oracle
trajectories. Each JSONL sample contains an RGB image, the episode instruction,
the navigation state used by the current inference prompt, and the Oracle
action. Images from one episode always stay in the same train/validation split.

This first pipeline is an inspectable controller-imitation baseline. Its prompt
contains Habitat's true target distance and angle, so it does not yet represent
instruction-only VLN and cannot be transferred directly to the real robot. A
later experiment must remove or replace this privileged simulator state.

The pipeline can be checked now, without MP3D and without loading Qwen:

```bash
conda run -n habitat_vlm python -m habitat_vln.training.train_qwen_qlora --self-test
```

After the MP3D scenes are available, first collect a small R2R Oracle dataset:

```bash
conda run -n habitat_vlm python -m habitat_vln.data.collect_oracle_training_data \
  --dataset-split train \
  --num-episodes 100 \
  --max-steps 500
```

The collector creates a timestamped directory under
`habitat_vln/outputs/oracle_training_data/`. It contains:

- `manifest.jsonl`: samples accepted for training.
- `episodes.csv`: success and inclusion status for every attempted episode.
- `images/`: the RGB observation saved before each Oracle action.

Failed or truncated Oracle episodes remain visible in `episodes.csv` but are
excluded from `manifest.jsonl` by default. Validate the generated data before
loading the model:

```bash
conda run -n habitat_vlm python -m habitat_vln.training.train_qwen_qlora \
  --manifest habitat_vln/outputs/oracle_training_data/collect_*/manifest.jsonl \
  --dry-run
```

Install the one additional training dependency only when a real QLoRA run is
ready:

```bash
conda run -n habitat_vlm python -m pip install \
  -r habitat_vln/requirements-training.txt
```

Start with a deliberately small overfitting run on the RTX 4060 8GB GPU:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n habitat_vlm python -m habitat_vln.training.train_qwen_qlora \
  --manifest habitat_vln/outputs/oracle_training_data/collect_*/manifest.jsonl \
  --max-samples 100 \
  --validation-ratio 0.1 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8
```

The final adapter is saved under
`habitat_vln/outputs/qwen_qlora/final_adapter/`. Evaluate it by keeping the base
model unchanged and adding `--adapter-path`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n habitat_vlm python habitat_vln/habitat_vln_nav.py \
  --dataset-split val_seen \
  --adapter-path habitat_vln/outputs/qwen_qlora/final_adapter \
  --load-in-4bit \
  --num-episodes 1 \
  --max-steps 80 \
  --width 224 \
  --height 224
```

## NaVIDA Core Reproduction

The NaVIDA path extends the single-step Oracle baseline with the paper's two
central ideas:

- Hierarchical Probabilistic Action Chunking (HPAC) converts atomic Habitat
  actions into variable-length chunks. The defaults match the paper:
  `merge_probability=0.7` and `max_level2_chunks=3`.
- Inverse Dynamics Supervision (IDS) pairs the RGB image before a chunk with the
  RGB image after it and asks Qwen to recover the intervening action chunk.

The collector now records `next_image` for every Oracle action. Existing
single-step manifests remain valid because this is an optional extra field.
Collect a deliberately small Habitat trajectory first:

```bash
conda run -n habitat_vlm python -m habitat_vln.data.collect_oracle_training_data \
  --task-config benchmark/nav/pointnav/pointnav_hm3d.yaml \
  --dataset-path 'data/datasets/pointnav/hm3d_smoke/v1/{split}/{split}.json.gz' \
  --dataset-split val \
  --scenes-dir data/scene_datasets \
  --instruction 'Navigate to the target location and stop when you reach it.' \
  --num-episodes 1 \
  --max-steps 80
```

Convert the timestamped Oracle manifest into paired VLN and IDS records:

```bash
conda run -n habitat_vlm python -m habitat_vln.data.build_navida_training_data \
  --source-manifest habitat_vln/outputs/oracle_training_data/collect_*/manifest.jsonl
```

The converter writes `navida/manifest.jsonl`, `navida/episodes.csv`, and
`navida/summary.json` beside the source manifest. Validate the full mixed-task
schema without loading Qwen:

```bash
conda run -n habitat_vlm python -m habitat_vln.training.train_navida_qlora \
  --manifest habitat_vln/outputs/oracle_training_data/collect_*/navida/manifest.jsonl \
  --validation-ratio 0 \
  --dry-run
```

When multiple successful episodes are available, start a small QLoRA run by
removing `--dry-run` and using a nonzero episode-level validation ratio. The
NaVIDA trainer freezes visual parameters and jointly learns the same JSON action
grammar from instruction-conditioned VLN samples and two-frame IDS samples.

Score generated action chunks separately for VLN and IDS:

```bash
conda run -n habitat_vlm python -m habitat_vln.evaluation.evaluate_navida_outputs \
  --manifest path/to/navida/manifest.jsonl \
  --adapter-path path/to/final_adapter \
  --output-dir path/to/offline_eval \
  --samples-per-task 10
```

Run the trained adapter as a chunked Habitat controller:

```bash
conda run -n habitat_vlm python habitat_vln/habitat_vln_nav.py \
  --task-config benchmark/nav/pointnav/pointnav_hm3d.yaml \
  --dataset-path 'data/datasets/pointnav/hm3d_smoke/v1/{split}/{split}.json.gz' \
  --dataset-split val \
  --scenes-dir data/scene_datasets \
  --instruction 'Navigate to the target location and stop when you reach it.' \
  --qwen-role controller \
  --navida-chunk-policy \
  --navida-max-executed-actions 1 \
  --adapter-path path/to/final_adapter \
  --load-in-4bit
```

`--force-stop-within-success-radius` is an optional privileged PointNav safety
guard. Results using it must be reported separately from the pure NaVIDA policy.

## Official NaVIDA HTTP Policy

Official NaVIDA runs in a separate Python 3.10 server process. The Habitat-side
Python 3.9 runtime connects to it over localhost and does not load Qwen,
Transformers, a local checkpoint, or an adapter:

```bash
conda run -n habitat_vlm python habitat_vln/habitat_vln_nav.py \
  --dataset-split val_seen \
  --official-navida-http \
  --official-navida-url http://127.0.0.1:8008 \
  --policy-protocol paper_pure
```

On native Linux, the renderer device from the task configuration is used by
default. Under WSLg/Mesa, CUDA-to-EGL device matching may fail for device `0`.
In that case, add `--gpu-device-id -1` so EGL selects the default WSLg renderer.
This changes only Habitat rendering; the separate NaVIDA server continues to
use its configured CUDA device.

The adapter calls `POST /v1/episodes/start` after every environment reset. On
each simulator step it calls `POST /v1/steps` with only the episode id,
simulator step, instruction, and a lossless base64-encoded RGB PNG. Depth, pose,
goal geometry, collision state, and other privileged simulator values are not
sent to the server.

`paper_pure` always uses joint scheduling. A valid atomic action, including an
early `stop`, is sent directly to Habitat without geometric, success-radius, or
anti-stuck overrides. An invalid response or HTTP failure ends that episode as
failed without executing a fallback action. Server decision ids, inference
status, latency, termination reason, and metadata are saved in
`trajectory.csv`.

## HM3D NaVIDA Engineering Pipeline

Use `habitat_vln.pipelines.hm3d_navida_pipeline` to run the HM3D-only workflow
in explicit stages.
The default small protocol uses two local HM3D example scenes for training and
one scene-disjoint example scene for validation. It is an engineering protocol,
not an R2R/RxR benchmark. No stage downloads HM3D automatically.

All commands run from the repository root in the `habitat_vlm` environment. Use
the same workspace for every stage:

```bash
WORKSPACE=habitat_vln/outputs/hm3d_navida_system

conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" check
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" prepare
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" collect
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" build --max-samples 100
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" train --epochs 3
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" offline-eval --samples-per-task 20
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" closed-loop --num-episodes 2 \
  --max-executed-actions 3
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" report
```

The generated PointNav episodes store a small set of truthful engineering
instructions. VLN samples also include relative target distance and bearing in
the prompt because an arbitrary PointNav coordinate is not observable from RGB
or a generic instruction alone. IDS samples remain image-pair-only. The pipeline
records current artifact paths in `pipeline_state.json`, uses timestamped build
and training directories, and keeps pure versus privileged-guarded closed-loop
results separate.

### Scaling episodes and state-action coverage

Before a longer QLoRA run, expand successful Oracle trajectories and measure
whether the new data covers useful navigation states. The following small-scale
profile uses the three local HM3D example scenes: two train scenes, one
scene-disjoint validation scene, 60 successful train episodes, and 600 mixed
VLN/IDS samples.

```bash
WORKSPACE=habitat_vln/outputs/hm3d_navida_scale_60ep

conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" prepare \
  --train-scene-count 2 --val-scene-count 1 \
  --train-episodes 60 --val-episodes 15 \
  --dataset-label hm3d_example_scale_60ep_v1
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" collect --num-episodes 60 --max-steps 80
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" coverage --data oracle \
  --minimum-cell-count 20
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" build --max-samples 600 \
  --sample-strategy coverage-balanced
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" coverage --data mixed \
  --minimum-cell-count 5
```

`coverage-balanced` keeps one terminal chunk per episode, then favors rare
target-bearing, target-distance, and Oracle-first-action combinations. Coverage
artifacts are written as JSON, CSV, and Markdown under `coverage/`. The report
contains both the complete nominal grid and a 20-cell core control grid for
left/right/ahead movement and stopping.

To add scenes without downloading the complete HM3D train split, use the
official Habitat-Sim downloader for the v0.2 minival Habitat assets and configs:

```bash
conda activate habitat_vlm
python -m habitat_sim.utils.datasets_download \
  --uids hm3d_minival_habitat_v0.2 hm3d_minival_configs_v0.2 \
  --data-path data/ --no-replace \
  --username YOUR_HM3D_USERNAME --password YOUR_HM3D_PASSWORD
```

These minival sources require HM3D/Matterport authentication. Run the command
yourself in a private local terminal; do not paste credentials into project
files, chat, or saved scripts. The current Habitat-Sim downloader forwards the
credentials to its download process, so avoid shared machines and clear shell
history afterward. Semantic annotations are not required by this RGB/depth
navigation pipeline.

### 13-scene coverage-oriented protocol

With the three example scenes plus ten minival scenes installed, use ten
scene-disjoint training environments and three validation environments. The
coverage sampler cycles evenly through standard, long-route, multi-turn,
low-clearance, and reorientation episodes. Low clearance is a navmesh geometry
proxy for narrow areas, not a semantic room or corridor annotation.

```bash
WORKSPACE=habitat_vln/outputs/hm3d_navida_scale_300ep_20260713

conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" prepare \
  --train-scene-count 10 --val-scene-count 3 \
  --train-episodes 300 --val-episodes 60 \
  --min-distance 2 --max-distance 15 \
  --sampling-profile coverage --long-distance 8 \
  --min-route-turns 2 --turn-threshold-deg 30 \
  --clearance-threshold 0.65 --min-low-clearance-fraction 0.7 \
  --dataset-label hm3d_13scene_coverage_300ep_v1
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" collect --num-episodes 300 --max-steps 180
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline \
  --workspace "$WORKSPACE" coverage --data oracle --minimum-cell-count 50
```

Every generated dataset embeds the following HM3D citation in its metadata:

```bibtex
@inproceedings{ramakrishnan2021hm3d,
  title={Habitat-Matterport 3D Dataset ({HM}3D): 1000 Large-scale 3D Environments for Embodied {AI}},
  author={Santhosh Kumar Ramakrishnan and Aaron Gokaslan and Erik Wijmans and Oleksandr Maksymets and Alexander Clegg and John M Turner and Eric Undersander and Wojciech Galuba and Andrew Westbury and Angel X Chang and Manolis Savva and Yili Zhao and Dhruv Batra},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2021},
  url={https://arxiv.org/abs/2109.08238}
}
```
