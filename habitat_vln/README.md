# Habitat VLN Experiments

This directory is the single home for the Habitat vision-and-language navigation
experiments.

## Naming

- **VLN** names the task: vision-and-language navigation.
- **VLM** names the model family used as a policy, such as Qwen-VL.

So the package is `habitat_vln`, while model-facing code can still use names
like `vlm_policy.py` or `QwenVLMPolicy`.

## Entrypoints

- `habitat_vln_nav.py`: main Habitat-Lab environment runner with policy classes,
  standard VLN episodes, frame export, video export, Success/SPL metrics, and
  trajectory CSV logging.
- `qwen_habitat_sim_nav.py`: direct `habitat_sim` Qwen runner kept as a focused
  experiment script. It writes a video plus an aligned trajectory CSV.

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
conda run -n habitat_vlm python habitat_vln/check_vln_data.py --splits val_seen
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
conda run -n habitat_vlm python habitat_vln/generate_hm3d_pointnav_smoke.py \
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
conda run -n habitat_vlm python habitat_vln/evaluate_pointnav_policies.py \
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

## Outputs

Runtime artifacts go under `habitat_vln/outputs/`, which is ignored by git.
