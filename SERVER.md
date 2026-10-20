# Running on the Linux GPU server (2× Titan RTX)

Step by step, from your laptop's terminal. Lines starting with `$` are commands you type; leave out the `$`.

## 0. What the server needs (check once)
```bash
$ ssh <user>@<server-address>          # log in
$ nvidia-smi                           # must list 2x "TITAN RTX"; "Driver Version" must be >= 515
$ conda --version                      # conda / miniconda / mamba must exist (otherwise install Miniconda in your home dir)
$ df -h ~                              # need ~40 GB free for the code + weights, plus room for results
$ curl -sI https://github.com | head -1   # internet access (setup downloads ~12 GB of model weights)
```
If any of these fail, send the output to me (or to whoever runs the server). Installing Miniconda needs no admin rights.

## 1. Get the code (once)
```bash
$ git clone https://github.com/<you>/<repo>.git CVProject      # the private repo; GitHub will ask for a login/token
$ cd CVProject
```
Copy your licensed SMPL file to the server too (it's deliberately not in git). From your **laptop**:
```bash
scp "SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl" <user>@<server-address>:~/
```

## 2. Install (once, ~30–60 min)
Run it inside `tmux`, so it survives if your ssh connection drops:
```bash
$ tmux new -s setup
$ bash setup_server.sh --smpl ~/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
#   detach and leave it running: press Ctrl-b, then d.   Come back later:  tmux attach -t setup
```
It ends with "TubeR and LART imports OK". If it stops with an error, copy the last ~30 lines to me.

## 3. A short test first (your videos stay untouched)
```bash
$ conda activate cvproject
$ python run.py --model lart --video /path/to/cam03_4h.mp4 --out results/test_lart \
      --start 01:00:00 --duration 120 --gpus 0,1 --render
```
This copies 2 minutes (from 1:00:00) into `results/test_lart/_input/`, runs LART, and prints its speed
(e.g. "0.15x real time per GPU (4 h of video ~ 26.7 GPU-hours)"). Look at the rendered videos in
`results/test_lart/chunks/*/results_temporal_videos/` and at `detections.csv`.

## 4. The full 4-hour video
```bash
$ tmux new -s cam03
$ conda activate cvproject
$ python run.py --model lart  --video /path/to/cam03_4h.mp4 --out results/cam03_lart  --gpus 0,1 --cleanup
#   Ctrl-b d to detach. Check progress any time: tmux attach -t cam03
```
- **Interrupted?** (reboot, crash, Ctrl-C): run **the same command again**. Finished chunks are skipped.
- **TubeR instead:** `--model tuber`. It's much faster.
- **Faster LART, but misses small/distant people:** add `--lart-detector maskrcnn`.
- **Disk:** while running, `<out>/_chunks` holds a copy of the video. `--cleanup` deletes it at the end.

## 5. Results
In `results/<name>/`:

| file | contents |
|---|---|
| `detections.csv` | one row per person every 0.5 s: time (on the original video's clock), box (0–1), top-3 actions + probability |
| `timeline.csv` | one row per minute: average number of people, and of people doing each action (probability ≥ 0.5) |
| `run_info.json` | settings, timing per chunk, speed |

Copy them to your laptop: `scp -r <user>@<server-address>:~/CVProject/results/cam03_lart ./`

## Updating the code later
```bash
$ cd ~/CVProject && git pull && conda activate cvproject
```
