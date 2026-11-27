# Option D — unconditional DDPM on AffectNet (Google Cloud)

Textbook DDPM (Ho et al. 2020; Lai et al., *The Principles of Diffusion Models*, Sec. 2.2) trained from
scratch with the production `diffusers` U-Net + DDPM scheduler and our own small training loop.
No labels: it generates random faces. Nothing here runs on your PC — everything runs on the cloud VM.

| File | Role |
|---|---|
| `configs/affectnet64.json` | every setting (data, U-Net, noise schedule, training, sampling) |
| `data.py` | AffectNet JPEGs only (no FFHQ), exact duplicates removed, seeded 80/20 split, 64x64 RGB in memory |
| `model.py` | builds the U-Net / scheduler; **diffusers <-> book notation table**; Cloud Storage helpers |
| `train.py` | Eq. (2.2.10) training loop, EMA, bf16 on L4, checkpoint + **automatic resume** via the bucket |
| `sample.py` | DDPM-1000 samples, backward/forward-pass pictures, copying check, FID |
| `startup.sh` | the VM's startup script: setup -> tests -> train -> sample -> upload -> **VM deletes itself** |
| `tests/` | data selection (runs anywhere), book equations == diffusers, exact resume (run on the VM) |

Data (measured on the local copy): 30,626 files -> 19,916 AffectNet JPEGs -> 18,074 unique
-> **14,459 train / 3,615 test**.

## One-time setup (console, bucket `your-bucket`)

1. **Data**: `affectnet.zip` (Train/, Test/, labels.csv) uploaded to the bucket root. ✔ (you did this)
2. **Code**: bucket -> *Create folder* `code` -> open it -> *Upload folder* -> choose this folder
   `ddpm_affectnet` (result: `gs://your-bucket/code/ddpm_affectnet/train.py`, ...).
3. **Permissions for the VM** (your organisation gives the default VM account no rights):
   * find the account: *IAM & Admin -> Service accounts* -> `<number>-compute@developer.gserviceaccount.com`
   * bucket `your-bucket` -> *Permissions* -> *Grant access* -> that account -> role **Storage Object Admin**
   * *IAM & Admin -> IAM* -> *Grant access* -> that account -> role **Compute Instance Admin (v1)**
     (lets the VM delete itself when finished; without it the VM only powers off and you delete it)
4. **Internet** (external IPs are blocked by policy): a **Cloud NAT** in the VM's region
   (`us-central1`: `nat-us` / `router-us`), and on *VPC network -> default -> subnet us-central1 -> Edit*
   set **Private Google Access: On** (free; bucket traffic then skips the NAT's per-GB fee).

## Each run

**Create instance** (same as option A, plus three settings):

| Setting | Value |
|---|---|
| Region / zone | `us-central1`, zone *Any* (T4 fallback: GPU type NVIDIA T4, n1-standard-4) |
| Machine | GPUs tab -> NVIDIA L4 x1 -> g2-standard-4 |
| Boot disk | Deep Learning VM with CUDA + Pytorch M132 (Ubuntu 24.04), 100 GB (re-check after choosing the machine) |
| Provisioning | Standard; time limit **5 hours**, on termination **Delete** (safety net) |
| Data protection | no backups |
| **Security -> Access scopes** | **Allow full access to all Cloud APIs** |
| **Advanced -> Metadata** | key `bucket` = `your-bucket`; for the first run also `steps` = `200` |
| **Advanced -> Automation -> Startup script** | paste the whole `startup.sh` |

1. **Timing run first** (metadata `steps` = `200`): creates the environment, runs the tests, trains 200 steps
   and deletes the VM (~15 min). Read `gs://your-bucket/runs/affectnet64_uncond/logs/startup.log`:
   the `s/step` value gives the real duration of the full run.
2. **Full run**: create the VM again **without** the `steps` key. Training resumes from step 200, runs to
   30,000, samples, evaluates, writes `DONE`, deletes the VM.

**Watch progress** (no need to keep anything open): bucket -> `runs/affectnet64_uncond/`
* `logs/startup.log` (every 5 min), `train_log.csv` + `checkpoint.pt` (every 2,000 steps),
  `previews/preview_stepNNNNNN.png` (every 5,000 steps), finally `results/` and `DONE`.

**Results** (`runs/affectnet64_uncond/results/`): `samples_ddpm1000.png`, `backward_pass.png`,
`forward_pass.png`, `results.json` (timings, copying check, FID). Download them from the bucket page.

**If anything stops early** (power, time limit, error): just create the VM again with the same metadata —
it resumes from the last checkpoint. A different config is refused (use a new `run-name`).

## Cleanup after the project

VM instances (should already be gone), Cloud NAT `nat-us` + router `router-us`, and when you no longer need
them, the bucket files (`checkpoint.pt` is ~0.5 GB).
