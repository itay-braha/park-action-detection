# 6. Park CCTV in Israel: domain notes, privacy and ethics

## 6.1 Why park CCTV is hard for an AVA-trained model

AVA is Hollywood movies: people are large and centred, with good lighting and colour at eye level.
Your footage breaks almost all of these assumptions:

| Challenge | Why it hurts | What to do in this project |
|---|---|---|
| **Small, distant people** (cameras on 4–6 m poles covering a whole lawn) | At a 256 px input, a person 60 px tall in 1080p becomes ~14 px, and boxes under 30 px² are dropped by the crop transform | Raise `IMG_SIZE` (320–384) with frames extracted at `--short-side` ≥ that. Report AP binned by box height. For very wide views, consider tiling the frame into overlapping crops at inference (not implemented) |
| **High viewpoint / top-down angle** | Poses look different from eye-level movie shots; "sit" vs. "stand" is harder | Fine-tune deeper (`FREEZE_BACKBONE: layer2` or lower) once you have enough data. Pre-fine-tune on MEVA/VIRAT (see [05](05_related_work_and_forks.md)) |
| **Night IR (grey) footage** | Colour cues vanish, noise increases, IR illuminators create hotspots | `DATA.GRAYSCALE_P`. Make sure val has night clips, and report day and night separately |
| **Hard Israeli sun: harsh shadows, over-exposure, heat haze** in summer afternoons | Shadows get detected as people, and haze blurs thin limbs | Annotate midday summer footage specifically. Colour jitter is on by default |
| **Low bit-rate / sub-stream recordings** (H.264/H.265 at 2–4 Mbps, often 12–15 fps) | Compression blocks hide motion, and low fps undersamples fast actions | Use the main stream if you can. We re-sample to 30 fps so the time scale is right, but duplicated frames carry no new information: for 12 fps sources, consider `FRAME_RATE: 1` |
| **Burnt-in OSD** (timestamp, camera name) | Constant text the model may latch onto | Crop or mask it during extraction if it covers the scene (a small change in `extract_frames.py`) |
| **Fish-eye / 360° cameras** | Strong distortion; people near the edges are rotated | Train on de-warped views (the vendor software can export them) or keep fish-eye cameras in a separate experiment |
| **Crowds** on weekends and holidays, and at events | More people than `QUERY_NUM`; overlapping boxes | `QUERY_NUM: 25–30`; annotate at least some crowded scenes |
| **Park-specific activities**: electric scooters and bikes (very common), children on playgrounds, dog walking, picnics and barbecues (*mangal*), exercise equipment, ball games | Some have no AVA counterpart | Add classes deliberately (e.g. `walk_dog`, `exercise`, `play_ball`); they start from random init or a loose AVA mapping |
| **Rare events of real interest** (falls, fights, vandalism) | Few examples, so AP is noisy and unstable | See [03 §3.7](03_data_and_annotation.md#37-how-much-data): staging, external datasets, over-sampling. Report these separately from the frequent classes |

## 6.2 Privacy, law and ethics: read before collecting data

> This is general orientation for a student project, not legal advice. Confirm with your course
> staff, the Technion's research-ethics process, and whoever owns the cameras (e.g. the municipality).

**The legal frame in Israel.** The **Privacy Protection Law, 5741-1981** (חוק הגנת הפרטיות, התשמ"א-1981)
applies to video of identifiable people. Its **Amendment 13**, in force since **14 August 2025**, is a major
overhaul in the direction of the GDPR. It modernises definitions, adds governance and transparency duties
for databases, and gives the **Privacy Protection Authority** (הרשות להגנת הפרטיות) much stronger
investigation and fining powers. The Authority has published guidance on surveillance cameras, and more
guidance since the amendment came into force. Footage of people in public parks, used to train a model
that analyses people's behaviour, is personal data, and parts of it (e.g. anything that can reveal
health, or biometric use) may count as more sensitive.

Sources: [Library of Congress summary](https://www.loc.gov/item/global-legal-monitor/2025-11-17/israel-amendment-to-privacy-protection-law-goes-into-effect/),
[IAPP overview](https://iapp.org/news/a/israel-marks-a-new-era-in-privacy-law-amendment-13-ushers-in-sweeping-reform),
[Pearl Cohen](https://www.pearlcohen.com/israel-significant-amendment-to-the-privacy-law-takes-effect/),
[DLA Piper: data protection laws of Israel](https://www.dlapiperdataprotection.com/index.html?t=law&c=IL).
The Authority's own guidance is on gov.il (search for הרשות להגנת הפרטיות מצלמות מעקב).

**Practical rules for this project:**
1. **Get the footage legitimately**, with written permission from the camera operator (municipality or
   park authority) that states the purpose (academic research) and the retention period.
2. **Ask about ethics approval.** Research on data about people usually goes through an ethics committee;
   ask your course staff which Technion process applies.
3. **Data minimisation:** extract only the segments you need, at the resolution you need. Don't
   keep raw archives longer than necessary.
4. **Keep it local.** Frames and annotations stay on an encrypted disk. Use a **self-hosted CVAT**,
   and don't upload footage to cloud annotation services, cloud drives, or LLM chats. `data/`, `runs/` and
   `outputs/` are git-ignored for this reason; **never** `git add -f` them.
5. **Blur faces** (and licence plates) in any figure, slide, video or report you share. The model
   doesn't need faces to learn "walk".
6. **No identification.** This project detects actions, not identities. Don't add face recognition,
   re-identification across cameras, or tracking of specific individuals.
7. **Think about misuse and bias.** An automatic "suspicious behaviour" detector in public spaces can
   harm people, for example if "loitering" or "running" gets flagged disproportionately for certain
   groups. Frame the classes neutrally, report error rates, and keep a human in the loop.
8. **Retention and deletion:** agree when the data will be deleted (e.g. at the end of the semester) and do it.

## 6.3 A suggested project plan (≈ one semester)

| Week | Milestone |
|---|---|
| 1–2 | Read [01](01_tuber_explained.md) and the paper; run the smoke tests and the zero-shot demo; get data access and permissions |
| 3–4 | Finalise the label set; set up CVAT; annotate a pilot of ~20 minutes from 2–3 cameras; build, preview, and get the zero-shot baseline numbers |
| 5–7 | Annotate more (aim for diversity across cameras, times and days); first fine-tuning runs; error analysis on videos |
| 8–10 | Experiments ([04 §4.6](04_finetuning_guide.md#46-experiments-worth-running-good-for-a-course-project-report)): resolution, freezing, head init, day vs. night |
| 11–12 | Optional comparison with a two-stage baseline (MMAction2) or STMixer; final test on held-out cameras |
| 13 | Report: method, data statement (incl. privacy measures), results, failure cases (with blurred faces) |
