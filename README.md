# spot-follow — Spot looks for a target and follows it

Day 2 project for the Spot training program. Spot scans its surroundings with all five
body cameras, finds a target (any person, one specific person, or Charlie the Charger),
and walks after it while keeping a safe following distance. If it loses the target it
heads to where it was last seen, and if that fails it turns in place to look around.

```
python scripts/run_follow.py --dry-run --show      # watch what Spot sees, nothing moves
python scripts/run_follow.py --show                # follow the nearest person
python scripts/run_follow.py --mode face --show    # follow one enrolled person only
python scripts/run_follow.py --mode open_vocab --show   # Charlie, zero-shot
```

---

## 1. How it works

```
 5 body cameras ──► make upright ──► detector ──► 2D boxes
 (grey + depth)                    (YOLO / face)       │
                                                       ▼
            depth image (or box height) ──► range ──► 3D point in camera frame
                                                       │  transform snapshot
                                                       ▼  (sent with each image)
                                         target position in ODOM frame
                                                       │
                                     single-target tracker (gating, smoothing,
                                     confirmation, face lock, camera fusion)
                                                       │
            SEARCH ◄──────────► FOLLOW ◄─────────► REACQUIRE
          turn in place        walk to a point      walk to where it
          + pause to look      `standoff_m` short   was heading
                               of target, facing it
                                                       │
                              synchro_se2_trajectory_point_command (odom frame,
                              velocity limits, obstacle avoidance, 1 s expiry)
```

**Perception.** Each cycle (5 Hz by default) the program requests the visual and the
depth-aligned-to-visual image from every configured camera in one RPC. Spot's front
cameras are mounted sideways and the right camera upside-down, so each frame is rotated
upright (angles from Boston Dynamics' own examples) before detection; results are mapped
back to the native pixel grid with the inverse transform.

**Range.** For each box, the program samples Spot's stereo depth over the target's torso
and takes a low percentile (target, not background). If depth is missing (too far, low
texture, glass), it falls back to `range = focal × assumed_height / box_height`.

**3D position.** The torso pixel is un-projected through the camera's real intrinsics
(pinhole, Brown–Conrady and Kannala–Brandt models are all handled) and transformed into
the **odom** frame using the `transforms_snapshot` that comes with that very image. Working
in odom (a world-fixed frame) is the key design choice: the target's position stays
continuous as it passes from one camera to another and as Spot itself walks and turns.

**Tracking.** One target is tracked. A new target must be seen in `confirm_frames` cycles
before Spot moves (kills one-frame false positives). Detections of the same thing seen by
two overlapping cameras are fused. Each cycle the detection closest to the predicted
position (within `gate_distance_m`) wins, with a small bonus for a similar appearance and a
large bonus for a matching face.

| mode | who gets followed | needs |
|---|---|---|
| `person` | the nearest person, then stays locked on them | nothing |
| `face` | **only** the enrolled person; acquires by face, keeps following from behind; drops the lock if a clearly different face shows up 3× | `enroll_target.py` |
| `open_vocab` | whatever the text prompts describe (YOLO-World), e.g. "horse mascot costume" | internet once for weights |
| `custom` | your trained YOLO class (Charlie) | ~150–300 labelled images |

**Control.** In FOLLOW, Spot is sent to a point `standoff_m` (1.5 m) short of the target on
the line between them, facing it. Inside `standoff ± deadband` it only turns to face. Every
command expires after `command_duration_s` (1 s) and is re-sent each cycle — **if the program
freezes, crashes or loses WiFi, Spot simply stops and stands**. Goals are clipped to a
`leash_radius_m` circle around the start point, velocities are capped, and Spot's own
obstacle avoidance stays on.

---

## 2. Hardware and network

* Spot with the 5 standard body cameras (any Spot software 3.x+; the arm is not needed).
* A laptop (Linux, macOS, or Windows) with Python 3.9–3.12. A GPU helps but isn't required:
  `yolov8n` on a modern laptop CPU runs ~15–25 ms per camera image, enough for ~3–5 Hz on
  all five cameras. Use `--cameras frontleft frontright` if it's slow.
* Network: join Spot's WiFi access point (robot at `192.168.80.3`), or the lab network if
  Spot is configured as a client. Spot's AP has no internet — **install and download
  everything beforehand.**
* Robot credentials (the user/password on the robot's sticker or set by your admin).
* The Spot tablet, kept open as the **E-Stop** for the whole session, with a person
  assigned to hold it.

---

## 3. One-time setup (do this tonight / on a network with internet)

```bash
git clone <this repo> spot-follow && cd spot-follow
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
#   (NVIDIA GPU: install the CUDA build of torch first, from pytorch.org)

python scripts/download_models.py                # yolov8n.pt + face models
python scripts/download_models.py --open-vocab   # + YOLO-World for Charlie (optional)
python -m pytest -q                              # 16 tests, no robot needed
```

Set credentials once per terminal so you aren't prompted every run:

```bash
export BOSDYN_CLIENT_USERNAME=user
export BOSDYN_CLIENT_PASSWORD='the-password'
```

Optional: copy `config/default.yaml` to `config/local.yaml`, edit, and pass
`--config config/local.yaml`. Every key is commented in the file.

### Try the detector without the robot

```bash
python scripts/test_detector.py --webcam --grey          # see it like Spot's grey cameras do
```

### Enroll yourself (for `face` mode)

Put 3–10 photos of yourself in `data/enroll/` (different angles, one face each), then:

```bash
python scripts/enroll_target.py --images "data/enroll/*.jpg"
# or capture live:   python scripts/enroll_target.py --webcam   (SPACE = capture, q = done)
python scripts/test_detector.py --webcam --mode face     # green box + id≥0.36 on you only
```

Enrolling from Spot's own camera gives the best match (same lighting and lens):
`python scripts/enroll_target.py --spot --camera frontleft` while standing ~1.5 m in front.
A printed photo works as a target too — hold it at face height.

---

## 4. Running it on the day (in this order)

**0. Area & people.** Clear a space of at least 6 × 6 m, no stairs or drop-offs. One person
holds the tablet with E-Stop ready; everyone else stays outside the leash radius except the
target.

**1. Robot on, tablet connected, then release control on the tablet** (menu → release
control / "Return lease"). Keep the tablet app open: its E-Stop stays active.

**2. Pre-flight check (nothing moves):**
```bash
python scripts/check_robot.py
```
Confirms login, battery, E-Stop, lease owner, all camera + depth sources, and saves
`runs/check/*_upright.jpg`. **Open those images: people and walls must be upright.** If one
camera is sideways, fix its angle in `CAMERA_SOURCES` in `spot_follow/cameras.py`.

**3. Dry run (nothing moves; Spot can be sitting):**
```bash
python scripts/run_follow.py --dry-run --show
```
Walk around Spot. You should see your box in each camera as you pass, state go
SEARCH → FOLLOW, and log lines like `[dry-run] would walk_to 2.31, -0.40, ...`. Check the
reported distance against a tape measure.

**4. Live, slow, short leash:**
```bash
python scripts/run_follow.py --show --max-speed 0.5 --leash 4
```
Spot powers on, stands, and starts looking. Step into view ~3 m away. Walk slowly; it
follows at 1.5 m. Turn around and walk behind it: it turns and finds you.
**Ctrl-C** = stop, sit, power off. **Ctrl-C twice** = cut motor power. Tablet E-Stop works
at all times.

**5. Then open it up:** `--max-speed 0.8 --leash 10`, `--mode face`, `--standoff 2.0`, etc.

### Useful flags

| flag | meaning |
|---|---|
| `--mode person/face/custom/open_vocab` | what to follow |
| `--dry-run` | perceive and plan, never move |
| `--show` / `--save-dir runs/today` | live mosaic window (q quits) / save annotated frames |
| `--cameras frontleft frontright` | fewer cameras = faster loop |
| `--standoff 1.5` `--max-speed 0.8` `--leash 10` | following distance, speed cap, leash radius |
| `--take-lease` | take control away from the tablet without releasing it there |
| `--own-estop` | register this program as the E-Stop (**replaces** the tablet's; then the laptop is your E-Stop) |
| `--stay-standing` | don't sit/power off at exit |
| `--device cuda:0` / `mps` | run the detector on a GPU / Apple Silicon |

---

## 5. Charlie the Charger

**Option A — zero-shot, today.** YOLO-World finds objects from a text description:

```bash
python scripts/test_detector.py --webcam --grey --mode open_vocab \
    --prompts "horse mascot costume" "person in a mascot costume"
python scripts/run_follow.py --mode open_vocab --show \
    --prompts "horse mascot costume"
```

Try a few prompts on photos of Charlie first (`--images`). It works best when Charlie fills a
reasonable part of the frame; it can confuse other costumed people.

**Option B — a trained Charlie detector (best).** A good exercise for a later day:

1. Collect images from Spot's own cameras (upright, greyscale, exactly as the follower sees):
   `python scripts/capture_dataset.py --out data/charlie_raw --seconds 180`
   while Charlie moves around Spot at 1–6 m, all sides, different lighting, partly occluded.
2. Label 150–300 images with one class `charlie` in Roboflow / CVAT / Label Studio, export
   **YOLO format** to `data/charlie/` (with `data.yaml`).
3. Train: `python scripts/train_custom.py --data data/charlie/data.yaml --epochs 60`
   (GPU or Google Colab recommended) → `models/charlie.pt`.
4. Run: `python scripts/run_follow.py --mode custom --custom-model models/charlie.pt --show`

Colour augmentation is disabled in training because Spot's body cameras are greyscale.

---

## 6. Tuning and troubleshooting

| symptom | fix |
|---|---|
| `No E-Stop is configured` | Keep the tablet app open, or run the SDK's `estop_gui` example, or `--own-estop`. |
| `LeaseUseError` / resource already claimed | Release control on the tablet, or `--take-lease`. |
| Login fails | Check `BOSDYN_CLIENT_USERNAME/PASSWORD`; `ping 192.168.80.3`. |
| Never detects anyone | Check `runs/check/*_upright.jpg` are upright; lower `detector.confidence` to 0.35; good lighting. |
| Detects you but Spot doesn't move | Needs `confirm_frames` (3) hits in 2 s; also in the deadband (already 1.5 m away) it only turns. |
| Distance is wrong | Depth fails on dark clothes/far away → height fallback; set `range.assumed_target_height_m` to the target's height. |
| Jerky / slow | Loop warning in the log → fewer cameras, `detector.imgsz: 480`, GPU, `image_quality: 50`. |
| Follows the wrong person | Use `--mode face`, or lower `tracking.gate_distance_m`. |
| `face` mode never locks | Faces are small in Spot's cameras — stand within ~3 m facing Spot to be acquired; enroll from Spot (`--spot`); lower `face.match_threshold` to ~0.30 (more false matches). |
| Stops at the edge | That's the leash; raise `--leash`. |
| Search turns the wrong way | It turns toward the side the target was last seen on. |

The knobs that matter most: `follow.standoff_m`, `follow.max_lin_vel`, `follow.leash_radius_m`,
`tracking.lost_timeout_s`, `detector.confidence`.

---

## 7. Code map (for students)

```
spot_follow/
  robot.py       SDK wrapper: auth, time sync, lease, E-Stop, power, images, walk/turn/stop
  cameras.py     camera sources, upright rotation, intrinsics, pixel→3D, depth sampling
  detectors.py   YOLO, YOLO-World, YuNet+SFace face identity, appearance histogram
  tracker.py     single-target tracker in the odom frame
  controller.py  pure math: target position → standoff goal, leash
  follower.py    the loop: perceive → track → SEARCH/FOLLOW/REACQUIRE → command
  viz.py         camera mosaic with boxes and state
scripts/         run_follow, check_robot, test_detector, enroll_target,
                 capture_dataset, train_custom, download_models
tests/           unit tests + a kinematic simulator that feeds real SDK image protos
                 through the full pipeline (python -m pytest -q)
```

Good extension exercises: follow with the arm camera (`hand_color_image`, colour); a
velocity-based controller instead of trajectory goals; use GraphNav to return home after the
session; replace the greyscale appearance histogram with a ReID network; announce "I see
you!" through Spot CAM's speaker.

---

## 8. Safety notes

* An E-Stop operator watches Spot for the entire run. Software stops are a convenience, not a
  safety system.
* Start with `--max-speed 0.5 --leash 4`. Spot's obstacle avoidance is on but does not see
  glass, thin poles, or small objects reliably.
* Don't let anyone stand between Spot and the target, and don't walk the target backward
  toward Spot.
* Commands expire after 1 s, so a hung program stops the robot; a dead laptop battery or WiFi
  drop has the same effect.
