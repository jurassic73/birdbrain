# Hailo-8 shadow detector: offline scorecard

Generated 2026-11-08T19:24:15 over 487 stored photos (281 real visits, 206 false triggers that still have a photo).

COCO has no squirrel, raccoon or possum class. "Flagged" means the best detection of any animal-like COCO class (bear, bird, cat, cow, dog, elephant, giraffe, horse, sheep, teddy_bear, zebra) reached the threshold; the class itself is reported separately below. Real visits use the owner's corrected labels where there are any.

Views: **frame** is the whole photo shrunk to the substream's width, the geometry the live detector sees with HAILO_VIEW=frame. **subcrop** is the crop box cut from that small frame and upscaled to the detector's input, what the live detector sees with HAILO_VIEW=crop (the default). Both are representative of the live service. **crop** is the stored crop box (what Claude sees) at full resolution, which shows what a detector on the main stream's crop could do.

## yolov8s_h8

Inference: 10.1 ms per view on the Hailo-8 (scoring took 23 s).

### View: frame

Real visits: 281 (bird 210, squirrel 71; 0 at night). False triggers: 206 (90 day, 116 night).

| threshold | recall | bird | squirrel | false ignored | day | night | precision |
|---|---|---|---|---|---|---|---|
| 0.2 | 23% | 14% | 52% | 99% | 88/90 | 116/116 | 97% |
| 0.3 | 21% | 11% | 48% | 100% | 89/90 | 116/116 | 98% |
| 0.4 | 17% | 10% | 39% | 100% | 89/90 | 116/116 | 98% |
| 0.5 | 16% | 9% | 35% | 100% | 90/90 | 116/116 | 100% |
| 0.6 | 12% | 8% | 23% | 100% | 90/90 | 116/116 | 100% |
| 0.7 | 10% | 7% | 20% | 100% | 90/90 | 116/116 | 100% |

What COCO called them (best animal-like class at the 0.2 floor, or nothing):

- bird: nothing 181, bird 26, cat 3
- squirrel: nothing 34, bird 23, cat 9, dog 4, teddy_bear 1
- false triggers: nothing 204, cat 1, bird 1

Best animal-like score per photo (0 = nothing found):

- real: n 281, min 0.0, q1 0.0, median 0.0, q3 0.0, max 0.92
- false: n 206, min 0.0, q1 0.0, median 0.0, q3 0.0, max 0.44
- false_flagged: n 2, min 0.24, q1 0.18, median 0.34, q3 0.5, max 0.44

### View: subcrop

Real visits: 281 (bird 210, squirrel 71; 0 at night). False triggers: 206 (90 day, 116 night).

| threshold | recall | bird | squirrel | false ignored | day | night | precision |
|---|---|---|---|---|---|---|---|
| 0.2 | 42% | 39% | 51% | 99% | 89/90 | 114/116 | 98% |
| 0.3 | 35% | 34% | 38% | 100% | 89/90 | 116/116 | 99% |
| 0.4 | 29% | 30% | 27% | 100% | 89/90 | 116/116 | 99% |
| 0.5 | 24% | 26% | 18% | 100% | 89/90 | 116/116 | 99% |
| 0.6 | 19% | 22% | 10% | 100% | 89/90 | 116/116 | 98% |
| 0.7 | 14% | 17% | 6% | 100% | 90/90 | 116/116 | 100% |

What COCO called them (best animal-like class at the 0.2 floor, or nothing):

- bird: nothing 128, bird 80, dog 1, cat 1
- squirrel: nothing 35, bird 17, cat 8, dog 7, bear 3, teddy_bear 1
- false triggers: nothing 203, cat 1, bird 1, dog 1

Best animal-like score per photo (0 = nothing found):

- real: n 281, min 0.0, q1 0.0, median 0.0, q3 0.49, max 0.92
- false: n 206, min 0.0, q1 0.0, median 0.0, q3 0.0, max 0.69
- false_flagged: n 3, min 0.23, q1 0.23, median 0.25, q3 0.69, max 0.69

### View: crop

Real visits: 281 (bird 210, squirrel 71; 0 at night). False triggers: 206 (90 day, 116 night).

| threshold | recall | bird | squirrel | false ignored | day | night | precision |
|---|---|---|---|---|---|---|---|
| 0.2 | 48% | 46% | 54% | 95% | 84/90 | 112/116 | 93% |
| 0.3 | 43% | 42% | 46% | 99% | 88/90 | 115/116 | 98% |
| 0.4 | 37% | 36% | 39% | 99% | 89/90 | 115/116 | 98% |
| 0.5 | 31% | 32% | 30% | 100% | 89/90 | 116/116 | 99% |
| 0.6 | 27% | 28% | 25% | 100% | 89/90 | 116/116 | 99% |
| 0.7 | 22% | 24% | 15% | 100% | 90/90 | 116/116 | 100% |

What COCO called them (best animal-like class at the 0.2 floor, or nothing):

- bird: nothing 113, bird 96, dog 1
- squirrel: nothing 33, bird 26, cat 8, dog 4
- false triggers: nothing 196, cat 5, bird 4, dog 1

Best animal-like score per photo (0 = nothing found):

- real: n 281, min 0.0, q1 0.0, median 0.0, q3 0.63, max 0.94
- false: n 206, min 0.0, q1 0.0, median 0.0, q3 0.0, max 0.66
- false_flagged: n 10, min 0.21, q1 0.22, median 0.23, q3 0.39, max 0.66

## yolov6n_h8

Inference: 5.2 ms per view on the Hailo-8 (scoring took 16 s).

### View: frame

Real visits: 281 (bird 210, squirrel 71; 0 at night). False triggers: 206 (90 day, 116 night).

| threshold | recall | bird | squirrel | false ignored | day | night | precision |
|---|---|---|---|---|---|---|---|
| 0.2 | 16% | 10% | 35% | 100% | 90/90 | 115/116 | 98% |
| 0.3 | 13% | 9% | 25% | 100% | 90/90 | 116/116 | 100% |
| 0.4 | 10% | 7% | 18% | 100% | 90/90 | 116/116 | 100% |
| 0.5 | 7% | 6% | 13% | 100% | 90/90 | 116/116 | 100% |
| 0.6 | 6% | 6% | 8% | 100% | 90/90 | 116/116 | 100% |
| 0.7 | 4% | 4% | 4% | 100% | 90/90 | 116/116 | 100% |

What COCO called them (best animal-like class at the 0.2 floor, or nothing):

- bird: nothing 189, bird 18, cat 3
- squirrel: nothing 46, bird 16, cat 9
- false triggers: nothing 205, cat 1

Best animal-like score per photo (0 = nothing found):

- real: n 281, min 0.0, q1 0.0, median 0.0, q3 0.0, max 0.86
- false: n 206, min 0.0, q1 0.0, median 0.0, q3 0.0, max 0.26
- false_flagged: n 1, min 0.26, q1 0.26, median 0.26, q3 0.26, max 0.26

### View: subcrop

Real visits: 281 (bird 210, squirrel 71; 0 at night). False triggers: 206 (90 day, 116 night).

| threshold | recall | bird | squirrel | false ignored | day | night | precision |
|---|---|---|---|---|---|---|---|
| 0.2 | 36% | 33% | 42% | 98% | 87/90 | 114/116 | 95% |
| 0.3 | 26% | 25% | 27% | 99% | 89/90 | 115/116 | 97% |
| 0.4 | 21% | 22% | 17% | 100% | 90/90 | 116/116 | 100% |
| 0.5 | 16% | 18% | 13% | 100% | 90/90 | 116/116 | 100% |
| 0.6 | 11% | 14% | 4% | 100% | 90/90 | 116/116 | 100% |
| 0.7 | 7% | 9% | 0% | 100% | 90/90 | 116/116 | 100% |

What COCO called them (best animal-like class at the 0.2 floor, or nothing):

- bird: nothing 140, bird 68, cat 2
- squirrel: nothing 41, cat 13, bird 9, dog 6, bear 2
- false triggers: nothing 201, bird 4, dog 1

Best animal-like score per photo (0 = nothing found):

- real: n 281, min 0.0, q1 0.0, median 0.0, q3 0.31, max 0.94
- false: n 206, min 0.0, q1 0.0, median 0.0, q3 0.0, max 0.37
- false_flagged: n 5, min 0.23, q1 0.23, median 0.23, q3 0.34, max 0.37

### View: crop

Real visits: 281 (bird 210, squirrel 71; 0 at night). False triggers: 206 (90 day, 116 night).

| threshold | recall | bird | squirrel | false ignored | day | night | precision |
|---|---|---|---|---|---|---|---|
| 0.2 | 43% | 40% | 55% | 96% | 82/90 | 115/116 | 93% |
| 0.3 | 33% | 29% | 46% | 98% | 85/90 | 116/116 | 95% |
| 0.4 | 21% | 19% | 28% | 99% | 87/90 | 116/116 | 95% |
| 0.5 | 16% | 16% | 14% | 100% | 89/90 | 116/116 | 98% |
| 0.6 | 12% | 12% | 11% | 100% | 90/90 | 116/116 | 100% |
| 0.7 | 7% | 8% | 4% | 100% | 90/90 | 116/116 | 100% |

What COCO called them (best animal-like class at the 0.2 floor, or nothing):

- bird: nothing 127, bird 81, cat 1, dog 1
- squirrel: nothing 32, bird 20, cat 12, dog 7
- false triggers: nothing 197, bird 7, dog 1, sheep 1

Best animal-like score per photo (0 = nothing found):

- real: n 281, min 0.0, q1 0.0, median 0.0, q3 0.37, max 0.93
- false: n 206, min 0.0, q1 0.0, median 0.0, q3 0.0, max 0.59
- false_flagged: n 9, min 0.21, q1 0.28, median 0.34, q3 0.44, max 0.59
