# BinSight data cleaning report

- candidates: 8811, kept: 4419, removed: 4392
- seed 42; Kaggle sample 1500; CLIP threshold 0.65; pHash distance <= 4

## Removed (reason x source)

| | kaggle | taco | trashnet | All |
|---|---|---|---|---|
| clip_not_food | 398 | 0 | 0 | 398 |
| duplicate_of | 3 | 0 | 28 | 31 |
| grayscale | 16 | 0 | 0 | 16 |
| image_missing | 0 | 2545 | 0 | 2545 |
| label_dropped | 0 | 1025 | 0 | 1025 |
| too_small | 5 | 371 | 0 | 376 |
| unreadable | 0 | 1 | 0 | 1 |
| All | 422 | 3942 | 28 | 4392 |

## Kept (source x bin)

| | blue | garbage | green | All |
|---|---|---|---|---|
| kaggle | 0 | 0 | 1078 | 1078 |
| taco | 648 | 181 | 13 | 842 |
| trashnet | 2362 | 137 | 0 | 2499 |
| All | 3010 | 318 | 1091 | 4419 |

## Split x bin

| | blue | garbage | green | All |
|---|---|---|---|---|
| test | 451 | 44 | 165 | 660 |
| train | 2108 | 227 | 762 | 3097 |
| val | 451 | 47 | 164 | 662 |
| All | 3010 | 318 | 1091 | 4419 |

## Split x material

| | cardboard | glass | metal | organic | other | paper | plastic | All |
|---|---|---|---|---|---|---|---|---|
| test | 74 | 88 | 82 | 163 | 23 | 78 | 152 | 660 |
| train | 326 | 352 | 365 | 756 | 115 | 471 | 712 | 3097 |
| val | 74 | 89 | 73 | 162 | 23 | 97 | 144 | 662 |
| All | 474 | 529 | 520 | 1081 | 161 | 646 | 1008 | 4419 |

## Review sheets

Open `review/*.png` and spot-check each sheet; record the error rate you observe here.
