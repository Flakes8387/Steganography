# Dataset Verification Feature

## Overview

Before training begins, the training script now automatically verifies the dataset by loading one batch and checking its integrity. This prevents silent bugs and ensures data is correctly formatted.

## What Gets Verified

### 1. Batch Shapes
```
Batch Shapes:
  Images:   [4, 3, 128, 128] (batch, channels, height, width)
  Messages: [4, 32] (batch, message_length)
```

### 2. Tensor Properties
```
Tensor Properties:
  Image dtype:   torch.float32
  Image range:   [0.000, 1.000]
  Message dtype: torch.float32
  Message range: [0, 1]
```

### 3. First Sample Display
```
First Sample in Batch:
  Image shape: [3, 128, 128]
  Image stats: mean=0.523, std=0.287
  Message bits: 10110101101001110010110110010100 (32 bits total)
  Saved sample image: checkpoints/dataset_sample.png
```

### 4. Validation Checks
```
Validation Checks:
  [OK] Images normalized to [0, 1]
  [OK] Messages are binary (0/1)
  [OK] Batch size correct: 4
  [OK] Image dimensions correct: 128x128
  [OK] Message length correct: 32 bits

[SUCCESS] All dataset verification checks passed!
```

## When It Runs

The verification runs automatically:
- **After** dataset is loaded
- **Before** model training begins
- Takes ~1 second

## Example Output

```
============================================================
DATASET VERIFICATION
============================================================

Loading one sample batch to verify dataset integrity...

Batch Shapes:
  Images:   [4, 3, 128, 128] (batch, channels, height, width)
  Messages: [4, 32] (batch, message_length)

Tensor Properties:
  Image dtype:   torch.float32
  Image range:   [0.024, 0.982]
  Message dtype: torch.float32
  Message range: [0, 1]

First Sample in Batch:
  Image shape: [3, 128, 128]
  Image stats: mean=0.507, std=0.291
  Message bits: 10010101011001110110101011001011 (32 bits total)
  Saved sample image: checkpoints/dataset_sample.png

Validation Checks:
  [OK] Images normalized to [0, 1]
  [OK] Messages are binary (0/1)
  [OK] Batch size correct: 4
  [OK] Image dimensions correct: 128x128
  [OK] Message length correct: 32 bits

[SUCCESS] All dataset verification checks passed!
============================================================
```

## Error Detection

### Example 1: Images Not Normalized
```
Validation Checks:
  [WARNING] Images not in [0, 1] range
  [OK] Messages are binary (0/1)
  ...

[WARNING] Some verification checks failed - review warnings above

Continue training anyway? (y/n):
```

### Example 2: Wrong Image Size
```
Validation Checks:
  [OK] Images normalized to [0, 1]
  [OK] Messages are binary (0/1)
  [OK] Batch size correct: 4
  [ERROR] Image dimensions 256x256 != 128x128  # MISMATCH!
  [OK] Message length correct: 32 bits

[WARNING] Some verification checks failed - review warnings above

Continue training anyway? (y/n):
```

### Example 3: Non-Binary Messages
```
Tensor Properties:
  ...
  Message range: [0, 2]  # Should be [0, 1]

Validation Checks:
  [OK] Images normalized to [0, 1]
  [WARNING] Messages contain non-binary values: [0.0, 1.0, 2.0]
  ...

[WARNING] Some verification checks failed - review warnings above
```

### Example 4: Dataset Loading Failure
```
[ERROR] ERROR during dataset verification: No images found in directory
   This indicates a problem with the dataset or data loader.

Traceback (most recent call last):
  ...

Aborting training due to dataset verification failure.
```

## Benefits

1. **Early Error Detection**: Catches dataset issues before wasting GPU time
2. **Visual Confirmation**: Sample image saved for manual inspection
3. **Shape Verification**: Ensures tensors have correct dimensions
4. **Value Range Check**: Confirms normalization is correct
5. **Binary Message Validation**: Verifies messages are 0/1 values

## Saved Sample Image

A sample from the first batch is saved to:
```
checkpoints/dataset_sample.png
```

You can open this to visually confirm:
- Image quality
- Resolution
- Color channels
- Overall appearance

## Implementation Details

### Code Location
- File: `train.py`
- Section: After dataset loading (lines 730-832)
- Runs: Once before training starts

### Checks Performed

| Check | Purpose | Abort on Fail |
|-------|---------|---------------|
| Image range [0,1] | Normalization correct | No (warning) |
| Messages binary | Values are 0/1 | No (warning) |
| Batch size | Matches config | No (info) |
| Image dimensions | Match config | Yes (error) |
| Message length | Matches config | Yes (error) |

### User Interaction

**If all checks pass:**
- Training continues automatically
- No user interaction required

**If warnings appear:**
- User is prompted: "Continue training anyway? (y/n)"
- Can choose to proceed or abort

**If critical errors:**
- Training aborts immediately
- Clear error message displayed

## Example Use Cases

### Case 1: New Dataset
```bash
python train.py --train_dir my_new_dataset --config config.yaml
```
- Verification runs automatically
- Confirms dataset is compatible
- Shows sample image for manual check

### Case 2: Debugging
If training behaves oddly:
1. Check verification output from previous run
2. Look at saved sample image
3. Verify tensor shapes and ranges

### Case 3: DIV2K Training
```bash
python train.py --config config_div2k_balanced.yaml
```
Output shows:
- 500 base images loaded
- 2,000 total patches (500 × 4)
- Patch size: 128×128
- Sample patch saved

## Tips

1. **Always check the sample image** in `checkpoints/dataset_sample.png`
   - Ensures images loaded correctly
   - Verifies patch extraction works

2. **Pay attention to warnings**
   - May indicate subtle bugs
   - Could affect training quality

3. **Verify shapes match expectations**
   - Batch size should match config
   - Image size should be correct
   - Message length should match config

4. **Check value ranges**
   - Images: [0, 1] (normalized)
   - Messages: [0, 1] (binary)

## Disable Verification

Currently verification cannot be disabled - it's essential for catching bugs. It adds <1 second overhead which is negligible compared to training time.

## Related

- See [DIV2K_REQUIREMENTS.md](DIV2K_REQUIREMENTS.md) for DIV2K-specific validation
- See [DATASET_LOADER_GUIDE.md](DATASET_LOADER_GUIDE.md) for dataset API
- See [train.py](train.py) lines 730-832 for implementation
