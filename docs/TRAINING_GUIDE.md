# Training guide (Onkar: Baseline CNN + MobileNetV2)

## 1. Dataset facts that shape everything
| Fact (printed by notebook 01) | Consequence |
|---|---|
| ~1,500 original photos, ~13,600 files; ~89 % are augmented copies (prefixes `rotated_by_N_`, `vertical_flip_`, `translation_`, `saltandpepper_`) | all copies of a photo share a `source_id` and stay in the same split |
| Many originals appear in both Kaggle `train/` and `test/` folders | Kaggle's folders are NOT a valid train/test split; exact duplicates are removed (MD5) |
| Originals are screenshots with the capture time in the file name | neighbouring photos are near-duplicates -> split by time blocks with a gap |

## 2. The split (shared, frozen)
1. per class, sort source photos by capture time; 2. cut into blocks of 25; 3. give whole blocks to
test, validation, train (≈15 / 15 / 70 %); 4. drop photos < 60 s from a photo of another split.
`verify_split` checks: no source/file overlap, all classes in every split, minimum gap ≥ 60 s.
Seed 42 / block 25 / buffer 60 s are fixed **before** training and are not tuned on any result.
Weakness to state honestly: each class has only a few blocks per split, so one unusual capture
session can move a class's score; validation/test sets are small (a few hundred photos).

## 3. Training setup (both models)
* Input: raw RGB 0-255, resized to 224×224; the model contains its own pre-processing.
* Loss: sparse categorical cross-entropy (= categorical cross-entropy with integer labels). Optimiser: Adam.
* Class weights ("balanced") from the training split; random augmentation per batch (flip, rotation, zoom, translation, brightness, contrast).
* Early stopping + ReduceLROnPlateau on validation loss; the epoch with the lowest validation loss is saved.
* Baseline CNN: 4 conv blocks with BatchNorm (momentum 0.9), ~1.2 M parameters, trained from scratch.
* MobileNetV2: stage 1 head only (lr 1e-3, backbone frozen) → stage 2 top-40 backbone layers (lr 2e-5), BatchNorm frozen, backbone in inference mode.

## 4. Experiments (validation only)
One change per experiment, same split and seed; see the table in notebook 04. Rule for the winner
(fixed in advance): highest validation macro-F1; among runs within 0.5 point of it, the lowest
validation loss. Gaps below ~1 point are noise. Report limited-hardware experiments (`alpha_0.75`,
`size_160`) as *accuracy lost vs size/time saved*.

## 5. Hand-off rules
* Give Rucha/Krishna the zip from notebook 05; the **test split is untouched**.
* Test evaluation (Rucha): primary view = one original image per source photo; confidence intervals by resampling source photos.

## 6. Red flags
* Validation accuracy ≥ 99 % → suspect leakage or an easy validation set; look at the confusion matrix.
* Cannot overfit 64 images (notebook 02 check) → labels / pre-processing bug.
* Validation much worse than train and rising loss → overfitting. Both low → underfitting.
* Stage 2 worse than stage 1 → fine-tuning lr too high (`ft_lr_1e-5`).
* Validation metrics look random right after training but fine on training images → BatchNorm statistics not converged (why momentum is 0.9).

## 7. Viva questions to prepare
1. Why can a random split give 99 % accuracy here and why is that misleading?
2. What is `source_id` and what is a time block? Why a 60 s gap?
3. Why two stages, and why is the backbone's BatchNorm kept frozen?
4. What does `verify_finetuning` prove?
5. Why are experiments judged on validation and not on test?
6. What is the cost/benefit of `alpha_0.75` and `size_160` for limited hardware?
7. What are the limitations of one seed and a small validation set?
