# GROUP-6-GET-324-LAB

 CONTRIBUTORS:
 
#
## 22/EG/PE/1485
## 23/EG/PE/055
## 22/EG/PE/1545
## 22/EG/PE/1505
## 22/EG/PE/1525
## 23/EG/PE/005
## 22/EG/PE/1475

# Potato Leaf Classifier — Healthy vs Early Blight

Binary image classifier distinguishing **Healthy** potato leaves from **Potato Early Blight**,
built as a CNN/transfer-learning comparison project and deployable as a Streamlit app on
Hugging Face Spaces.

**Live demo:** `https://group-6-get-324-lab-8.streamlit.app/`

---



- **This is the easy binary cut, not the real problem.** The PlantVillage potato subset has
  three classes — Healthy, Early Blight, Late Blight. This model was trained on only two,
  dropping the genuinely hard confusion pair (Early vs Late Blight, which look far more
  alike than either does to a healthy leaf). Expect high accuracy — that reflects the task's
  easiness, not model quality. If asked, say this plainly rather than let the number imply
  more than it does.
- **No dedicated test set — `val/` was used instead.** This dataset ships as `train/` +
  `val/` only, with no separate `test/` folder. The notebook uses `val/` as the held-out
  evaluation set. If anyone asks how this was evaluated, call it what it is — the dataset's
  validation split, not an independent test set someone else curated.
- **Leakage check was run, not assumed.** PlantVillage has documented near-duplicate images
  across splits. The notebook runs a perceptual-hash duplicate check between train and the
  held-out set before reporting any accuracy — see the leakage rate in Section 3 of the
  notebook, and quote it alongside the accuracy number, not instead of it.
- **Lab-background bias.** These are controlled-background lab photos, not field photos. A
  model trained here is not validated for real phone-camera field photos with dirt,
  shadows, or other leaves in frame. Don't present this as field-deployment-ready without
  new field data.

---

## Dataset

- **Source:** [PlantVillage (mohitsingh1804)](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)
- **Full dataset:** 38 crop/disease classes across many crops (tomato, apple, corn, grape, etc.)
- **Subset used:** `Potato___healthy` and `Potato___Early_blight` only, from the `train/` and
  `val/` splits (folder names use underscores, not the "Healthy Potato" style shown in some
  dataset descriptions — see the notebook's class-folder resolution logic).
- **Class balance:** `<FILL IN: actual train/val counts per class from the notebook's EDA section>`
- **Leakage rate found:** `<FILL IN: the percentage reported by the notebook's Section 3 duplicate check>`

## Models compared

| Model | Type | Notes |
|---|---|---|
| Custom CNN | Baseline, trained from scratch | 4-block Conv2D/BatchNorm/MaxPool stack, GAP head |
| EfficientNetV2-B0 | Transfer learning | ImageNet-pretrained, head-trained then fine-tuned (top ~30 layers unfrozen) |

The model used by the deployed app is `potato_blight_best_model.keras` — whichever of the
two scored higher on held-out ROC-AUC, not necessarily higher raw accuracy (see notebook
Section 8 for why accuracy alone was not used to pick the winner).

## Evaluation

Metrics computed on the held-out (`val/`) set: accuracy, precision, recall, F1, ROC-AUC, PR-AUC.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Custom CNN | `0.380952` | `1.0` | `0.285` | `0.443580` | `0.982097` | `0.997189` |
| EfficientNetV2-B0 | `0.900433` | `1.0` | `0.885` | `0.938992` | `0.999355` | `0.999902` |

*Fill this in from the notebook's actual Section 8 output before publishing. A README with
invented numbers is worse than no README — don't ship this table half-`TODO`.*

## Explainability

Grad-CAM overlays were generated for the winning model against Early Blight samples, to
check whether it's attending to lesion regions rather than background/lighting artifacts —
see notebook Section 9. Note it as "generated," not "verified," unless you've actually
looked at whether the heatmaps land on the lesions.

## Robustness testing

Held-out accuracy was re-evaluated under synthetic blur, brightness, and contrast
perturbations (notebook Section 11) to estimate sensitivity to non-lab capture conditions.

| Perturbation | Accuracy |
|---|---|
| None (baseline) | `0.900433` |
| Blur | `0.909091` |
| Low brightness | `0.805195` |
| High brightness | `0.939394` |

A large drop under any of these confirms the lab-background-bias caveat above — say so in
your conclusions rather than only reporting the clean-condition number.

---

## Repository structure

```
.
├── potato_blight_cnn.ipynb          # Data pipeline, training, evaluation, explainability, robustness
├── app.py                            # Streamlit inference app (Hugging Face Space entrypoint)
├── requirements.txt                  # Streamlit app dependencies
├── potato_blight_best_model.keras    # Trained model weights (hosted on HF, downloaded at runtime)
└── README.md                          # This file
```

## Running the notebook

Designed for Google Colab. Requires a Kaggle API token stored in Colab Secrets
(`KAGGLE_USERNAME`, `KAGGLE_KEY`) — see the notebook's data-download cell for setup
instructions. Run top to bottom after a fresh Runtime → Restart runtime; don't hand-patch
individual cells out of order, it corrupts notebook state.

## Running the Streamlit app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Before running: set `HF_REPO_ID` in `app.py` to your real Hugging Face repo (it ships with
a placeholder value and will fail loudly, not silently, if left unchanged) and confirm
`MODEL_FILENAME` matches the exact filename in that repo's Files tab.

## Deploying to Hugging Face Spaces

1. Create a Space (SDK: Streamlit).
2. Push `app.py` and `requirements.txt` to the Space repo.
3. Upload `potato_blight_best_model.keras` to the model repo referenced by `HF_REPO_ID`.
4. Confirm `MODEL_FILENAME` in `app.py` matches exactly what's in that repo's Files tab —
   the single most common deployment failure is a filename mismatch here.


