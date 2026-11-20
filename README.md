# Handwritten Medicine-Name Recognition in Bangladeshi Prescriptions

This repository accompanies a study on recognizing handwritten medicine names in Bangladeshi prescriptions. The primary dataset contains 317 medicine classes. The study evaluates seven classifiers on an external set of 250 images from 25 overlapping classes, with 10 images per class.

## Models and experimental configuration

The seven models are VGG16, VGG19, DenseNet121, MobileNetV2, EfficientNet-B0, CNN–BiLSTM fusion, and CNN attention fusion. Each model is trained independently with seeds `42`, `123`, and `456`, mapped to `run_1`, `run_2`, and `run_3`, respectively. Both external protocols evaluate the same 21 model–seed checkpoints.

## Data availability and privacy

The study distinguishes samples derived from prescriptions from handwriting contributed under controlled conditions. Raw prescription images and personally identifiable information are not publicly shared. The primary dataset and external test images are not publicly included in this repository. **The datasets are available from the study authors upon reasonable request, subject to applicable privacy and ethical restrictions.**

## Repository structure

| Notebook or directory | Purpose |
| --- | --- |
| [`Training/baseline_models_multirun.ipynb`](Training/baseline_models_multirun.ipynb) | Trains ImageNet-initialized VGG16, VGG19, DenseNet121, MobileNetV2, and EfficientNet-B0 classifiers. |
| [`Training/bilstm_multirun.ipynb`](Training/bilstm_multirun.ipynb) | Concatenates MobileNetV2 and EfficientNet-B0 feature vectors and feeds the combined vector to a bidirectional LSTM classifier. |
| [`Training/attention_multirun.ipynb`](Training/attention_multirun.ipynb) | Learns attention weights to combine MobileNetV2 and EfficientNet-B0 features before classification. |
| [`Cross Dataset Prep/cross_test_dataset.ipynb`](Cross%20Dataset%20Prep/cross_test_dataset.ipynb) | Matches labeled external images against primary class names and copies matched images into class folders. |
| [`Cross Testing/cross-dataset-testing-restricted-25-class.ipynb`](Cross%20Testing/cross-dataset-testing-restricted-25-class.ipynb) | Evaluates predictions ranked over the 25 shared classes. |
| [`Cross Testing/cross-dataset-testing-unrestricted-317-class.ipynb`](Cross%20Testing/cross-dataset-testing-unrestricted-317-class.ipynb) | Evaluates predictions ranked over the original 317-class vocabulary. |
| [`CrossDataset_3Seed_RESULTS/`](CrossDataset_3Seed_RESULTS/) | Saved external evaluation files, grouped under [`restricted/`](CrossDataset_3Seed_RESULTS/restricted/) and [`unrestricted/verified_3seed/`](CrossDataset_3Seed_RESULTS/unrestricted/verified_3seed/). Each protocol contains model folders with `seed_42/`, `seed_123/`, and `seed_456/` subfolders. |

## Environment

The notebook metadata records Python 3.10.8. Run the notebooks in Jupyter with a Python kernel and install a compatible pair of `torch` and `torchvision` builds for your CPU or CUDA environment. The notebooks select CUDA when available and otherwise use CPU. The training backbones request ImageNet pretrained weights, so training needs internet access for the initial download or an existing local weight cache.

In the same environment, install the remaining dependencies.

```sh
python -m pip install jupyterlab ipykernel numpy pandas pillow scikit-learn matplotlib seaborn tqdm openpyxl
```

`openpyxl` supports the training notebooks' Excel exports. The repository does not contain a pinned environment file. For the BiLSTM notebook, install `openpyxl` before running it and skip the standalone `pip install openpyxl` cell if the kernel treats it as Python code.

## Running the notebooks

Open local working copies in Jupyter, select the environment above, and update the hard-coded paths for your machine. Run cells in order and direct generated outputs to a separate writable location. For training, use the three training notebooks. For external evaluation, use the preparation notebook or obtain the prepared set, then run either protocol with its required checkpoints and metadata.

### Training workflow

1. Point `DATASET_PATH` at the original `Full_Dataset_Split`. Each of `Train/`, `Val/`, and `Test/` uses one subdirectory per class. Keep class names and the resulting `ImageFolder` class order consistent across splits and with the checkpoints.
2. Run the training notebooks. The baseline notebook resizes images to 224 × 224. The fusion notebooks use the supplied image dimensions, which must be compatible within a batch. Training applies augmentation and ImageNet normalization.
3. Each run is configured for 25 epochs. The notebooks select the checkpoint with the best validation accuracy for internal test evaluation and export training summaries. Keep the best checkpoints and `run_complete.json` files for external evaluation.

### External test preparation

Configure the preparation notebook with `testing_labels.csv`, the `testing_words/` image directory, a primary class directory, and an output directory for `Cross_Test_Final/`. The notebook reads the CSV's `IMAGE` and `MEDICINE_NAME` fields, matches medicine names case-insensitively, and copies matched images into the corresponding class folders. Both evaluation notebooks validate the exact 25 configured class names and require 10 images per class.

### External evaluation setup

Configure the primary `Full_Dataset_Split/Train/` directory, prepared external set, checkpoint archive, and output location. The restricted notebook also requires `MODEL_ROOT` to exist. Both notebooks rebuild the 317-class index from the primary training folders, validate the recorded seeds and checkpoint compatibility for all 21 model–seed pairs, and apply 224 × 224 resizing and ImageNet normalization to external images.

Under `ARCHIVE_ROOT`, supply the following checkpoints and a matching `run_complete.json` in each group's `run_1/`, `run_2/`, and `run_3/` directories. Use trusted checkpoints because the notebooks load them with `weights_only=False`.

| Training output directory | Best checkpoint files |
| --- | --- |
| `multirun_baseline/` | `best_vgg16.pth`, `best_vgg19.pth`, `best_densenet121.pth`, `best_mobilenetv2.pth`, `best_efficientnetb0.pth` |
| `multirun_bilstm/` | `best_bilstm_317.pth` |
| `multirun_attention/` | `best_attention_317.pth` |

## Restricted 25-class evaluation

Selects the 25 shared logits from the model's 317 outputs before ranking predictions for top-1 and top-5 accuracy. Each confusion matrix has 25 true-class rows and 25 predicted-class columns.

## Unrestricted 317-class evaluation

Ranks all 317 logits for top-1 and top-5 accuracy. Top-1 predictions among the 292 classes outside the overlap count as errors, and the notebook reports their count and rate. Macro and weighted class metrics use the 25 external labels. Each confusion matrix has 25 true-class rows and 317 predicted-class columns.

## Saved results and artifacts

Each protocol includes a predictions CSV, classification reports in CSV and JSON, and confusion matrices in CSV and PNG for every model and seed. The per-seed CSV and JSON files provide quantitative evidence across the three runs. Each PNG visualizes one model and one seed. The model folders also contain element-wise sum and mean confusion matrices across the three seeds. The restricted aggregate filenames use the suffix `available_runs`.

The evaluation notebooks can generate the following protocol-level tables, including per-model means and sample standard deviations across seeds. These files are absent from the repository.

| Protocol | Generated CSVs |
| --- | --- |
| Restricted | `completed_runs_progressive.csv`, `all_available_runs_results.csv`, `summary_available_runs.csv` |
| Unrestricted | `completed_runs_progressive.csv`, `all_models_all_seeds_results.csv`, `summary_mean_sample_sd_3seeds.csv` |

## Reproducibility and required inputs

The saved external evaluation artifacts can be inspected without the original images or checkpoints. Reproducing the original training and inference experiments requires access to the corresponding data and checkpoint files. The repository omits the original `Full_Dataset_Split`, exact split indices, external source labels and images, prepared external test dataset, trained checkpoints, and training run metadata and exports.


