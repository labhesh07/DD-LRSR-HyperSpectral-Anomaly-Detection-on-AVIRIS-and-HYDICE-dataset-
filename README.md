# Hyperspectral Anomaly Detection (JSR + LRSR)

Python implementation of the method from:
**"Hyperspectral Anomaly Detection via Background and Potential Anomaly Detection"**  
IEEE: [https://ieeexplore.ieee.org/abstract/document/8519775](https://ieeexplore.ieee.org/abstract/document/8519775)

This repository detects anomalies (for example, airplanes in AVIRIS scenes) from hyperspectral data without supervised training labels. It combines:
- **Joint Sparse Representation (JSR)** for dictionary construction
- **Low-Rank and Sparse Representation (LRSR)** for global decomposition

---

## Features

- Unsupervised hyperspectral anomaly detection pipeline
- Background and potential anomaly dictionary construction
- LRSR-based decomposition: `X = D_bg * Z + D_an * S + E`
- ROC and AUC evaluation
- Visualization of background/anomaly/noise and cluster maps
- Enhanced utilities for:
  - config-driven runs (`config_default.yaml`)
  - richer metrics (precision, recall, F1, confusion matrix)
  - result export (`json`, `csv`, `mat`)
  - performance tracking

---

## Repository Structure

- `demo.py`: baseline entry point
- `dic_constr.py`: background/potential anomaly dictionary construction
- `LRSR.py`: ALM-based low-rank + sparse optimization
- `HyperProTool.py`: hyperspectral processing utilities (reshape, normalization, windowing, clustering, SOMP)
- `result_show.py`: visualization of decomposition outputs
- `ROC_AUC.py`: ROC curve and AUC computation
- `config_default.yaml`: configurable parameters for enhanced workflow
- `enhanced_metrics.py`, `enhanced_visualization.py`, `result_exporter.py`, `performance_tracker.py`, `config_manager.py`: enhanced modules
- `run_enhanced.bat`, `run_enhanced.sh`: convenience runners for enhanced demo
- `results/`: output folder for metrics/reports/scores
- `info_given.txt`: detailed conceptual and pipeline notes

---

## Requirements

## Python
- Python 3.8+ recommended

## Core Dependencies
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`

## Optional / Enhanced Dependencies
- `pyyaml` (for YAML config loading)
- `seaborn` (for nicer confusion matrix plots)

Install:

```bash
pip install numpy scipy scikit-learn matplotlib pyyaml seaborn
```

---

## Dataset

This code expects MATLAB `.mat` files in the project root:
- `Sandiego.mat` with variable key `Sandiego` (hyperspectral cube)
- `PlaneGT.mat` with variable key `PlaneGT` (ground truth mask for evaluation)

Default assumption in code/config:
- crop size: `100 x 100`
- original bands: `224`
- bad-band removal applied before processing

If you use the San Diego AVIRIS dataset from prior work, you may also cite:

```bibtex
@article{xu2015anomaly,
  title={Anomaly detection in hyperspectral images based on low-rank and sparse representation},
  author={Xu, Yang and Wu, Zebin and Li, Jun and Plaza, Antonio and Wei, Zhihui},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={54},
  number={4},
  pages={1990--2000},
  year={2015},
  publisher={IEEE}
}
```

---

## Quick Start

1. Put `Sandiego.mat` and `PlaneGT.mat` in the repository root.
2. Install dependencies.
3. Run baseline demo:

```bash
python demo.py
```

This executes:
1. data loading and preprocessing
2. dictionary construction (`dic_constr`)
3. LRSR decomposition (`LRSR`)
4. visualization (`result_show`)
5. ROC/AUC evaluation (`ROC_AUC`)

---

## Enhanced Run (Config + Exports + Extra Metrics)

`run_enhanced.bat` and `run_enhanced.sh` call:

```bash
python demo_enhanced.py
```

If `demo_enhanced.py` is not present in your project root, copy it from the documentation bundle (or add it at root) before using the enhanced runners.

Then run:

- Windows:
  - `run_enhanced.bat`
- Linux/macOS:
  - `bash run_enhanced.sh`

Enhanced run adds:
- comprehensive metrics (AUC, precision, recall, F1, accuracy)
- confusion matrix visualization
- anomaly score CSV export
- summary JSON reports
- MATLAB-compatible output export
- stage-wise timing report

---

## Configuration

Edit `config_default.yaml` to tune:
- data files and crop size
- bad-band ranges
- dictionary construction parameters:
  - `win_size`
  - `cluster_num`
  - `K`
  - `selected_dic_percent`
  - `target_dic_num`
- LRSR parameters:
  - `beta`
  - `lmda`
  - iteration/convergence settings
- output saving and export formats

---

## Outputs

Baseline and enhanced workflows generate visual and numeric outputs.  
Enhanced workflow writes timestamped artifacts to `results/`, such as:
- `metrics_*.json`
- `summary_report_*.json`
- `anomaly_scores_*.csv`
- `results_*.mat` (if enabled)
- plot images (heatmaps/confusion matrix/overlays)

---

## Notes and Troubleshooting

- `result_show.py` expects `cluster_assment.mat` produced during dictionary construction.
- Ground truth shape must match cropped scene size.
- First run can be slow (dictionary construction + iterative LRSR).
- If YAML loading fails, install `pyyaml`.
- If seaborn is missing, confusion matrix plotting falls back to matplotlib.

---

## Citation

If this repository helps your research, please cite:

```bibtex
@article{li2018hyperspectral,
  title={Hyperspectral Anomaly Detection via Background and Potential Anomaly Detection},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2018}
}
```

(Please replace with the exact bibliographic metadata used in your manuscript/reference manager.)

---

## Acknowledgments

- AVIRIS data source and prior low-rank sparse anomaly detection work
- Original and enhanced code contributors for this implementation
