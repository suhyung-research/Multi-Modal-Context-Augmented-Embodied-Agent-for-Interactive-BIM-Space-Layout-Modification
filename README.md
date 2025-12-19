# Multi-Modal Context-Augmented Embodied Agent for Interactive BIM Space Layout Modification

## Overview
This project implements an iterative agent loop for revising BIM floorplans. The loop plans with an LLM, applies wall relocations via a rule-based executor, renders updated floorplans, and has a supervisor LLM review the outcome. Two prompt contexts are supported: structured-only JSON and multimodal (JSON + rendered image). Experiments target expanding the living room while respecting surrounding spaces and the building boundary.

## Repository Structure
- [plan_revision_runner.py](plan_revision_runner.py): End-to-end validation runner that loads `validation_set.csv`, iterates plan→execute→review up to a max iteration budget, logs with MLflow, and writes artifacts under `results/`.
- [utils.py](utils.py): Floorplan rendering utilities (JSON to PNG) and geometry helpers.
- [iterative_image_generation.py](iterative_image_generation.py): Batch renderer to produce `dataset/image/*.png` from `dataset/json/*.json` using the same rendering pipeline.
- [results_visualization.py](results_visualization.py): Generates filtered success-rate and iteration plots from `results/validation_result_manual.csv` and saves summaries under `results/visualization/`.
- [validation_set.csv](validation_set.csv): Evaluation cases; columns `data`, `task`, `context` (`structured` or `multimodal`).
- `dataset/`: Source floorplan JSON files (`dataset/json`) and rendered reference images (`dataset/image`).
- `results/`: Auto-generated run outputs (JSONs, PNGs, CSV summaries, plots). This folder can be large and is typically excluded from version control.

## Prerequisites
- Python 3.10+.
- Packages: `openai`, `mlflow`, `python-dotenv`, `pandas`, `numpy`, `matplotlib` (install via pip as needed).
- Environment: `OPENAI_API_KEY` set (e.g., via a local `.env`, not committed).

## Setup
1. Create and activate a virtual environment.
2. Install dependencies, for example:
   ```bash
   pip install openai mlflow python-dotenv pandas numpy matplotlib
   ```
3. Ensure `OPENAI_API_KEY` is available in your environment (or add to `.env`).
4. Place floorplan JSON files in `dataset/json/` (each with `Elements` and `Rooms` as expected by `utils.py`).

## Usage
### 1) Render reference images
Renders every JSON in `dataset/json` to `dataset/image`:
```bash
python iterative_image_generation.py
```
Note: the script currently points to absolute paths under this project directory; adjust if you relocate the project.

### 2) Run validation loop
Executes the iterative plan→execute→review loop for all rows in `validation_set.csv` and saves outputs to `results/<data>_<task>_<context>/` plus `results/validation_result.csv`:
```bash
python plan_revision_runner.py
```
Optional: launch MLflow UI to inspect runs
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### 3) Visualize filtered results
Consumes `results/validation_result_manual.csv` (manual labels) and writes filtered summaries and plots to `results/visualization/`:
```bash
python results_visualization.py --root .
```

## Data and Outputs
- Input JSON schema: `Elements` (with `element_id`, `start_point`, `end_point`) and `Rooms` (with `room_id`, `room_name`, `bounding_elements_ids`).
- Validation CSV: rows specify which JSON to use, the task phrasing, and context mode.
- Outputs per case: `input.png`, `candidate_iter{i}.json`, `candidate_iter{i}.png`, plus aggregated CSVs under `results/`.

## Notes
- Keep `.env` untracked; avoid committing keys or large result artifacts.
- The agent uses the `gpt-5.1` chat completion model with JSON schema-constrained responses for planning and review.
