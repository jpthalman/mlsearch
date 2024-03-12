# mlsearch

Download miniconda:
https://docs.conda.io/projects/miniconda/en/latest/

From the main folder run:
```conda env create -f environment.yaml```

Once installed run:
```conda activate mlsearch```

To update the environment run:
```conda env update```

## Data Extraction

From the repo root, run this command:
```python -m data.extract```

This can take several hours, so should be run from a tmux terminal.

## Training

For quick local training run this from the repo root:
```python -m train.run```

For full training runs:
```python -m train.run --experiment_name="something" --for-real```

This can take several hours, so should be run from a tmux terminal.

## Visualization

To visualize individual scenarios, run:
```PYTHONPATH="${pwd}:$PYTHONPATH" streamlit run vis/scenario.py```

## Evaluation

To generate metrics on the test set for a completed experiment, run:
```python -m eval.generate_metrics --experiment_name=something```

This can take several hours, so should be run from a tmux terminal.

To plot the generated metrics, run:
```python -m eval.plot_metrics --experiment_name=something```
