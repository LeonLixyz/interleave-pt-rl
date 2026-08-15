# Evaluation Framework

This module provides a flexible evaluation framework for chess move prediction tasks.

## Overview

The framework consists of:

1. **Model Inference** (`inference.py`): Functions to generate move predictions from chess states
2. **Metrics** (`metrics.py`): Calculate legal move accuracy and move matching accuracy
3. **Base Evaluator** (`base_evaluator.py`): Abstract base class for creating custom evaluators

## Quick Start

### Step 1: Create a Custom Evaluator

Subclass `BaseEvaluator` and implement three methods:

```python
from evaluation import BaseEvaluator
import pandas as pd

class MyEvaluator(BaseEvaluator):
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load your evaluation dataset."""
        return pd.read_parquet(file_path)
    
    def get_state_column(self) -> str:
        """Return the name of the column containing chess states."""
        return "state"  # Change to your column name
    
    def get_move_column(self) -> str:
        """Return the name of the column containing target moves."""
        return "move"  # Change to your column name
```

### Step 2: Run Evaluation

```python
from evaluation import MyEvaluator
import torch

# Initialize evaluator
evaluator = MyEvaluator(
    model=model,
    tokenizer=tokenizer,
    device=torch.device("cuda"),
    move_format="san",  # or "uci", "lan"
    temperature=1.0,
    top_k=None
)

# Run evaluation
metrics = evaluator.evaluate(
    file_path="path/to/eval_data.parquet",
    max_samples=1000,  # Optional: limit number of samples
    verbose=True
)

print(f"Legal Move Accuracy: {metrics['legal_accuracy']:.2%}")
print(f"Move Match Accuracy: {metrics['match_accuracy']:.2%}")
```

## Data Format

Your evaluation dataset should be a parquet file (or CSV) with at least two columns:

- **State column**: Chess position as PGN sequence or FEN string
- **Move column**: Target move in SAN/UCI/LAN format

Example:

| state | move |
|-------|------|
| "1. e4 e5 2. Nf3" | "Nc6" |
| "1. d4 d5 2. c4" | "e6" |

## Metrics

The framework calculates two metrics:

1. **Legal Move Accuracy**: Percentage of generated moves that are legal in the given position
2. **Move Match Accuracy**: Percentage of generated moves that exactly match the target move

## Advanced Usage

### Custom Preprocessing

Override `preprocess_state()` or `preprocess_move()` to add custom preprocessing:

```python
class MyEvaluator(BaseEvaluator):
    def preprocess_state(self, state: str) -> str:
        """Add custom preprocessing for states."""
        state = state.strip()
        # Add any custom logic here
        return state
    
    def preprocess_move(self, move: str) -> str:
        """Add custom preprocessing for moves."""
        return move.strip().upper()
```

### Save Predictions

Save predictions along with metrics:

```python
metrics = evaluator.evaluate_and_save_predictions(
    file_path="path/to/eval_data.parquet",
    output_path="path/to/predictions.parquet",
    max_samples=1000,
    verbose=True
)
```

This creates a file with columns: original data + `predicted_move`, `is_legal`, `is_match`

### Direct Inference

Use the inference functions directly without the evaluator class:

```python
from evaluation import generate_move

predicted_move = generate_move(
    model=model,
    tokenizer=tokenizer,
    state="1. e4 e5 2. Nf3",
    device=torch.device("cuda"),
    max_new_tokens=20,
    temperature=1.0,
    top_k=50
)
```

### Direct Metric Calculation

Calculate metrics directly:

```python
from evaluation import calculate_legal_move_accuracy, calculate_move_matching_accuracy

# Legal move accuracy
legal_metrics = calculate_legal_move_accuracy(
    states=["1. e4 e5 2. Nf3", "1. d4 d5"],
    predicted_moves=["Nc6", "c5"],
    move_format="san"
)

# Move matching accuracy
match_metrics = calculate_move_matching_accuracy(
    predicted_moves=["Nc6", "c5"],
    target_moves=["Nc6", "c6"],
    normalize=True
)
```

## Integration with Trainer

To integrate evaluation into training, see the trainer integration example in the main training loop.

## Examples

See `example_evaluator.py` for complete examples of:
- Simple evaluator with standard column names
- Custom evaluator with configurable columns and preprocessing

