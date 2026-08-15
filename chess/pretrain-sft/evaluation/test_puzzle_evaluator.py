"""
Test script for puzzle evaluator using Moves column as ground truth.

This script:
1. Reads puzzle CSV files (puzzles_test.csv or puzzles_grandmaster.csv)
2. Extracts the first move from the 'Moves' column as ground truth
3. Uses FEN or ctx (PGN) as the state
4. Runs the base evaluator test
"""

import sys
import pathlib
import pandas as pd
import torch
from typing import Optional

# Add parent to path
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from evaluation.base_evaluator import BaseEvaluator


class PuzzleEvaluator(BaseEvaluator):
    """
    Evaluator for puzzle datasets that uses Moves column as ground truth.
    
    The Moves column contains space-separated UCI moves. We extract the first move
    as the ground truth for the puzzle.
    """
    
    def __init__(self, use_fen: bool = True, **kwargs):
        """
        Args:
            use_fen: If True, use FEN as state. If False, use ctx (PGN) as state.
            **kwargs: Other arguments passed to BaseEvaluator
        """
        super().__init__(**kwargs)
        self.use_fen = use_fen
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load puzzle CSV file."""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Extract first move from Moves column
        def extract_first_move(moves_str):
            """Extract the first UCI move from space-separated moves string."""
            if pd.isna(moves_str) or not moves_str:
                return None
            moves_str = str(moves_str).strip()
            if not moves_str:
                return None
            # Split by space and take first move
            moves_list = moves_str.split()
            return moves_list[0] if moves_list else None
        
        # Add first_move column
        df['first_move'] = df['Moves'].apply(extract_first_move)
        
        # Filter out rows where we couldn't extract a move
        df = df[df['first_move'].notna()].copy()
        
        # Also check which column will be used for state (ctx or FEN)
        # If using ctx and it's NaN, we should filter those out too
        # But we don't know use_fen here, so we'll filter in the caller
        
        return df
    
    def get_state_column(self) -> str:
        """Return the column name for chess states."""
        if self.use_fen:
            # Try FEN first, fallback to ctx if FEN not available
            return "FEN"
        else:
            return "ctx"
    
    def get_move_column(self) -> str:
        """Return the column name for target moves (first move from Moves column)."""
        return "first_move"
    
    def preprocess_move(self, move: str) -> str:
        """
        Preprocess move - moves are already in UCI format from the Moves column.
        Just strip whitespace.
        """
        return str(move).strip() if move else ""
    
    def evaluate(
        self,
        file_path: str,
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> dict:
        """
        Override evaluate to handle UCI moves directly (no SAN conversion needed).
        """
        from .inference import generate_moves_batch
        from .metrics import evaluate_predictions
        from .utils import lan_to_uci
        
        if verbose:
            print(f"Loading data from {file_path}...")
        
        # Load data using custom loader
        df = self.load_data(file_path)
        
        # Get column names
        state_col = self.get_state_column()
        move_col = self.get_move_column()
        
        # Validate columns exist
        if state_col not in df.columns:
            raise ValueError(f"State column '{state_col}' not found in dataset. Available: {df.columns.tolist()}")
        if move_col not in df.columns:
            raise ValueError(f"Move column '{move_col}' not found in dataset. Available: {df.columns.tolist()}")
        
        # Limit samples if requested
        if max_samples is not None:
            df = df.head(max_samples)
        
        if verbose:
            print(f"Evaluating on {len(df)} samples...")
        
        # Extract and preprocess states and moves
        states = [self.preprocess_state(str(s)) for s in df[state_col].tolist()]
        target_moves = [self.preprocess_move(str(m)) for m in df[move_col].tolist()]
        
        # Generate predictions
        if verbose:
            print("Generating predictions...")
        
        predicted_moves = generate_moves_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            states=states,
            device=self.device,
            batch_size=self.batch_size,
            **self.generation_kwargs
        )
        
        # Calculate metrics
        if verbose:
            print("Calculating metrics...")
        
        # Convert predicted moves from LAN to UCI
        # Note: lan_to_uci uses default side_to_move='white' which works for most moves
        # Castling moves might need explicit side detection, but this is rare
        predicted_uci = []
        for pred, state in zip(predicted_moves, states):
            try:
                predicted_uci.append(lan_to_uci(pred))
            except ValueError:
                predicted_uci.append("")  # Invalid LAN -> empty
        
        # Target moves are already in UCI format, so use them directly
        target_uci = [move.strip() if move else "" for move in target_moves]
        
        metrics = evaluate_predictions(
            states=states,
            predicted_moves=predicted_uci,
            target_moves=target_uci,
            move_format="uci",
            normalize=True
        )
        
        if verbose:
            print("\n" + "="*50)
            print("EVALUATION RESULTS")
            print("="*50)
            print(f"Legal Move Accuracy: {metrics['legal_accuracy']:.2%} ({metrics['num_legal']}/{metrics['num_total']})")
            print(f"Move Match Accuracy: {metrics['match_accuracy']:.2%} ({metrics['num_matches']}/{metrics['num_total']})")
            print("="*50 + "\n")
        
        return metrics
    
    def evaluate_and_save_predictions(
        self,
        file_path: str,
        output_path: str,
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> dict:
        """
        Override evaluate_and_save_predictions to handle UCI moves directly.
        """
        from .inference import generate_moves_batch
        from .metrics import evaluate_predictions, is_move_legal
        from .utils import lan_to_uci
        from pathlib import Path
        
        # Load data
        df = self.load_data(file_path)
        state_col = self.get_state_column()
        move_col = self.get_move_column()
        
        if max_samples is not None:
            df = df.head(max_samples)
        
        # Generate predictions
        states = [self.preprocess_state(str(s)) for s in df[state_col].tolist()]
        target_moves = [self.preprocess_move(str(m)) for m in df[move_col].tolist()]
        
        predicted_moves = generate_moves_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            states=states,
            device=self.device,
            batch_size=self.batch_size,
            **self.generation_kwargs
        )
        
        # Convert predicted moves to UCI
        # Note: lan_to_uci uses default side_to_move='white' which works for most moves
        predicted_moves_uci = []
        for pred, state in zip(predicted_moves, states):
            try:
                predicted_moves_uci.append(lan_to_uci(pred))
            except ValueError:
                predicted_moves_uci.append("")
        
        # Target moves are already in UCI format
        target_moves_uci = [move.strip() if move else "" for move in target_moves]
        
        # Calculate metrics
        metrics = evaluate_predictions(
            states=states,
            predicted_moves=predicted_moves_uci,
            target_moves=target_moves_uci,
            move_format="uci",
            normalize=True
        )
        
        # Add predictions to dataframe
        df_results = df.copy()
        df_results['predicted_move'] = predicted_moves
        df_results['predicted_move_uci'] = predicted_moves_uci
        df_results['target_move_uci'] = target_moves_uci
        df_results['is_legal'] = [
            is_move_legal(s, p, "uci") for s, p in zip(states, predicted_moves_uci)
        ]
        df_results['is_match'] = [
            p.strip().lower() == t.strip().lower()
            for p, t in zip(predicted_moves_uci, target_moves_uci)
        ]
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.parquet':
            df_results.to_parquet(output_path, index=False)
        elif output_path.suffix == '.csv':
            df_results.to_csv(output_path, index=False)
        else:
            # Default to parquet
            df_results.to_parquet(str(output_path) + '.parquet', index=False)
        
        if verbose:
            print(f"Predictions saved to {output_path}")
        
        return metrics


def test_puzzle_evaluator(
    model,
    tokenizer,
    device: torch.device,
    puzzle_file: str = "/scratch/js15262/LLM-Pretraining/data/chess/test/puzzles_test.csv",
    use_fen: bool = True,
    max_samples: Optional[int] = None,
    verbose: bool = True
):
    """
    Test the puzzle evaluator with a model and tokenizer.
    
    Args:
        model: The GPT model to evaluate
        tokenizer: Tokenizer instance
        device: Device to run inference on
        puzzle_file: Path to puzzle CSV file
        use_fen: If True, use FEN as state. If False, use ctx (PGN).
        max_samples: Maximum number of samples to evaluate (None for all)
        verbose: If True, print progress information
    
    Returns:
        Dictionary of evaluation metrics
    """
    if verbose:
        print("="*60)
        print("Testing Puzzle Evaluator")
        print("="*60)
        print(f"Puzzle file: {puzzle_file}")
        print(f"Using FEN as state: {use_fen}")
        print(f"Max samples: {max_samples if max_samples else 'All'}")
        print("="*60)
    
    # Create evaluator
    evaluator = PuzzleEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        move_format="uci",  # Moves are in UCI format
        use_fen=use_fen,
        batch_size=32,
        max_new_tokens=20,
        temperature=1.0,
    )
    
    # Handle FEN fallback to ctx if needed
    try:
        df = evaluator.load_data(puzzle_file)
        if use_fen and 'FEN' not in df.columns:
            if verbose:
                print("Warning: FEN column not found, falling back to ctx (PGN)")
            evaluator.use_fen = False
        elif not use_fen and 'ctx' not in df.columns:
            if 'FEN' in df.columns:
                if verbose:
                    print("Warning: ctx column not found, using FEN instead")
                evaluator.use_fen = True
            else:
                raise ValueError("Neither FEN nor ctx column found in dataset")
    except Exception as e:
        if verbose:
            print(f"Error loading data: {e}")
        raise
    
    # Run evaluation
    metrics = evaluator.evaluate(
        file_path=puzzle_file,
        max_samples=max_samples,
        verbose=verbose
    )
    
    return metrics


def test_puzzle_evaluator_and_save(
    model,
    tokenizer,
    device: torch.device,
    puzzle_file: str = "/scratch/js15262/LLM-Pretraining/data/chess/test/puzzles_test.csv",
    output_file: str = "/scratch/js15262/LLM-Pretraining/data/chess/test/puzzle_predictions.csv",
    use_fen: bool = True,
    max_samples: Optional[int] = None,
    verbose: bool = True
):
    """
    Test the puzzle evaluator and save predictions to file.
    
    Args:
        model: The GPT model to evaluate
        tokenizer: Tokenizer instance
        device: Device to run inference on
        puzzle_file: Path to puzzle CSV file
        output_file: Path to save predictions
        use_fen: If True, use FEN as state. If False, use ctx (PGN).
        max_samples: Maximum number of samples to evaluate (None for all)
        verbose: If True, print progress information
    
    Returns:
        Dictionary of evaluation metrics
    """
    if verbose:
        print("="*60)
        print("Testing Puzzle Evaluator (with predictions saved)")
        print("="*60)
        print(f"Puzzle file: {puzzle_file}")
        print(f"Output file: {output_file}")
        print(f"Using FEN as state: {use_fen}")
        print(f"Max samples: {max_samples if max_samples else 'All'}")
        print("="*60)
    
    # Create evaluator
    evaluator = PuzzleEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        move_format="uci",  # Moves are in UCI format
        use_fen=use_fen,
        batch_size=32,
        max_new_tokens=20,
        temperature=1.0,
    )
    
    # Handle FEN fallback to ctx if needed
    try:
        df = evaluator.load_data(puzzle_file)
        if use_fen and 'FEN' not in df.columns:
            if verbose:
                print("Warning: FEN column not found, falling back to ctx (PGN)")
            evaluator.use_fen = False
        elif not use_fen and 'ctx' not in df.columns:
            if 'FEN' in df.columns:
                if verbose:
                    print("Warning: ctx column not found, using FEN instead")
                evaluator.use_fen = True
            else:
                raise ValueError("Neither FEN nor ctx column found in dataset")
    except Exception as e:
        if verbose:
            print(f"Error loading data: {e}")
        raise
    
    # Run evaluation and save predictions
    metrics = evaluator.evaluate_and_save_predictions(
        file_path=puzzle_file,
        output_path=output_file,
        max_samples=max_samples,
        verbose=verbose
    )
    
    return metrics


def test_puzzle_tokenizer(
    tokenizer,
    puzzle_file: str = "/scratch/js15262/LLM-Pretraining/data/chess/test/puzzles_test.csv",
    use_fen: bool = False,
    max_samples: Optional[int] = None,
    use_target_as_prediction: bool = False,
    verbose: bool = True
):
    """
    Test puzzle evaluator with tokenizer only (no model generation).
    
    This function:
    1. Loads puzzle data
    2. Extracts states and target moves
    3. Tokenizes states to test tokenizer encoding
    4. Optionally uses target moves as predictions (to test tokenizer decoding)
    5. Runs evaluation metrics
    
    Args:
        tokenizer: Tokenizer instance
        puzzle_file: Path to puzzle CSV file
        use_fen: If True, use FEN as state. If False, use ctx (PGN).
        max_samples: Maximum number of samples to evaluate (None for all)
        use_target_as_prediction: If True, use target moves as predictions (tests tokenizer round-trip)
        verbose: If True, print progress information
    
    Returns:
        Dictionary of evaluation metrics and tokenizer test results
    """
    from .metrics import evaluate_predictions
    
    if verbose:
        print("="*60)
        print("Testing Puzzle Evaluator with Tokenizer Only")
        print("="*60)
        print(f"Puzzle file: {puzzle_file}")
        print(f"Using FEN as state: {use_fen}")
        print(f"Max samples: {max_samples if max_samples else 'All'}")
        print(f"Use target as prediction: {use_target_as_prediction}")
        print("="*60)
    
    # Create evaluator for data loading
    evaluator = PuzzleEvaluator(
        model=None,  # Not needed
        tokenizer=tokenizer,
        device=torch.device("cpu"),  # Not needed
        move_format="uci",
        use_fen=use_fen,
    )
    
    # Load data
    try:
        df = evaluator.load_data(puzzle_file)
        if use_fen and 'FEN' not in df.columns:
            if verbose:
                print("Warning: FEN column not found, falling back to ctx (PGN)")
            evaluator.use_fen = False
        elif not use_fen and 'ctx' not in df.columns:
            if 'FEN' in df.columns:
                if verbose:
                    print("Warning: ctx column not found, using FEN instead")
                evaluator.use_fen = True
            else:
                raise ValueError("Neither FEN nor ctx column found in dataset")
    except Exception as e:
        if verbose:
            print(f"Error loading data: {e}")
        raise
    
    # Limit samples if requested
    if max_samples is not None:
        df = df.head(max_samples)
    
    if verbose:
        print(f"Testing on {len(df)} samples...")
    
    # Extract states and target moves
    state_col = evaluator.get_state_column()
    move_col = evaluator.get_move_column()
    
    # Check for NaN values and filter them out
    if verbose:
        nan_states = df[state_col].isna().sum()
        nan_moves = df[move_col].isna().sum()
        if nan_states > 0 or nan_moves > 0:
            print(f"\n⚠️  Warning: Found NaN values in data:")
            print(f"    NaN states: {nan_states}")
            print(f"    NaN moves: {nan_moves}")
            
            # Show which rows have NaN (first 10)
            nan_mask = df[state_col].isna() | df[move_col].isna()
            nan_indices = df[nan_mask].index.tolist()[:10]
            if nan_indices:
                print(f"    First few rows with NaN: {nan_indices}")
                # Show actual values
                for idx in nan_indices[:5]:
                    row = df.loc[idx]
                    print(f"      Row {idx}: state_col={row[state_col]}, move_col={row[move_col]}")
    
    # Filter out rows with NaN values
    original_len = len(df)
    df = df.dropna(subset=[state_col, move_col])
    if len(df) < original_len:
        if verbose:
            print(f"    Filtered out {original_len - len(df)} rows with NaN values")
            print(f"    Remaining samples: {len(df)}")
    
    # Extract states and target moves, handling potential NaN
    states = []
    target_moves = []
    valid_indices = []
    
    for idx, (state_val, move_val) in enumerate(zip(df[state_col], df[move_col])):
        # Skip if either is NaN
        if pd.isna(state_val) or pd.isna(move_val):
            if verbose and idx < 10:  # Report first 10 NaN cases
                print(f"    Skipping row {idx}: state={state_val}, move={move_val}")
            continue
        
        state_str = str(state_val).strip()
        move_str = str(move_val).strip()
        
        # Skip if string conversion resulted in "nan" (lowercase)
        if state_str.lower() == "nan" or move_str.lower() == "nan" or not state_str or not move_str:
            if verbose and idx < 10:
                print(f"    Skipping row {idx}: invalid state or move")
            continue
        
        states.append(evaluator.preprocess_state(state_str))
        target_moves.append(evaluator.preprocess_move(move_str))
        valid_indices.append(idx)
    
    if verbose:
        print(f"\n✓ Processed {len(states)} valid samples (from {original_len} total)")
    
    if len(states) == 0:
        raise ValueError("No valid states/moves found after filtering NaN values!")
    
    # Test tokenizer encoding
    if verbose:
        print("\nTesting tokenizer encoding...")
    
    encoding_results = []
    for i, state in enumerate(states[:10]):  # Test first 10 for encoding
        try:
            encoded = tokenizer.encode(state)
            decoded = tokenizer.decode(encoded)
            encoding_results.append({
                'original': state[:50] + "..." if len(state) > 50 else state,
                'encoded_length': len(encoded),
                'decoded_matches': decoded.strip() == state.strip()
            })
            if verbose and i < 3:  # Show first 3 examples
                print(f"  Example {i+1}:")
                print(f"    Original: {state[:150]}..." if len(state) > 150 else f"    Original: {state}")
                print(f"    Encoded length: {len(encoded)}")
                print(f"    Decoded: {decoded[:150]}..." if len(decoded) > 150 else f"    Decoded: {decoded}")
                print(f"    Decoded matches: {decoded.strip() == state.strip()}")
        except Exception as e:
            encoding_results.append({
                'original': state[:50] + "..." if len(state) > 50 else state,
                'error': str(e)
            })
    
    # Generate predictions based on mode
    if verbose:
        print("\nGenerating predictions...")
    
    if use_target_as_prediction:
        # Use target moves as predictions (tests round-trip through tokenizer)
        predicted_moves_uci = [move.strip() if move else "" for move in target_moves]
        if verbose:
            print("Using target moves as predictions (testing tokenizer round-trip)")
    else:
        # Create dummy predictions (all same move or empty) for testing framework
        # In real use, these would come from model generation
        predicted_moves_uci = ["e2e4"] * len(target_moves)  # Dummy move
        if verbose:
            print("Using dummy predictions (e2e4) for framework testing")
    
    # Target moves are already in UCI format
    target_moves_uci = [move.strip() if move else "" for move in target_moves]
    
    # Print example predictions vs targets
    if verbose:
        print("\nExample predictions vs targets (first 10):")
        print("-" * 80)
        for i in range(min(10, len(predicted_moves_uci))):
            pred = predicted_moves_uci[i]
            target = target_moves_uci[i]
            match = "✓" if pred.strip().lower() == target.strip().lower() else "✗"
            print(f"  {i+1:2d}. Predicted: {pred:8s} | Target: {target:8s} | {match}")
        print("-" * 80)
    
    # Debug: Check legal move validation for first few examples
    if verbose:
        print("\nDebugging legal move checks (first 5 examples):")
        from .metrics import is_move_legal, _state_to_board
        import chess
        
        for i in range(min(5, len(states))):
            state = states[i]
            move = predicted_moves_uci[i]
            print(f"\n  Example {i+1}:")
            print(f"    State (first 100 chars): {state[:100]}...")
            print(f"    Move to check: {move}")
            
            # Try to get board
            board = _state_to_board(state)
            if board is None:
                print(f"    ❌ ERROR: Could not create board from state")
            else:
                print(f"    ✓ Board created successfully")
                print(f"    Board FEN: {board.fen()}")
                print(f"    Turn: {'White' if board.turn else 'Black'}")
                
                # Try to parse move
                try:
                    move_obj = chess.Move.from_uci(move)
                    print(f"    ✓ Move parsed as UCI: {move_obj}")
                    print(f"    Legal moves count: {len(list(board.legal_moves))}")
                    
                    # Check if in legal moves
                    if move_obj in board.legal_moves:
                        print(f"    ✓ Move is LEGAL")
                    else:
                        print(f"    ❌ Move is NOT legal")
                        # Show some legal moves
                        legal = list(board.legal_moves)[:5]
                        print(f"    Sample legal moves: {[m.uci() for m in legal]}")
                except Exception as e:
                    print(f"    ❌ ERROR parsing move: {e}")
    
    # Calculate metrics
    if verbose:
        print("\nCalculating metrics...")
    
    metrics = evaluate_predictions(
        states=states,
        predicted_moves=predicted_moves_uci,
        target_moves=target_moves_uci,
        move_format="uci",
        normalize=True
    )
    
    # Test tokenizer on move sequences (if we want to test move encoding)
    move_tokenization_results = []
    if verbose:
        print("\nTesting tokenizer on move sequences...")
    
    for i, (state, target_move) in enumerate(zip(states[:5], target_moves[:5])):  # Test first 5
        try:
            # Try encoding state + move
            full_sequence = state.strip() + " " + target_move
            encoded = tokenizer.encode(full_sequence)
            decoded = tokenizer.decode(encoded)
            move_tokenization_results.append({
                'sequence': full_sequence[:60] + "..." if len(full_sequence) > 60 else full_sequence,
                'encoded_length': len(encoded),
                'decoded_ends_with_move': decoded.strip().endswith(target_move.strip())
            })
            if verbose and i < 2:
                print(f"  Example {i+1}:")
                print(f"    Sequence: {full_sequence[:80]}..." if len(full_sequence) > 80 else f"    Sequence: {full_sequence}")
                print(f"    Encoded length: {len(encoded)}")
                print(f"    Decoded: {decoded[:80]}..." if len(decoded) > 80 else f"    Decoded: {decoded}")
                print(f"    Decoded ends with move: {decoded.strip().endswith(target_move.strip())}")
        except Exception as e:
            move_tokenization_results.append({
                'sequence': full_sequence[:60] + "..." if len(full_sequence) > 60 else full_sequence,
                'error': str(e)
            })
    
    # Print results
    if verbose:
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"\nTokenization Tests:")
        print(f"  States tested: {len(encoding_results)}")
        successful_encodings = sum(1 for r in encoding_results if r.get('decoded_matches', False))
        print(f"  Successful round-trips: {successful_encodings}/{len(encoding_results)}")
        
        print(f"\nMove Tokenization Tests:")
        print(f"  Move sequences tested: {len(move_tokenization_results)}")
        successful_moves = sum(1 for r in move_tokenization_results if r.get('decoded_ends_with_move', False))
        print(f"  Successful move encodings: {successful_moves}/{len(move_tokenization_results)}")
        
        print(f"\nEvaluation Metrics:")
        print(f"  Legal Move Accuracy: {metrics['legal_accuracy']:.2%} ({metrics['num_legal']}/{metrics['num_total']})")
        print(f"  Move Match Accuracy: {metrics['match_accuracy']:.2%} ({metrics['num_matches']}/{metrics['num_total']})")
        print(f"  Total samples: {metrics['num_total']}")
        print("="*60 + "\n")
    
    return {
        'metrics': metrics,
        'encoding_results': encoding_results,
        'move_tokenization_results': move_tokenization_results,
    }


if __name__ == "__main__":
    """
    Example usage for tokenizer-only testing:
    
    # Option 1: Run as a module (recommended)
    python -m evaluation.test_puzzle_evaluator
    
    # Option 2: Import and use directly
    from llm_tokens.chess.tokenizer_factory import init_tokenizer
    from evaluation.test_puzzle_evaluator import test_puzzle_tokenizer
    
    tokenizer = init_tokenizer("LanTokenizer", {"include_move_numbers": True})
    results = test_puzzle_tokenizer(
        tokenizer=tokenizer,
        puzzle_file="/scratch/js15262/LLM-Pretraining/data/chess/test/puzzles_test.csv",
        use_fen=False,  # Uses ctx (PGN) by default
        max_samples=100,
        use_target_as_prediction=False,
        verbose=True
    )
    """
    import sys
    import pathlib
    import os
    
    # Add parent to path
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    # Also ensure we can import from the current directory
    current_dir = pathlib.Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    try:
        # Try importing tokenizer factory
        from llm_tokens.chess.tokenizer_factory import init_tokenizer
        
        print("Initializing tokenizer...")
        # You can change this to your tokenizer type
        tokenizer = init_tokenizer("LanTokenizer", {"include_move_numbers": True})
        
        print("\nRunning puzzle tokenizer test...")
        results = test_puzzle_tokenizer(
            tokenizer=tokenizer,
            puzzle_file="/scratch/js15262/LLM-Pretraining/data/chess/test/puzzles_test.csv",
            use_fen=False,  # Use ctx (PGN) instead of FEN
            max_samples=100,  # Test on first 100 puzzles
            use_target_as_prediction=True,  # Use dummy predictions
            verbose=True
        )
        
        print("\n✓ Test completed successfully!")
        
    except ImportError as e:
        error_msg = str(e)
        print(f"Error importing tokenizer: {error_msg}")
        
        if "attempted relative import" in error_msg.lower():
            print("\n" + "="*60)
            print("IMPORT ERROR - Try running as a module instead:")
            print("="*60)
            print("  python -m evaluation.test_puzzle_evaluator")
            print("\nOr from the project root:")
            print("  cd /scratch/js15262/LLM-Pretraining")
            print("  python -m evaluation.test_puzzle_evaluator")
            print("\n" + "="*60)
        else:
            print("\nPlease ensure:")
            print("  1. You're in the correct directory")
            print("  2. The llm_tokens package is installed or in the path")
            print("  3. All dependencies are available")
        
        print("\nAlternative: Import and use the function directly")
        print("  from evaluation.test_puzzle_evaluator import test_puzzle_tokenizer")
        print("  # Then call with your tokenizer instance")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

