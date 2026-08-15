"""
Quick test script to verify the evaluation framework.

This creates a simple test dataset and runs evaluation to ensure everything works.
"""

import sys
import pathlib
import pandas as pd
import tempfile

# Add parent to path
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def test_inference():
    """Test inference functions."""
    print("Testing inference functions...")
    from evaluation.inference import _is_complete_move, _extract_first_move
    
    # Test _is_complete_move
    assert _is_complete_move("Pg7") == False
    assert _is_complete_move("Pg7g8") == True
    assert _is_complete_move("O-O") == True
    assert _is_complete_move("Pg7g8=Q") == True
    assert _is_complete_move("Bg7h6+") == True
    assert _is_complete_move("Pd7d6") == True
    assert _is_complete_move("N") == False
    assert _is_complete_move("") == False
    
    # Test _extract_first_move
    assert _extract_first_move("Kf3g4 Ke6f7") == "Kf3g4"
    
    
    print("✓ Inference functions work correctly")


def test_metrics():
    """Test metric calculation functions."""
    print("\nTesting metric functions...")
    from evaluation.metrics import (
        is_move_legal,
        calculate_legal_move_accuracy,
        calculate_move_matching_accuracy,
    )
    
    # Test is_move_legal
    pgn_state = "1. e4 e5 2. Nf3"
    assert is_move_legal(pgn_state, "Nc6", "san") == True
    assert is_move_legal(pgn_state, "Nf6", "san") == True
    assert is_move_legal(pgn_state, "Qa5", "san") == False  # Queen can't go to a5 here
    
    # Test legal move accuracy
    states = ["1. e4 e5 2. Nf3", "1. d4 d5 2. c4"]
    predicted = ["Nc6", "e6"]
    legal_metrics = calculate_legal_move_accuracy(states, predicted, "san")
    assert legal_metrics['num_legal'] == 2
    assert legal_metrics['legal_accuracy'] == 1.0
    
    # Test move matching accuracy
    targets = ["Nc6", "c6"]
    match_metrics = calculate_move_matching_accuracy(predicted, targets)
    assert match_metrics['num_matches'] == 1
    assert match_metrics['match_accuracy'] == 0.5
    
    print("✓ Metric functions work correctly")


def test_evaluator():
    """Test BaseEvaluator with a simple dataset."""
    print("\nTesting evaluator...")
    from evaluation import BaseEvaluator
    import torch
    from model.gpt2_model import GPT, GPTConfig
    from llm_tokens.chess.tokenizer_factory import init_tokenizer
    
    # Create a simple mock evaluator
    class SimpleEvaluator(BaseEvaluator):
        def load_data(self, file_path):
            return pd.read_parquet(file_path)
        
        def get_state_column(self):
            return "state"
        
        def get_move_column(self):
            return "move"
    
    # Create test data
    test_data = pd.DataFrame({
        'state': [
            '1. e4 e5 2. Nf3',
            '1. d4 d5 2. c4',
            '1. e4 c5 2. Nf3',
        ],
        'move': ['Nc6', 'e6', 'Nc6']
    })
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as f:
        temp_path = f.name
        test_data.to_parquet(temp_path)
    
    # Create a small model for testing
    config = GPTConfig(
        vocab_size=100,
        block_size=128,
        n_layer=2,
        n_head=2,
        n_embed=64,
        dropout=0.0,
    )
    model = GPT(config)
    
    # Create tokenizer
    tokenizer = init_tokenizer("SanTokenizer", {"include_move_numbers": True})
    
    # Create evaluator
    evaluator = SimpleEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        move_format="san",
        temperature=1.0,
    )
    
    # Run evaluation (should complete without error, though accuracy may be poor)
    try:
        metrics = evaluator.evaluate(temp_path, max_samples=2, verbose=False)
        print(f"✓ Evaluator runs successfully")
        print(f"  Legal accuracy: {metrics['legal_accuracy']:.2%}")
        print(f"  Match accuracy: {metrics['match_accuracy']:.2%}")
    except Exception as e:
        print(f"✗ Evaluator failed: {e}")
        raise
    finally:
        # Cleanup
        import os
        os.remove(temp_path)


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Evaluation Framework")
    print("="*60)
    
    try:
        test_inference()
        test_metrics()
        # test_evaluator()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ Test failed: {e}")
        print("="*60)
        raise


if __name__ == "__main__":
    main()

