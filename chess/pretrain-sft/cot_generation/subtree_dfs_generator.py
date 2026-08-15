import sys
import argparse
import pandas as pd
import torch
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import chess
import chess.engine
import chess.pgn
import json
from tqdm import tqdm
import numpy as np
import io
import random
import re
import traceback
from collections import deque
import queue as _queue_module
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path so we can import cot_generation
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from cot_generation.policy import (
    random_sampling_policy,
    stockfish_sampling_policy,
    stockfish_eval,
)
from cot_generation.generator import TreeNode, board_from_pgn
from evaluation.inference import generate_move


def move_to_lan(board: chess.Board, move: chess.Move) -> str:
    """
    Convert a chess.Move to LAN string (combined format).
    """
    if board.is_castling(move):
        if chess.square_file(move.to_square) == 6:
            castle = "O-O"
        else:
            castle = "O-O-O"
        board.push(move)
        suffix = "#" if board.is_checkmate() else "+" if board.is_check() else ""
        board.pop()
        return castle + suffix

    parts = []
    piece = board.piece_at(move.from_square)
    parts.append(piece.symbol().upper() if piece else "P")
    parts.append(chess.square_name(move.from_square))
    if board.is_capture(move):
        parts.append("x")
    parts.append(chess.square_name(move.to_square))
    if move.promotion:
        parts.append("=")
        parts.append(chess.piece_symbol(move.promotion).upper())
    board.push(move)
    if board.is_checkmate():
        parts.append("#")
    elif board.is_check():
        parts.append("+")
    board.pop()
    return "".join(parts)


class ModelSamplingPolicy:
    """Wrapper for model-based sampling policy."""

    def __init__(self, model, tokenizer, device, temperature=1.0, seed=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.model.eval()
        self.seed = seed

    def _board_to_state(self, board: chess.Board) -> str:
        """Convert a chess board to PGN state string."""
        game = chess.pgn.Game()
        node = game
        move_stack = []
        for move in board.move_stack:
            move_stack.append(move)
        for move in move_stack:
            node = node.add_variation(move)
        exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
        pgn_str = game.accept(exporter).strip()
        return pgn_str if pgn_str else ""

    def _parse_move_from_output(self, first_move: str, board: chess.Board, legal_moves: List[chess.Move]) -> Optional[chess.Move]:
        """Try to parse a move string into a chess.Move from legal_moves."""
        if not first_move:
            return None
        try:
            parsed_move = chess.Move.from_uci(first_move)
            if parsed_move in legal_moves:
                return parsed_move
        except Exception:
            pass
        try:
            parsed_move = board.parse_san(first_move)
            if parsed_move in legal_moves:
                return parsed_move
        except Exception:
            pass
        for move in legal_moves:
            if move.uci() == first_move or board.san(move) == first_move:
                return move
        return None

    def __call__(self, board: chess.Board, legal_moves: List[chess.Move]) -> Optional[chess.Move]:
        if not legal_moves:
            return None

        state = self._board_to_state(board)

        try:
            full_output, first_move = generate_move(
                model=self.model,
                tokenizer=self.tokenizer,
                state=state,
                device=self.device,
                max_new_tokens=20,
                temperature=self.temperature,
                top_k=None,
                seed=self.seed,
            )
            parsed_move = self._parse_move_from_output(first_move, board, legal_moves)
            if parsed_move:
                return parsed_move
            return random.choice(legal_moves)
        except Exception as e:
            print(f"Error in model sampling: {e}")
            traceback.print_exc()
            return random.choice(legal_moves)

    def batch_sample(self, boards: List[chess.Board], legal_moves_list: List[List[chess.Move]], batch_size: int = 32) -> List[Optional[chess.Move]]:
        """
        Batch sample moves from multiple boards.

        Args:
            boards: List of chess boards to sample from
            legal_moves_list: List of legal moves for each board (must match length of boards)
            batch_size: Batch size for model inference

        Returns:
            List of sampled moves (one per board), or None if no valid move found
        """
        if len(boards) != len(legal_moves_list):
            raise ValueError("boards and legal_moves_list must have the same length")

        if not boards:
            return []

        # Convert boards to state strings
        states = [self._board_to_state(board) for board in boards]

        try:
            from evaluation.inference import generate_moves_batch
            moves, first_moves = generate_moves_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                states=states,
                device=self.device,
                batch_size=batch_size,
                max_new_tokens=20,
                temperature=self.temperature,
                top_k=None,
                seed=self.seed,
            )

            # Parse each move
            results = []
            for i, (board, legal_moves, first_move) in enumerate(zip(boards, legal_moves_list, first_moves)):
                if not legal_moves:
                    results.append(None)
                    continue
                parsed_move = self._parse_move_from_output(first_move, board, legal_moves)
                if parsed_move:
                    results.append(parsed_move)
                else:
                    # Fallback to random if parsing fails
                    results.append(random.choice(legal_moves))
            return results
        except Exception as e:
            print(f"Error in batch model sampling: {e}")
            traceback.print_exc()
            # Fallback to random for all
            return [random.choice(legal_moves) if legal_moves else None for legal_moves in legal_moves_list]


def load_gpt2_model_and_tokenizer(model_path: str, device: str, config_path: Optional[str] = None):
    """Load the model and tokenizer from the given path."""
    from model.gpt2_model import GPT, GPTConfig
    from llm_tokens.chess.tokenizer_factory import init_tokenizer

    if not config_path:
        raise ValueError("--config_path is required when using model sampling policy")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    tokenizer = init_tokenizer(name=cfg["tokenizer"]["name"], config=cfg["tokenizer"])
    vocab_size = int(tokenizer.get_vocab_size()) if hasattr(tokenizer, "get_vocab_size") else int(len(tokenizer.get_vocab()))

    model_cfg = cfg["model"]
    model = GPT(
        GPTConfig(
            vocab_size=vocab_size,
            block_size=model_cfg["block_size"],
            n_layer=model_cfg["n_layer"],
            n_head=model_cfg["n_head"],
            n_embed=model_cfg["n_embed"],
            dropout=model_cfg.get("dropout", 0.0),
            mlp_type=model_cfg.get("mlp_type", "mlp"),
        )
    )

    checkpoint_path = Path(model_path)
    if checkpoint_path.is_dir():
        weight_files = (
            list(checkpoint_path.glob("model.safetensors"))
            + list(checkpoint_path.glob("pytorch_model.bin"))
            + [f for f in checkpoint_path.glob("*.safetensors") if f.name != "optimizer.safetensors"]
            + [f for f in checkpoint_path.glob("*.bin") if f.name != "optimizer.bin"]
            + list(checkpoint_path.glob("*.pt"))
            + list(checkpoint_path.glob("*.pth"))
        )
        if not weight_files:
            raise FileNotFoundError(f"No model weights found in {checkpoint_path}")
        checkpoint_path = weight_files[0]

    if str(checkpoint_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected}")

    model = model.to(device)
    model.eval()
    return model, tokenizer

def load_hf_model_and_tokenizer(
    hf_model_path: str,
    device: str,
    config_path: str,
    torch_dtype: Optional[torch.dtype] = None,
):
    """
    Load a HF CausalLM via from_pretrained, but keep tokenizer identical to your YAML init_tokenizer.

    hf_model_path:
      - can be a local directory with config.json + model weights,
      - or a HF hub id (if your environment allows internet).
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    from llm_tokens.chess.tokenizer_factory import init_tokenizer

    if not config_path:
        raise ValueError("--config_path is required to build the SAME tokenizer")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # your tokenizer (source of truth)
    tokenizer = init_tokenizer(name=cfg["tokenizer"]["name"], config=cfg["tokenizer"])

    # HF model
    # (If your HF config includes wrong token ids, we overwrite them to match your tokenizer.)
    hf_cfg = AutoConfig.from_pretrained(hf_model_path)
    hf_cfg.vocab_size = tokenizer.get_vocab_size()
    hf_cfg.bos_token_id = tokenizer.bos_id()
    hf_cfg.eos_token_id = tokenizer.eos_id()
    hf_cfg.pad_token_id = tokenizer.pad_id()

    # dtype handling
    kwargs = {"config": hf_cfg}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(hf_model_path, **kwargs)

    # Move to device (HF may use device_map; keep it simple here)
    model = model.to(device)
    model.eval()

    return model, tokenizer

def create_sampling_policy(args, device):
    if args.sampling_policy == "random":
        return random_sampling_policy(seed=args.seed)
    if args.sampling_policy == "stockfish":
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)

        def policy(board, legal_moves):
            return stockfish_sampling_policy(
                board,
                legal_moves,
                engine,
                time_limit=args.stockfish_time,
                temperature=args.temperature,
                seed=args.seed,
            )

        return policy
    if args.sampling_policy == "model":
        if not args.model_path:
            raise ValueError("--model_path is required when using model sampling policy")
        if not args.config_path:
            raise ValueError("--config_path is required when using model sampling policy")
        model, tokenizer = load_gpt2_model_and_tokenizer(args.model_path, device, config_path=args.config_path)
        return ModelSamplingPolicy(model, tokenizer, device, temperature=args.temperature, seed = args.seed)
    if args.sampling_policy == "hf_model":
        if not args.model_path:
            raise ValueError("--model_path is required when using HF sampling policy")
        if not args.config_path:
            raise ValueError("--config_path is required when using HF sampling policy")
        model, tokenizer = load_hf_model_and_tokenizer(args.model_path, device, config_path=args.config_path)
        return ModelSamplingPolicy(model, tokenizer, device, temperature=args.temperature, seed = None)
    raise ValueError(f"Unknown sampling policy: {args.sampling_policy}")


# --- Trajectory Tree Node -------------------------------------------------- #

class TrajectoryTreeNode:
    """Tree node built from trajectories with generation order tracking."""

    def __init__(
        self,
        board: chess.Board,
        move: Optional[chess.Move] = None,
        parent: Optional["TrajectoryTreeNode"] = None,
        generation_order: int = 0,
    ):
        self.board = board
        self.move = move
        self.parent = parent
        self.children: Dict[chess.Move, "TrajectoryTreeNode"] = {}
        self.generation_order = generation_order  # Order in which this node was first created
        self.stockfish_value: Optional[float] = None


# --- Candidate and Trajectory Generation ----------------------------------- #

def sample_candidates_with_policy(
    board: chess.Board,
    sampling_policy,
    num_candidates: int,
    batch_size: int = 32,
) -> List[chess.Move]:
    """
    Sample candidate moves using the policy (without replacement).
    Returns moves in the order they were sampled by the policy.

    If sampling_policy is a ModelSamplingPolicy, uses batch sampling for efficiency.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []

    # Check if this is a ModelSamplingPolicy that supports batch sampling
    is_model_policy = isinstance(sampling_policy, ModelSamplingPolicy)

    if is_model_policy and num_candidates > 1:
        # Use batch sampling for model policies
        # Sample multiple times from the same board in a batch
        candidates: List[chess.Move] = []
        remaining = set(legal_moves)

        # We'll sample in batches, handling duplicates as we go
        # Sample more than needed to account for duplicates/invalid moves
        num_to_sample = min(num_candidates * 2, len(legal_moves))

        # Create batch: same board, but we'll filter results
        batch_boards = [board] * num_to_sample
        batch_legal_moves = [list(remaining)] * num_to_sample

        # Perform batch sampling
        sampled_moves = sampling_policy.batch_sample(
            boards=batch_boards,
            legal_moves_list=batch_legal_moves,
            batch_size=batch_size,
        )

        # Process results and add unique candidates in order
        seen = set()
        for mv in sampled_moves:
            if mv is not None and mv in remaining and mv not in seen:
                candidates.append(mv)
                seen.add(mv)
                remaining.remove(mv)
                if len(candidates) >= num_candidates:
                    break

        # If we still need more candidates and have remaining moves, do another batch
        if len(candidates) < num_candidates and remaining:
            additional_needed = num_candidates - len(candidates)
            additional_to_sample = min(additional_needed * 2, len(remaining))

            batch_boards = [board] * additional_to_sample
            batch_legal_moves = [list(remaining)] * additional_to_sample

            sampled_moves = sampling_policy.batch_sample(
                boards=batch_boards,
                legal_moves_list=batch_legal_moves,
                batch_size=batch_size,
            )

            for mv in sampled_moves:
                if mv is not None and mv in remaining and mv not in seen:
                    candidates.append(mv)
                    seen.add(mv)
                    remaining.remove(mv)
                    if len(candidates) >= num_candidates:
                        break

        return candidates[:num_candidates]
    else:
        # Fallback to sequential sampling for non-model policies or single candidate
        candidates: List[chess.Move] = []
        remaining = set(legal_moves)

        while len(candidates) < num_candidates and remaining:
            mv = sampling_policy(board, list(remaining))
            if mv is None or mv not in remaining:
                break
            candidates.append(mv)
            remaining.remove(mv)

        return candidates


def generate_single_trajectory(
    board: chess.Board,
    sampling_policy,
    max_depth: int,
) -> List[chess.Move]:
    """Generate a single trajectory (rollout) using the policy."""
    trajectory: List[chess.Move] = []
    current_board = board.copy()

    for _ in range(max_depth):
        if current_board.is_game_over():
            break
        legal_moves = list(current_board.legal_moves)
        if not legal_moves:
            break
        mv = sampling_policy(current_board, legal_moves)
        if mv is None:
            break
        trajectory.append(mv)
        current_board.push(mv)

    return trajectory


def generate_trajectories_for_candidate(
    root_board: chess.Board,
    candidate_move: chess.Move,
    sampling_policy,
    num_trajectories: int,
    trajectory_depth: int,
) -> List[List[chess.Move]]:
    """
    Generate multiple trajectories starting with the candidate move.
    Each trajectory is a list of moves starting from the candidate.
    """
    trajectories: List[List[chess.Move]] = []
    cand_board = root_board.copy()
    cand_board.push(candidate_move)

    for _ in range(num_trajectories):
        # Generate continuation after the candidate move
        continuation = generate_single_trajectory(
            board=cand_board,
            sampling_policy=sampling_policy,
            max_depth=trajectory_depth - 1,  # -1 because candidate is already 1 move
        )
        # Full trajectory includes the candidate move as first move
        full_trajectory = [candidate_move] + continuation
        trajectories.append(full_trajectory)

    return trajectories


# --- Tree Building --------------------------------------------------------- #

def build_tree_from_trajectories(
    root_board: chess.Board,
    all_trajectories: List[List[chess.Move]],
) -> TrajectoryTreeNode:
    """
    Build a tree by merging trajectories with common prefixes.
    Nodes are assigned generation_order based on when they are first created.
    """
    root = TrajectoryTreeNode(board=root_board.copy(), generation_order=0)
    order_counter = [1]  # Start from 1 since root is 0

    for trajectory in all_trajectories:
        current_node = root
        current_board = root_board.copy()

        for move in trajectory:
            if move in current_node.children:
                # Node already exists, just traverse
                current_node = current_node.children[move]
                current_board.push(move)
            else:
                # Create new node
                current_board.push(move)
                new_node = TrajectoryTreeNode(
                    board=current_board.copy(),
                    move=move,
                    parent=current_node,
                    generation_order=order_counter[0],
                )
                order_counter[0] += 1
                current_node.children[move] = new_node
                current_node = new_node

    return root


def annotate_tree_with_stockfish(
    node: TrajectoryTreeNode,
    engine: chess.engine.SimpleEngine,
    root_color: chess.Color,
    stockfish_time: float,
):
    """Recursively annotate all nodes in the tree with Stockfish evaluations."""
    try:
        node.stockfish_value = stockfish_eval(node.board, engine, stockfish_time, root_color)
    except Exception as e:
        print(f"Warning: failed to eval node: {e}")
        node.stockfish_value = 0.0

    for child in node.children.values():
        annotate_tree_with_stockfish(child, engine, root_color, stockfish_time)


# --- Stockfish Pool for Parallel Evaluation -------------------------------- #

class StockfishPool:
    """
    Thread-safe pool of Stockfish engine instances for parallel evaluation.

    Usage:
        pool = StockfishPool(stockfish_path, num_engines=4)
        value = pool.evaluate(board, stockfish_time, root_color)
        pool.quit_all()

    For temporary direct engine access:
        engine = pool.get_engine()
        ...
        pool.return_engine(engine)
    """

    def __init__(self, stockfish_path: str, num_engines: int = 1):
        self._path = stockfish_path
        self._pool: _queue_module.Queue = _queue_module.Queue()
        self._engines: List[chess.engine.SimpleEngine] = []
        for _ in range(max(1, num_engines)):
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self._engines.append(engine)
            self._pool.put(engine)

    @property
    def size(self) -> int:
        return len(self._engines)

    def get_engine(self) -> chess.engine.SimpleEngine:
        """Borrow an engine from the pool (blocks until one is available)."""
        return self._pool.get()

    def return_engine(self, engine: chess.engine.SimpleEngine):
        """Return a borrowed engine back to the pool."""
        self._pool.put(engine)

    def evaluate(
        self,
        board: chess.Board,
        stockfish_time: float,
        root_color: chess.Color,
    ) -> float:
        """Evaluate a position, automatically managing engine checkout/return."""
        engine = self._pool.get()
        try:
            val = stockfish_eval(board, engine, stockfish_time, root_color)
        except Exception as e:
            print(f"Warning: stockfish eval failed: {e}")
            val = 0.0
        finally:
            self._pool.put(engine)
        return val

    def quit_all(self):
        """Quit all engines in the pool."""
        for engine in self._engines:
            try:
                engine.quit()
            except Exception:
                pass


def annotate_tree_with_stockfish_pooled(
    node: TrajectoryTreeNode,
    pool: StockfishPool,
    root_color: chess.Color,
    stockfish_time: float,
    num_threads: int = 4,
):
    """
    Annotate all tree nodes with Stockfish values in parallel using a StockfishPool.

    Collects all nodes via BFS then evaluates them concurrently with ThreadPoolExecutor.
    Each thread borrows an engine from the pool, evaluates, and returns it.
    Much faster than the recursive single-engine version for trees with many nodes.

    num_threads should not exceed pool.size to avoid blocking.
    """
    # Collect all nodes via iterative BFS (avoids recursion depth limits)
    all_nodes: List[TrajectoryTreeNode] = []
    bfs_queue: deque = deque([node])
    while bfs_queue:
        current = bfs_queue.popleft()
        all_nodes.append(current)
        for child in current.children.values():
            bfs_queue.append(child)

    def _eval_node(n: TrajectoryTreeNode):
        n.stockfish_value = pool.evaluate(n.board, stockfish_time, root_color)

    actual_threads = min(num_threads, len(all_nodes), pool.size)
    with ThreadPoolExecutor(max_workers=max(1, actual_threads)) as executor:
        list(executor.map(_eval_node, all_nodes))


def compute_trajectory_tree_stats(node: TrajectoryTreeNode) -> Dict:
    """
    Compute statistics for the trajectory tree.
    Returns dict with: node_count, leaf_count, max_depth, branching_factors.
    """
    node_count = 0
    leaf_count = 0
    max_depth = 0
    branching_factors: List[int] = []

    def _dfs(n: TrajectoryTreeNode, depth: int):
        nonlocal node_count, leaf_count, max_depth
        node_count += 1

        if not n.children:
            leaf_count += 1
            if depth > max_depth:
                max_depth = depth
        else:
            branching_factors.append(len(n.children))
            for child in n.children.values():
                _dfs(child, depth + 1)

    _dfs(node, 0)

    avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0.0

    return {
        "node_count": node_count,
        "leaf_count": leaf_count,
        "max_depth": max_depth,
        "avg_branching_factor": avg_branching,
        "max_branching_factor": max(branching_factors) if branching_factors else 0,
    }


def collect_leaf_values_from_trajectory_tree(
    node: TrajectoryTreeNode,
) -> List[float]:
    """Collect all leaf Stockfish values from the tree."""
    values: List[float] = []

    def _collect(n: TrajectoryTreeNode):
        if not n.children:
            if n.stockfish_value is not None:
                values.append(n.stockfish_value)
        else:
            for child in n.children.values():
                _collect(child)

    _collect(node)
    return values


# --- Traversal Methods ----------------------------------------------------- #

def dfs_policy_order(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
    include_directions: bool = False,
    include_subtree_sep: bool = False,
) -> str:
    """
    DFS traversal ordered by generation order (policy order).
    Children are visited in the order they were first generated.
    If include_subtree_sep is True, adds <sep> between different candidate subtrees.
    """
    tokens: List[str] = []
    board = root_board.copy()

    def _dfs(n: TrajectoryTreeNode):
        # Order children by generation order (earlier = first)
        ordered_children = sorted(n.children.items(), key=lambda x: x[1].generation_order)

        for mv, child in ordered_children:
            if include_directions:
                tokens.append("<down>")
            try:
                tokens.append(move_to_lan(board, mv))
            except Exception:
                tokens.append(mv.uci())
            board.push(mv)
            _dfs(child)
            board.pop()
            if include_directions:
                tokens.append("<up>")

    # Process root's children (candidates) with optional <sep>
    ordered_root_children = sorted(node.children.items(), key=lambda x: x[1].generation_order)

    for i, (mv, child) in enumerate(ordered_root_children):
        if include_subtree_sep and i > 0:
            tokens.append("<sep>")
        if include_directions:
            tokens.append("<down>")
        try:
            tokens.append(move_to_lan(board, mv))
        except Exception:
            tokens.append(mv.uci())
        board.push(mv)
        _dfs(child)
        board.pop()
        if include_directions:
            tokens.append("<up>")

    return " ".join(tokens)


def dfs_verifier_order(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
    include_directions: bool = False,
    include_subtree_sep: bool = False,
) -> str:
    """
    DFS traversal ordered by Stockfish value (verifier order).
    At each node, children are ordered by stockfish value (highest first), then DFS.
    If include_subtree_sep is True, adds <sep> between different candidate subtrees.
    """
    tokens: List[str] = []
    board = root_board.copy()

    def _dfs(n: TrajectoryTreeNode):
        # Order children by stockfish value (descending - best first)
        ordered_children = sorted(
            n.children.items(),
            key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
            reverse=True,
        )

        for mv, child in ordered_children:
            if include_directions:
                tokens.append("<down>")
            try:
                tokens.append(move_to_lan(board, mv))
            except Exception:
                tokens.append(mv.uci())
            board.push(mv)
            _dfs(child)
            board.pop()
            if include_directions:
                tokens.append("<up>")

    # Process root's children (candidates) with optional <sep>
    ordered_root_children = sorted(
        node.children.items(),
        key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
        reverse=True,
    )

    for i, (mv, child) in enumerate(ordered_root_children):
        if include_subtree_sep and i > 0:
            tokens.append("<sep>")
        if include_directions:
            tokens.append("<down>")
        try:
            tokens.append(move_to_lan(board, mv))
        except Exception:
            tokens.append(mv.uci())
        board.push(mv)
        _dfs(child)
        board.pop()
        if include_directions:
            tokens.append("<up>")

    return " ".join(tokens)


def bfs_policy_order(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
    include_level_markers: bool = False,
    include_subtree_sep: bool = False,
) -> str:
    """
    BFS traversal ordered by generation order within each level.
    Visits nodes level by level, with siblings ordered by generation order.
    If include_subtree_sep is True, processes each candidate subtree separately with <sep>.
    """
    if include_subtree_sep:
        # Process each candidate subtree separately
        subtree_strs: List[str] = []
        root_children = sorted(node.children.items(), key=lambda x: x[1].generation_order)

        for mv, child in root_children:
            # Create a temporary root with just this candidate
            temp_root = TrajectoryTreeNode(board=root_board.copy())
            temp_root.children[mv] = child
            subtree_str = bfs_policy_order(root_board, temp_root, include_level_markers, include_subtree_sep=False)
            subtree_strs.append(subtree_str)

        return " <sep> ".join(subtree_strs)

    tokens: List[str] = []

    # Queue entries: (node, move that led to it, parent_board)
    queue: deque = deque()

    # Add root's children to queue, ordered by generation order
    root_children = sorted(node.children.items(), key=lambda x: x[1].generation_order)
    current_level_size = len(root_children)
    next_level_size = 0

    for mv, child in root_children:
        queue.append((child, mv, root_board.copy()))

    if include_level_markers and current_level_size > 0:
        tokens.append("<level>")

    nodes_in_level = 0
    while queue:
        current, move, parent_board = queue.popleft()
        nodes_in_level += 1

        try:
            tokens.append(move_to_lan(parent_board, move))
        except Exception:
            tokens.append(move.uci())

        # Add children to queue, ordered by generation order
        current_board = parent_board.copy()
        current_board.push(move)
        ordered_children = sorted(current.children.items(), key=lambda x: x[1].generation_order)
        for mv, child in ordered_children:
            queue.append((child, mv, current_board.copy()))
            next_level_size += 1

        # Check if we've completed this level
        if nodes_in_level == current_level_size:
            if include_level_markers and next_level_size > 0:
                tokens.append("<level>")
            current_level_size = next_level_size
            next_level_size = 0
            nodes_in_level = 0

    return " ".join(tokens)


def bfs_verifier_order(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
    include_level_markers: bool = False,
    include_subtree_sep: bool = False,
) -> str:
    """
    BFS traversal ordered by Stockfish value within each level.
    Visits nodes level by level, with siblings ordered by stockfish value (highest first).
    If include_subtree_sep is True, processes each candidate subtree separately with <sep>.
    """
    if include_subtree_sep:
        # Process each candidate subtree separately
        subtree_strs: List[str] = []
        root_children = sorted(
            node.children.items(),
            key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
            reverse=True,
        )

        for mv, child in root_children:
            # Create a temporary root with just this candidate
            temp_root = TrajectoryTreeNode(board=root_board.copy())
            temp_root.children[mv] = child
            subtree_str = bfs_verifier_order(root_board, temp_root, include_level_markers, include_subtree_sep=False)
            subtree_strs.append(subtree_str)

        return " <sep> ".join(subtree_strs)

    tokens: List[str] = []

    queue: deque = deque()

    # Add root's children to queue, ordered by stockfish value
    root_children = sorted(
        node.children.items(),
        key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
        reverse=True,
    )
    current_level_size = len(root_children)
    next_level_size = 0

    for mv, child in root_children:
        queue.append((child, mv, root_board.copy()))

    if include_level_markers and current_level_size > 0:
        tokens.append("<level>")

    nodes_in_level = 0
    while queue:
        current, move, parent_board = queue.popleft()
        nodes_in_level += 1

        try:
            tokens.append(move_to_lan(parent_board, move))
        except Exception:
            tokens.append(move.uci())

        # Add children to queue, ordered by stockfish value
        current_board = parent_board.copy()
        current_board.push(move)
        ordered_children = sorted(
            current.children.items(),
            key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
            reverse=True,
        )
        for mv, child in ordered_children:
            queue.append((child, mv, current_board.copy()))
            next_level_size += 1

        # Check if we've completed this level
        if nodes_in_level == current_level_size:
            if include_level_markers and next_level_size > 0:
                tokens.append("<level>")
            current_level_size = next_level_size
            next_level_size = 0
            nodes_in_level = 0

    return " ".join(tokens)


def dfs_policy_order_with_layers(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
) -> str:
    """
    DFS traversal ordered by generation order (policy order) with layer markers.
    Each move is followed by [l{depth}] indicating its depth in the tree.
    E.g., candidate moves get [l1], their children get [l2], etc.
    """
    tokens: List[str] = []
    board = root_board.copy()

    def _dfs(n: TrajectoryTreeNode, depth: int):
        # Order children by generation order (earlier = first)
        ordered_children = sorted(n.children.items(), key=lambda x: x[1].generation_order)

        for mv, child in ordered_children:
            try:
                tokens.append(move_to_lan(board, mv))
            except Exception:
                tokens.append(mv.uci())
            tokens.append(f"[l{depth}]")
            board.push(mv)
            _dfs(child, depth + 1)
            board.pop()

    _dfs(node, 1)  # Start at depth 1 for candidate moves
    return " ".join(tokens)


def dfs_verifier_order_with_layers(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
) -> str:
    """
    DFS traversal ordered by Stockfish value (verifier order) with layer markers.
    Each move is followed by [l{depth}] indicating its depth in the tree.
    """
    tokens: List[str] = []
    board = root_board.copy()

    def _dfs(n: TrajectoryTreeNode, depth: int):
        # Order children by stockfish value (descending - best first)
        ordered_children = sorted(
            n.children.items(),
            key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
            reverse=True,
        )

        for mv, child in ordered_children:
            try:
                tokens.append(move_to_lan(board, mv))
            except Exception:
                tokens.append(mv.uci())
            tokens.append(f"[l{depth}]")
            board.push(mv)
            _dfs(child, depth + 1)
            board.pop()

    _dfs(node, 1)  # Start at depth 1 for candidate moves
    return " ".join(tokens)


def bfs_policy_order_with_layers(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
) -> str:
    """
    BFS traversal ordered by generation order within each level, with layer markers.
    Each move is followed by [l{depth}] indicating its depth in the tree.
    """
    tokens: List[str] = []

    # Queue entries: (node, move that led to it, parent_board, depth)
    queue: deque = deque()

    # Add root's children to queue, ordered by generation order
    root_children = sorted(node.children.items(), key=lambda x: x[1].generation_order)

    for mv, child in root_children:
        queue.append((child, mv, root_board.copy(), 1))  # depth 1 for candidates

    while queue:
        current, move, parent_board, depth = queue.popleft()

        try:
            tokens.append(move_to_lan(parent_board, move))
        except Exception:
            tokens.append(move.uci())
        tokens.append(f"[l{depth}]")

        # Add children to queue, ordered by generation order
        current_board = parent_board.copy()
        current_board.push(move)
        ordered_children = sorted(current.children.items(), key=lambda x: x[1].generation_order)
        for mv, child in ordered_children:
            queue.append((child, mv, current_board.copy(), depth + 1))

    return " ".join(tokens)


def bfs_verifier_order_with_layers(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
) -> str:
    """
    BFS traversal ordered by Stockfish value within each level, with layer markers.
    Each move is followed by [l{depth}] indicating its depth in the tree.
    """
    tokens: List[str] = []

    queue: deque = deque()

    # Add root's children to queue, ordered by stockfish value
    root_children = sorted(
        node.children.items(),
        key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
        reverse=True,
    )

    for mv, child in root_children:
        queue.append((child, mv, root_board.copy(), 1))  # depth 1 for candidates

    while queue:
        current, move, parent_board, depth = queue.popleft()

        try:
            tokens.append(move_to_lan(parent_board, move))
        except Exception:
            tokens.append(move.uci())
        tokens.append(f"[l{depth}]")

        # Add children to queue, ordered by stockfish value
        current_board = parent_board.copy()
        current_board.push(move)
        ordered_children = sorted(
            current.children.items(),
            key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
            reverse=True,
        )
        for mv, child in ordered_children:
            queue.append((child, mv, current_board.copy(), depth + 1))

    return " ".join(tokens)


# --- CoT Generation -------------------------------------------------------- #

def generate_traversal_cot(
    traversal_str: str,
    target_move: chess.Move,
    root_board: chess.Board,
    pgn_context: str = "",
) -> dict:
    """
    Build CoT string from a single traversal string.
    Format: <T> traversal </T> target_move
    """
    if not traversal_str or not target_move:
        return {
            "cot_format": "",
            "cot_format_with_context": "",
            "target_move_lan": "",
        }

    target_lan = move_to_lan(root_board, target_move)
    cot_format = f"<T> {traversal_str} </T> {target_lan}"
    cot_format_with_context = f"{pgn_context} {cot_format}".strip() if pgn_context else cot_format

    return {
        "cot_format": cot_format,
        "cot_format_with_context": cot_format_with_context,
        "target_move_lan": target_lan,
    }


def generate_multi_path_cot(
    traversal_paths: List[str],
    target_move: chess.Move,
    root_board: chess.Board,
    pgn_context: str = "",
) -> dict:
    """
    Build CoT string where each <sep> encloses one traversal path.
    Format: <T> <sep> path1 <sep> path2 ... <sep> </T> target_move
    """
    if not traversal_paths or not target_move:
        return {
            "cot_format": "",
            "cot_format_with_context": "",
            "target_move_lan": "",
        }

    cot_parts = ["<T>"]
    for path in traversal_paths:
        cot_parts.append("<sep>")
        cot_parts.append(path)
    cot_parts.append("<sep>")
    cot_parts.append("</T>")
    target_lan = move_to_lan(root_board, target_move)
    cot_parts.append(target_lan)
    cot_format = " ".join(cot_parts)
    cot_format_with_context = f"{pgn_context} {cot_format}".strip() if pgn_context else cot_format

    return {
        "cot_format": cot_format,
        "cot_format_with_context": cot_format_with_context,
        "target_move_lan": target_lan,
    }


def trajectory_to_string(
    root_board: chess.Board,
    trajectory: List[chess.Move],
) -> str:
    """Convert a trajectory (list of moves) to a string of LAN moves."""
    tokens: List[str] = []
    board = root_board.copy()
    for move in trajectory:
        try:
            tokens.append(move_to_lan(board, move))
        except Exception:
            tokens.append(move.uci())
        board.push(move)
    return " ".join(tokens)


def generate_trajectory_sep_cot(
    root_board: chess.Board,
    all_trajectories: List[List[chess.Move]],
    target_move: chess.Move,
    pgn_context: str = "",
) -> dict:
    """
    Build CoT string where each trajectory is separated by <sep>.
    Format: <T> <sep> traj1 <sep> traj2 ... <sep> </T> target_move

    This gives finer granularity than per-candidate separation.
    """
    if not all_trajectories or not target_move:
        return {
            "cot_format": "",
            "cot_format_with_context": "",
            "target_move_lan": "",
        }

    trajectory_strs: List[str] = []
    for trajectory in all_trajectories:
        traj_str = trajectory_to_string(root_board, trajectory)
        trajectory_strs.append(traj_str)

    cot_parts = ["<T>"]
    for traj_str in trajectory_strs:
        cot_parts.append(traj_str)
        cot_parts.append("<sep>")
    cot_parts.pop() # remove the last <sep>
    cot_parts.append("</T>")
    target_lan = move_to_lan(root_board, target_move)
    cot_parts.append(target_lan)
    cot_format = " ".join(cot_parts)
    cot_format_with_context = f"{pgn_context} {cot_format}".strip() if pgn_context else cot_format

    return {
        "cot_format": cot_format,
        "cot_format_with_context": cot_format_with_context,
        "target_move_lan": target_lan,
        "num_trajectories": len(trajectory_strs),
    }


# --- Legacy utilities for compatibility ------------------------------------ #

def order_moves_by_stockfish(
    board: chess.Board,
    legal_moves: List[chess.Move],
    engine: chess.engine.SimpleEngine,
    root_color: chess.Color,
    stockfish_time: float,
) -> List[Tuple[chess.Move, float]]:
    scored_moves: List[Tuple[chess.Move, float]] = []
    for mv in legal_moves:
        b = board.copy()
        b.push(mv)
        val = stockfish_eval(b, engine, stockfish_time, root_color)
        scored_moves.append((mv, val))
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return scored_moves


def order_moves_by_stockfish_pooled(
    board: chess.Board,
    legal_moves: List[chess.Move],
    pool: StockfishPool,
    root_color: chess.Color,
    stockfish_time: float,
    num_threads: int = 4,
) -> List[Tuple[chess.Move, float]]:
    """
    Parallel version of order_moves_by_stockfish using a StockfishPool.
    Evaluates all candidate moves concurrently via ThreadPoolExecutor.
    """
    scored_moves: List[Optional[Tuple[chess.Move, float]]] = [None] * len(legal_moves)

    def _eval_move(args_tuple):
        i, mv = args_tuple
        b = board.copy()
        b.push(mv)
        val = pool.evaluate(b, stockfish_time, root_color)
        scored_moves[i] = (mv, val)

    actual_threads = min(num_threads, len(legal_moves), pool.size)
    with ThreadPoolExecutor(max_workers=max(1, actual_threads)) as executor:
        list(executor.map(_eval_move, enumerate(legal_moves)))

    result = [s for s in scored_moves if s is not None]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def select_target_by_stockfish_direct(
    board: chess.Board,
    candidate_moves: List[chess.Move],
    engine: chess.engine.SimpleEngine,
    stockfish_time: float,
) -> Optional[chess.Move]:
    """
    Select the best candidate move via a single Stockfish multipv call on the root board.

    Runs multipv analysis over all legal moves and returns the highest-ranked move
    that appears in candidate_moves. Falls back to the first candidate on failure.

    This is an alternative to tree-based selection: it is a single fresh Stockfish
    call rather than relying on the already-annotated tree values.
    """
    if not candidate_moves:
        return None
    candidate_set = set(candidate_moves)
    try:
        # Analyse all legal moves so we can find whichever candidate ranks highest
        num_legal = len(list(board.legal_moves))
        info_list = engine.analyse(
            board,
            chess.engine.Limit(time=stockfish_time),
            multipv=num_legal,
        )
        # info_list is sorted best-to-worst
        for info in info_list:
            pv = info.get("pv", [])
            if pv and pv[0] in candidate_set:
                return pv[0]
    except Exception as e:
        print(f"Warning: direct stockfish target selection failed: {e}")
    # Fallback: return first candidate
    return candidate_moves[0]


# --- Per-position processing (extracted for parallelism) ------------------- #

def _process_single_position(
    idx: int,
    row: "pd.Series",
    args,
    sampling_policy,
    stockfish_pool: StockfishPool,
    has_ground_truth: bool,
    moves_col: str,
) -> Optional[dict]:
    """
    Process a single chess position and generate CoT data.

    Returns a dict with keys:
        'result'    – the main result dict to append to results list
        'cot_entry' – dict for cot_output_path (or None if not requested)
        'is_match'  – bool or None
        'has_gt'    – bool (True if ground truth was available for this row)
    Returns None if the position should be skipped entirely.
    """
    pgn_str = row["ctx"]
    if pgn_str is np.nan or pgn_str is None:
        print(f"Skipping position {idx} because it has no PGN string")
        return None

    ground_truth_move = None
    is_match = None
    actual_pgn = pgn_str
    moves_list: List[str] = []
    row_has_gt = False

    if has_ground_truth and pd.notna(row[moves_col]):
        moves_str = str(row[moves_col]).strip()
        moves_list = moves_str.split()
        if len(moves_list) >= 2:
            row_has_gt = True
            first_move_san = moves_list[0]
            ground_truth_san = moves_list[1]
            ground_truth_move = ground_truth_san
            try:
                game = chess.pgn.read_game(io.StringIO(pgn_str))
                board = game.end().board() if game is not None else chess.Board()
                opp_move = board.parse_san(first_move_san)
                board.push(opp_move)
                new_game = chess.pgn.Game.from_board(board)
                exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
                new_pgn = new_game.accept(exporter)
                new_pgn = re.sub(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$", "", new_pgn).strip()
                actual_pgn = new_pgn
            except Exception as e:
                print(f"Error applying first move for position {idx}: {e}")
                return None
        else:
            print(f"Skipping position {idx}: not enough moves in Moves column")
            return None

    try:
        game = chess.pgn.read_game(io.StringIO(actual_pgn))
        board = game.end().board() if game is not None else chess.Board()
        root_color = board.turn

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            print(f"Position {idx} has no legal moves (game over)")
            return None

        # Step 1: Generate candidates
        if args.candidate_selection == "policy":
            candidate_moves = sample_candidates_with_policy(
                board=board,
                sampling_policy=sampling_policy,
                num_candidates=args.num_candidates,
            )
        else:
            # Order by stockfish and take top candidates
            if args.stockfish_pool_size > 1:
                scored_moves = order_moves_by_stockfish_pooled(
                    board, legal_moves, stockfish_pool, root_color,
                    args.stockfish_time, args.stockfish_pool_size,
                )
            else:
                engine = stockfish_pool.get_engine()
                try:
                    scored_moves = order_moves_by_stockfish(
                        board, legal_moves, engine, root_color, args.stockfish_time,
                    )
                finally:
                    stockfish_pool.return_engine(engine)
            candidate_moves = [mv for mv, _ in scored_moves[: min(args.num_candidates, len(scored_moves))]]

        if not candidate_moves:
            print(f"Position {idx}: no candidates generated")
            return None

        # Step 2: Generate trajectories for each candidate
        all_trajectories: List[List[chess.Move]] = []
        trajectories_per_candidate: Dict[chess.Move, List[List[chess.Move]]] = {}

        if args.num_trajectories <= 0 or args.trajectory_depth <= 1:
            # No subtree mode: each candidate is a "trajectory" of length 1
            for cand_move in candidate_moves:
                trajectories_per_candidate[cand_move] = [[cand_move]]
                all_trajectories.append([cand_move])
        else:
            # Parallel trajectory generation across candidates when possible.
            # Stockfish sampling policy uses a shared engine object that is NOT
            # thread-safe, so we skip threading for it.
            use_parallel_traj = (
                args.trajectory_threads > 1
                and len(candidate_moves) > 1
                and args.sampling_policy not in ("stockfish",)
            )

            if use_parallel_traj:
                def _gen_for_candidate(cand_move):
                    trajs = generate_trajectories_for_candidate(
                        root_board=board,
                        candidate_move=cand_move,
                        sampling_policy=sampling_policy,
                        num_trajectories=args.num_trajectories,
                        trajectory_depth=args.trajectory_depth,
                    )
                    return cand_move, trajs

                n_threads = min(args.trajectory_threads, len(candidate_moves))
                with ThreadPoolExecutor(max_workers=n_threads) as executor:
                    for cand_move, cand_trajs in executor.map(_gen_for_candidate, candidate_moves):
                        trajectories_per_candidate[cand_move] = cand_trajs
                        all_trajectories.extend(cand_trajs)
            else:
                for cand_move in candidate_moves:
                    cand_trajectories = generate_trajectories_for_candidate(
                        root_board=board,
                        candidate_move=cand_move,
                        sampling_policy=sampling_policy,
                        num_trajectories=args.num_trajectories,
                        trajectory_depth=args.trajectory_depth,
                    )
                    trajectories_per_candidate[cand_move] = cand_trajectories
                    all_trajectories.extend(cand_trajectories)

        # Step 3: Build tree from all trajectories
        tree_root = build_tree_from_trajectories(
            root_board=board,
            all_trajectories=all_trajectories,
        )

        # Step 4: Annotate tree with Stockfish values
        if args.stockfish_pool_size > 1:
            annotate_tree_with_stockfish_pooled(
                node=tree_root,
                pool=stockfish_pool,
                root_color=root_color,
                stockfish_time=args.stockfish_time,
                num_threads=args.stockfish_pool_size,
            )
        else:
            engine = stockfish_pool.get_engine()
            try:
                annotate_tree_with_stockfish(
                    node=tree_root,
                    engine=engine,
                    root_color=root_color,
                    stockfish_time=args.stockfish_time,
                )
            finally:
                stockfish_pool.return_engine(engine)

        # Compute tree statistics
        tree_stats = compute_trajectory_tree_stats(tree_root)

        # Collect leaf values for determining best move
        leaf_values = collect_leaf_values_from_trajectory_tree(tree_root)

        # Compute values per candidate (max leaf value in each candidate's subtree)
        move_values: Dict[chess.Move, float] = {}
        direct_move_values: Dict[chess.Move, float] = {}
        candidate_details: List[Dict] = []

        for cand_move in candidate_moves:
            if cand_move in tree_root.children:
                cand_node = tree_root.children[cand_move]
                cand_leaf_values = collect_leaf_values_from_trajectory_tree(cand_node)
                best_leaf_val = max(cand_leaf_values) if cand_leaf_values else 0.0
                direct_val = cand_node.stockfish_value if cand_node.stockfish_value is not None else 0.0

                move_values[cand_move] = best_leaf_val
                direct_move_values[cand_move] = direct_val

                cand_tree_stats = compute_trajectory_tree_stats(cand_node)
                delta = best_leaf_val - direct_val
                ratio = (best_leaf_val / direct_val) if direct_val not in (0, None) else None

                candidate_details.append({
                    "move": cand_move.uci(),
                    "move_san": board.san(cand_move),
                    "direct_value": float(direct_val),
                    "best_leaf_value": float(best_leaf_val),
                    "leaf_values": [float(v) for v in cand_leaf_values],
                    "leaf_count": cand_tree_stats["leaf_count"],
                    "max_depth": cand_tree_stats["max_depth"],
                    "node_count": cand_tree_stats["node_count"],
                    "num_trajectories": len(trajectories_per_candidate.get(cand_move, [])),
                    "leaf_vs_direct_delta": float(delta),
                    "leaf_vs_direct_ratio": float(ratio) if ratio is not None else None,
                })

        # Step 4b: Determine target move
        # --final_move_selection controls how the target move is chosen:
        #   "tree"      – use tree-annotated Stockfish values (existing behaviour)
        #   "stockfish" – call Stockfish directly on the root board (single multipv call)
        if args.final_move_selection == "stockfish":
            engine = stockfish_pool.get_engine()
            try:
                target_move = select_target_by_stockfish_direct(
                    board, candidate_moves, engine, args.stockfish_time
                )
            finally:
                stockfish_pool.return_engine(engine)
            best_val = direct_move_values.get(target_move, 0.0) if target_move else None
        else:
            # "tree" mode: existing behaviour
            if args.num_trajectories <= 0 or args.trajectory_depth <= 1:
                # direct-only
                target_move = max(direct_move_values.keys(), key=lambda m: direct_move_values[m])
                best_val = direct_move_values[target_move]
            else:
                if args.evaluation_policy == "stockfish_leaf_max":
                    target_move = max(move_values.keys(), key=lambda m: move_values[m]) if move_values else None
                    best_val = move_values[target_move] if move_values else None
                elif args.evaluation_policy == "stockfish_direct_max":
                    target_move = max(direct_move_values.keys(), key=lambda m: direct_move_values[m])
                    best_val = direct_move_values[target_move] if direct_move_values else None
                else:
                    raise ValueError(f"Unknown evaluation policy: {args.evaluation_policy}")

        target_move_san = board.san(target_move) if target_move else None

        # Step 5: Generate traversals with different methods
        traversals: Dict[str, Dict[str, str]] = {}

        for method in args.traversal_methods:
            if method == "dfs_policy":
                trav = dfs_policy_order(board, tree_root, include_directions=False)
                trav_dirs = dfs_policy_order(board, tree_root, include_directions=True)
                trav_sep = dfs_policy_order(board, tree_root, include_directions=False, include_subtree_sep=True)
            elif method == "dfs_verifier":
                trav = dfs_verifier_order(board, tree_root, include_directions=False)
                trav_dirs = dfs_verifier_order(board, tree_root, include_directions=True)
                trav_sep = dfs_verifier_order(board, tree_root, include_directions=False, include_subtree_sep=True)
            elif method == "bfs_policy":
                trav = bfs_policy_order(board, tree_root, include_level_markers=False)
                trav_dirs = bfs_policy_order(board, tree_root, include_level_markers=True)
                trav_sep = bfs_policy_order(board, tree_root, include_level_markers=False, include_subtree_sep=True)
            elif method == "bfs_verifier":
                trav = bfs_verifier_order(board, tree_root, include_level_markers=False)
                trav_dirs = bfs_verifier_order(board, tree_root, include_level_markers=True)
                trav_sep = bfs_verifier_order(board, tree_root, include_level_markers=False, include_subtree_sep=True)
            elif method == "dfs_policy_layers":
                trav = dfs_policy_order(board, tree_root, include_directions=False)
                trav_dirs = dfs_policy_order_with_layers(board, tree_root)
                trav_sep = dfs_policy_order(board, tree_root, include_directions=False, include_subtree_sep=True)
            elif method == "dfs_verifier_layers":
                trav = dfs_verifier_order(board, tree_root, include_directions=False)
                trav_dirs = dfs_verifier_order_with_layers(board, tree_root)
                trav_sep = dfs_verifier_order(board, tree_root, include_directions=False, include_subtree_sep=True)
            elif method == "bfs_policy_layers":
                trav = bfs_policy_order(board, tree_root, include_level_markers=False)
                trav_dirs = bfs_policy_order_with_layers(board, tree_root)
                trav_sep = bfs_policy_order(board, tree_root, include_level_markers=False, include_subtree_sep=True)
            elif method == "bfs_verifier_layers":
                trav = bfs_verifier_order(board, tree_root, include_level_markers=False)
                trav_dirs = bfs_verifier_order_with_layers(board, tree_root)
                trav_sep = bfs_verifier_order(board, tree_root, include_level_markers=False, include_subtree_sep=True)
            else:
                print(f"Warning: Unknown traversal method '{method}', skipping")
                continue

            traversals[method] = {
                "traversal": trav,
                "traversal_with_markers": trav_dirs,
                "traversal_with_subtree_sep": trav_sep,
            }

        # Generate CoT for each traversal method
        cot_results: Dict[str, Dict] = {}
        for method, trav_data in traversals.items():
            cot = generate_traversal_cot(
                traversal_str=trav_data["traversal_with_subtree_sep"],
                target_move=target_move,
                root_board=board,
                pgn_context=actual_pgn,
            )
            cot_with_markers = generate_traversal_cot(
                traversal_str=trav_data["traversal_with_markers"],
                target_move=target_move,
                root_board=board,
                pgn_context=actual_pgn,
            )
            cot_results[method] = {
                "cot_format": cot["cot_format"],
                "cot_format_with_context": cot["cot_format_with_context"],
                "cot_format_with_markers": cot_with_markers["cot_format"],
                "cot_format_with_markers_and_context": cot_with_markers["cot_format_with_context"],
                "target_move_lan": cot["target_move_lan"],
            }

        # Per-trajectory CoT (each trajectory is a <sep> block - finer granularity)
        trajectory_sep_cot = generate_trajectory_sep_cot(
            root_board=board,
            all_trajectories=all_trajectories,
            target_move=target_move,
            pgn_context=actual_pgn,
        )

        # Check against ground truth
        if has_ground_truth and ground_truth_move:
            if target_move:
                try:
                    gt_parsed = board.parse_san(ground_truth_move)
                    is_match = target_move == gt_parsed
                except ValueError:
                    if target_move_san:
                        is_match = target_move_san == ground_truth_move
            else:
                is_match = False

        # Build result dict
        move_values_uci = {mv.uci(): float(val) for mv, val in move_values.items()} if move_values else {}
        direct_move_values_uci = {mv.uci(): float(val) for mv, val in direct_move_values.items()} if direct_move_values else {}

        result = {
            "index": idx,
            "original_pgn": pgn_str,
            "pgn": actual_pgn,
            "opponent_move": moves_list[0] if (has_ground_truth and pd.notna(row[moves_col]) and len(moves_list) >= 2) else None,
            "target_move": target_move.uci() if target_move else None,
            "target_move_san": target_move_san,
            "target_move_lan": cot_results.get(args.traversal_methods[0], {}).get("target_move_lan", ""),
            "best_val": float(best_val) if best_val is not None else None,
            "move_values": move_values_uci,
            "direct_move_values": direct_move_values_uci,
            "ground_truth_move": ground_truth_move,
            "is_match": is_match,
            "num_candidates": len(candidate_moves),
            "num_trajectories_total": len(all_trajectories),
            # Per-traversal-method CoT (now uses subtree <sep> by default)
            "cot_by_method": cot_results,
            # Per-trajectory CoT (each individual trajectory separated by <sep>)
            "trajectory_sep_cot": trajectory_sep_cot,
            # Raw traversals
            "traversals": traversals,
            # Candidate details
            "candidate_details": candidate_details,
            # Tree statistics
            "tree_stats": tree_stats,
        }

        # Backward compatibility: add primary cot_format fields using first traversal method
        primary_method = args.traversal_methods[0] if args.traversal_methods else "dfs_policy"
        if primary_method in cot_results:
            result["cot_format"] = cot_results[primary_method]["cot_format"]
            result["cot_format_with_context"] = cot_results[primary_method]["cot_format_with_context"]
            result["cot_format_with_directions"] = cot_results[primary_method].get("cot_format_with_markers", "")
            result["cot_format_with_directions_and_context"] = cot_results[primary_method].get("cot_format_with_markers_and_context", "")

        # Build optional cot_entry for cot_output_path
        cot_entry = None
        if args.cot_output_path:
            cot_entry = {
                "index": idx,
                "pgn": actual_pgn,
                "target_move_san": target_move_san,
                "target_move_lan": result.get("target_move_lan", ""),
            }
            for method, cot_data in cot_results.items():
                cot_entry[f"cot_{method}"] = cot_data["cot_format_with_context"]
                cot_entry[f"cot_{method}_with_markers"] = cot_data["cot_format_with_markers_and_context"]
            cot_entry["cot_trajectory_sep"] = trajectory_sep_cot["cot_format_with_context"]

        return {
            "result": result,
            "cot_entry": cot_entry,
            "is_match": is_match,
            "has_gt": row_has_gt,
        }

    except Exception as e:
        print(f"Error processing position {idx}: {e}")
        traceback.print_exc()
        return None


# --- Multiprocessing worker support ---------------------------------------- #
# Module-level globals are initialised once per worker process by _worker_init.

_g_args = None
_g_sampling_policy = None
_g_stockfish_pool: Optional[StockfishPool] = None


def _worker_init(args_dict: dict):
    """Initialise per-worker resources (called once per worker process)."""
    global _g_args, _g_sampling_policy, _g_stockfish_pool
    import argparse as _argparse
    _g_args = _argparse.Namespace(**args_dict)
    device = torch.device(_g_args.device)
    _g_sampling_policy = create_sampling_policy(_g_args, device)
    pool_size = max(1, getattr(_g_args, "stockfish_pool_size", 1))
    _g_stockfish_pool = StockfishPool(_g_args.stockfish_path, pool_size)


def _worker_process_row(task: tuple) -> Optional[dict]:
    """Process one position inside a worker process."""
    idx, row_dict, has_ground_truth, moves_col = task
    row = pd.Series(row_dict)
    try:
        return _process_single_position(
            idx=idx,
            row=row,
            args=_g_args,
            sampling_policy=_g_sampling_policy,
            stockfish_pool=_g_stockfish_pool,
            has_ground_truth=has_ground_truth,
            moves_col=moves_col,
        )
    except Exception as e:
        print(f"Worker error at position {idx}: {e}")
        traceback.print_exc()
        return None


# --- Main pipeline -------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate CoT data using trajectory-based tree with multiple traversal orders")

    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV file containing positions")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generated CoT data (JSON)")
    parser.add_argument("--cot_output_path", type=str, default=None, help="Path to save CoT format file (optional)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--num_skip_samples", type=int, default=None, help="Number of samples to skip (default: none)")

    parser.add_argument("--sampling_policy", type=str, choices=["random", "stockfish", "model", "hf_model"], default="random")
    parser.add_argument("--evaluation_policy", type=str, choices=["stockfish_leaf_max", "stockfish_direct_max"], default="stockfish_direct_max")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidate moves to explore")
    parser.add_argument("--num_trajectories", type=int, default=3, help="Number of trajectories per candidate")
    parser.add_argument("--trajectory_depth", type=int, default=4, help="Maximum depth for each trajectory")
    parser.add_argument("--data_name", type=str, default="puzzle", help="Name of the data")
    parser.add_argument(
        "--candidate_selection",
        type=str,
        choices=["policy", "stockfish"],
        default="policy",
        help="How to select candidate moves: 'policy' samples from policy, 'stockfish' orders by stockfish",
    )
    parser.add_argument(
        "--traversal_methods",
        type=str,
        nargs="+",
        default=["dfs_policy", "dfs_verifier", "bfs_policy", "bfs_verifier"],
        help="Traversal methods to use: dfs_policy, dfs_verifier, bfs_policy, bfs_verifier, "
             "dfs_policy_layers, dfs_verifier_layers, bfs_policy_layers, bfs_verifier_layers",
    )

    parser.add_argument("--stockfish_path", type=str, default="/usr/games/stockfish")
    parser.add_argument("--stockfish_time", type=float, default=0.01)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # --- New parallelism / efficiency flags --------------------------------- #
    parser.add_argument(
        "--final_move_selection",
        type=str,
        choices=["tree", "stockfish"],
        default="tree",
        help=(
            "How to select the final target move. "
            "'tree': use Stockfish values from the annotated tree (existing behaviour, "
            "respects --evaluation_policy). "
            "'stockfish': call Stockfish directly on the root position via multipv to pick "
            "the best candidate (single fresh call, independent of tree annotation)."
        ),
    )
    parser.add_argument(
        "--stockfish_pool_size",
        type=int,
        default=1,
        help=(
            "Number of Stockfish engine instances to run in parallel for tree annotation "
            "and candidate evaluation. Values > 1 enable parallel annotation via "
            "ThreadPoolExecutor (recommended: 4-8). Each engine is a separate subprocess."
        ),
    )
    parser.add_argument(
        "--trajectory_threads",
        type=int,
        default=1,
        help=(
            "Number of threads for parallel trajectory generation across candidates. "
            "Values > 1 speed up random and model policies by generating each candidate's "
            "trajectories concurrently. Automatically disabled for the stockfish sampling "
            "policy (its engine is not thread-safe)."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes for position-level parallelism. "
            "Values > 1 process multiple positions simultaneously using multiprocessing. "
            "Each worker creates its own Stockfish pool and sampling policy. "
            "Not supported for model/hf_model sampling policies (GPU memory constraints); "
            "falls back to 1 worker automatically."
        ),
    )

    args = parser.parse_args()

    print(f"Reading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    if "ctx" not in df.columns:
        raise ValueError("CSV file must contain 'ctx' column with PGN positions")

    has_ground_truth = "Moves" in df.columns or "moves" in df.columns
    if has_ground_truth:
        moves_col = "Moves" if "Moves" in df.columns else "moves"
        print(f"Found ground truth moves in '{moves_col}' column - will compare generated moves")
    else:
        moves_col = ""

    if args.num_samples is not None:
        if args.num_skip_samples is None:
            df = df.head(args.num_samples)
        else:
            df = df.iloc[args.num_skip_samples : args.num_skip_samples + args.num_samples]

    print(f"Processing {len(df)} positions...")

    # Guard: multi-process doesn't work for GPU model policies
    if args.num_workers > 1 and args.sampling_policy in ("model", "hf_model"):
        print(
            "Warning: --num_workers > 1 is not supported for model/hf_model sampling policies "
            "(GPU memory / CUDA fork constraints). Falling back to --num_workers 1."
        )
        args.num_workers = 1

    device = torch.device(args.device)
    sampling_policy = create_sampling_policy(args, device)
    stockfish_pool = StockfishPool(args.stockfish_path, args.stockfish_pool_size)

    print(f"Using trajectory-based tree generator:")
    print(f"  Candidate selection: {args.candidate_selection}")
    print(f"  Num candidates: {args.num_candidates}")
    print(f"  Trajectories per candidate: {args.num_trajectories}")
    print(f"  Trajectory depth: {args.trajectory_depth}")
    print(f"  Traversal methods: {args.traversal_methods}")
    print(f"  Final move selection: {args.final_move_selection}")
    print(f"  Stockfish pool size: {args.stockfish_pool_size}")
    print(f"  Trajectory threads: {args.trajectory_threads}")
    print(f"  Num workers: {args.num_workers}")

    results = []
    cot_format_outputs = []
    matches = 0
    total_with_ground_truth = 0

    def _collect_worker_result(wr):
        """Accumulate a single worker result into the outer lists/counters."""
        nonlocal matches, total_with_ground_truth
        if wr is None:
            return
        results.append(wr["result"])
        if wr["has_gt"] and wr["is_match"] is not None:
            total_with_ground_truth += 1
            if wr["is_match"]:
                matches += 1
        if args.cot_output_path and wr.get("cot_entry"):
            cot_format_outputs.append(wr["cot_entry"])

    if args.num_workers > 1:
        import multiprocessing
        # Pass args as a plain dict so it can be pickled across processes
        args_dict = vars(args)
        tasks = [
            (idx, row.to_dict(), has_ground_truth, moves_col)
            for idx, row in df.iterrows()
        ]
        with multiprocessing.Pool(
            processes=args.num_workers,
            initializer=_worker_init,
            initargs=[args_dict],
        ) as pool:
            for wr in tqdm(
                pool.imap(_worker_process_row, tasks),
                total=len(tasks),
                desc="Generating CoT data (parallel)",
            ):
                _collect_worker_result(wr)
    else:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating CoT data"):
            wr = _process_single_position(
                idx=idx,
                row=row,
                args=args,
                sampling_policy=sampling_policy,
                stockfish_pool=stockfish_pool,
                has_ground_truth=has_ground_truth,
                moves_col=moves_col,
            )
            _collect_worker_result(wr)

    stockfish_pool.quit_all()

    from pathlib import Path

    base_output_path = Path(args.output_path)

    # 1) hyperparameters -> directory name
    hyperparam_dir = (
        f"{args.sampling_policy}"
        f"_seed{args.seed}"
        f"_eval{args.evaluation_policy}"
        f"_cand{args.candidate_selection}"
        f"_nc{args.num_candidates}"
        f"_nt{args.num_trajectories}"
        f"_d{args.trajectory_depth}"
        f"_temp{args.temperature}"
        f"_sf{args.stockfish_time}"
    )

    # 2) shard -> filename
    if args.num_samples is not None and args.num_skip_samples is not None:
        shard_name = f"data{args.data_name}_shard{args.num_skip_samples}-{args.num_skip_samples + args.num_samples}.json"
    elif args.num_samples is not None:
        shard_name = f"data{args.data_name}_n{args.num_samples}.json"
    else:
        shard_name = f"data{args.data_name}_results.json"

    # 3) decide base directory cleanly
    if base_output_path.suffix:
        # e.g. /path/to/foo.json → /path/to/foo/<hyperparam>/shard.json
        base_dir = base_output_path.parent / base_output_path.stem
    else:
        # e.g. /path/to/foo → /path/to/foo/<hyperparam>/shard.json
        base_dir = base_output_path

    # 4) final output path
    output_path = base_dir / hyperparam_dir / shard_name

    print(f"Saving results to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)


    # Compute summary statistics
    total_trajectories = sum(r["num_trajectories_total"] for r in results)
    summary = {
        "total_positions": len(results),
        "total_trajectories": total_trajectories,
        "avg_trajectories_per_position": total_trajectories / len(results) if results else 0,
        "traversal_methods": args.traversal_methods,
    }

    # Aggregate tree stats
    node_counts = [r["tree_stats"]["node_count"] for r in results if "tree_stats" in r]
    leaf_counts = [r["tree_stats"]["leaf_count"] for r in results if "tree_stats" in r]
    depths = [r["tree_stats"]["max_depth"] for r in results if "tree_stats" in r]
    branching_factors = [r["tree_stats"]["avg_branching_factor"] for r in results if "tree_stats" in r]

    if node_counts:
        summary["avg_node_count"] = sum(node_counts) / len(node_counts)
    if leaf_counts:
        summary["avg_leaf_count"] = sum(leaf_counts) / len(leaf_counts)
    if depths:
        summary["avg_max_depth"] = sum(depths) / len(depths)
    if branching_factors:
        summary["avg_branching_factor"] = sum(branching_factors) / len(branching_factors)

    # Leaf vs direct stats
    deltas = []
    ratios = []
    for r in results:
        for cand in r.get("candidate_details", []):
            if cand.get("leaf_vs_direct_delta") is not None:
                deltas.append(cand["leaf_vs_direct_delta"])
            if cand.get("leaf_vs_direct_ratio") is not None:
                ratios.append(cand["leaf_vs_direct_ratio"])
    if deltas:
        summary["leaf_vs_direct_delta_avg"] = sum(deltas) / len(deltas)
    if ratios:
        summary["leaf_vs_direct_ratio_avg"] = sum(ratios) / len(ratios)

    if has_ground_truth and total_with_ground_truth > 0:
        accuracy = matches / total_with_ground_truth
        summary["ground_truth_comparison"] = {
            "total_compared": total_with_ground_truth,
            "matches": matches,
            "accuracy": accuracy,
            "accuracy_percent": f"{accuracy * 100:.2f}%",
        }

    output_data = {
        "config": {
            "sampling_policy": args.sampling_policy,
            "candidate_selection": args.candidate_selection,
            "evaluation_policy": args.evaluation_policy,
            "final_move_selection": args.final_move_selection,
            "num_candidates": args.num_candidates,
            "num_trajectories": args.num_trajectories,
            "trajectory_depth": args.trajectory_depth,
            "traversal_methods": args.traversal_methods,
            "temperature": args.temperature,
            "stockfish_time": args.stockfish_time,
            "stockfish_pool_size": args.stockfish_pool_size,
            "trajectory_threads": args.trajectory_threads,
            "num_workers": args.num_workers,
            "note": "Trajectory-based tree with multiple traversal orderings",
        },
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    if args.cot_output_path and cot_format_outputs:
        base_path = Path(args.cot_output_path)
        # Use the same hyperparam_dir and shard_name format as main output
        hyperparam_str = f"_{hyperparam_dir}" if hyperparam_dir else ""
        shard_str = f"_{shard_name.replace('.json', '')}" if shard_name else ""
        if base_path.suffix:
            cot_output_path = base_path.parent / f"{base_path.stem}{hyperparam_str}{shard_str}{base_path.suffix}"
        else:
            cot_output_path = base_path / f"cot_output{hyperparam_str}{shard_str}.txt"
        print(f"Saving CoT format to {cot_output_path}...")
        cot_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cot_output_path, "w") as f:
            for entry in cot_format_outputs:
                f.write(f"# Position {entry['index']}\n")
                f.write(f"# PGN: {entry['pgn']}\n")
                f.write(f"# Target (SAN): {entry['target_move_san']}\n")
                f.write(f"# Target (LAN): {entry['target_move_lan']}\n")
                for method in args.traversal_methods:
                    f.write(f"# CoT ({method} with subtree <sep>):\n")
                    f.write(f"{entry.get(f'cot_{method}', '')}\n")
                    f.write(f"# CoT ({method} with markers):\n")
                    f.write(f"{entry.get(f'cot_{method}_with_markers', '')}\n")
                # Write per-trajectory CoT (each individual trajectory is a <sep> block)
                f.write(f"# CoT (trajectory_sep - per trajectory <sep>):\n")
                f.write(f"{entry.get('cot_trajectory_sep', '')}\n")
                f.write("\n")

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully generated CoT data for {len(results)} positions")
    print(f"Total trajectories generated: {summary['total_trajectories']}")
    print(f"Average trajectories per position: {summary['avg_trajectories_per_position']:.2f}")
    print(f"\nTree Statistics:")
    if "avg_node_count" in summary:
        print(f"  Average node count: {summary['avg_node_count']:.2f}")
    if "avg_leaf_count" in summary:
        print(f"  Average leaf count: {summary['avg_leaf_count']:.2f}")
    if "avg_max_depth" in summary:
        print(f"  Average max depth: {summary['avg_max_depth']:.2f}")
    if "avg_branching_factor" in summary:
        print(f"  Average branching factor: {summary['avg_branching_factor']:.2f}")
    if has_ground_truth and total_with_ground_truth > 0:
        print("\nGround Truth Comparison:")
        print(f"  Positions compared: {total_with_ground_truth}")
        print(f"  Matches: {matches}")
        print(f"  Accuracy: {summary['ground_truth_comparison']['accuracy_percent']}")
    if "leaf_vs_direct_delta_avg" in summary:
        print(f"\nLeaf vs direct delta (avg): {summary['leaf_vs_direct_delta_avg']:.2f}")
    if "leaf_vs_direct_ratio_avg" in summary:
        print(f"Leaf vs direct ratio (avg): {summary['leaf_vs_direct_ratio_avg']:.3f}")
    print(f"\nTraversal methods: {', '.join(args.traversal_methods)}")
    print(f"\nResults saved to {output_path}")
    if args.cot_output_path:
        print(f"CoT format saved to {cot_output_path if 'cot_output_path' in locals() else args.cot_output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
