# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import random

DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

if DEBUG:
    print(f"REWARD_MODEL_TYPE: {os.environ.get('REWARD_MODEL_TYPE', 'NOT_SET')}")

REWARD_MODEL_TYPE = os.environ['REWARD_MODEL_TYPE'].upper()


# =============================================================================
# FORMAT VALIDATION (for RANDOM_REWARD and RLVR_FORMAT)
# =============================================================================

def format_validity(solution_str):
    """
    Check if solution has valid format:
    - Exactly one \\boxed{} or \\\\boxed{}
    """
    try:
        boxed_patterns = [r'\\\\boxed\{.*?\}', r'\\boxed\{.*?\}']
        total_boxed = sum(len(re.findall(pattern, solution_str)) for pattern in boxed_patterns)
        if total_boxed != 1:
            return 0.0
        
        return 1.0
    
    except Exception as e:
        return 0.0


def thinking_format_validity(solution_str):
    """
    Check if solution has valid thinking format:
    - Must have <think>...</think> tags
    - Must have at least one \\boxed{} or \\\\boxed{} AFTER </think>
    
    Returns 1.0 if valid, 0.0 if invalid.
    """
    try:
        # Check if <think> and </think> tags exist
        if '<think>' not in solution_str or '</think>' not in solution_str:
            if DEBUG:
                print("THINKING_FORMAT INVALID: Missing <think>...</think> tags")
            return 0.0
        
        # Get content after </think>
        think_end_pos = solution_str.find('</think>')
        content_after_think = solution_str[think_end_pos + len('</think>'):]
        
        # Check if there's at least one boxed answer in the content after </think>
        boxed_patterns = [r'\\\\boxed\{.*?\}', r'\\boxed\{.*?\}']
        total_boxed = sum(len(re.findall(pattern, content_after_think)) for pattern in boxed_patterns)
        
        if total_boxed == 0:
            if DEBUG:
                print("THINKING_FORMAT INVALID: No \\boxed{} found after </think>")
            return 0.0
        
        if DEBUG:
            print(f"THINKING_FORMAT VALID: Found <think>...</think> tags and {total_boxed} \\boxed{{}} after </think>")
        
        return 1.0
    
    except Exception as e:
        if DEBUG:
            print(f"THINKING_FORMAT ERROR: Exception during validation: {e}")
        return 0.0


def deepseek_thinking_format_validity(solution_str):
    """
    Check if solution has valid DeepSeek thinking format:
    - Only checks for </think> tag (because <think> is provided in prompt)
    - Must have at least one \\boxed{} or \\\\boxed{} AFTER </think>
    
    Returns 1.0 if valid, 0.0 if invalid.
    """
    try:
        # Check if </think> tag exists (no need to check <think> since it's in the prompt)
        if '</think>' not in solution_str:
            if DEBUG:
                print("DEEPSEEK_THINKING_FORMAT INVALID: Missing </think> tag")
            return 0.0
        
        # Get content after </think>
        think_end_pos = solution_str.find('</think>')
        content_after_think = solution_str[think_end_pos + len('</think>'):]
        
        # Check if there's at least one boxed answer in the content after </think>
        boxed_patterns = [r'\\\\boxed\{.*?\}', r'\\boxed\{.*?\}']
        total_boxed = sum(len(re.findall(pattern, content_after_think)) for pattern in boxed_patterns)
        
        if total_boxed == 0:
            if DEBUG:
                print("DEEPSEEK_THINKING_FORMAT INVALID: No \\boxed{} found after </think>")
            return 0.0
        
        if DEBUG:
            print(f"DEEPSEEK_THINKING_FORMAT VALID: Found </think> tag and {total_boxed} \\boxed{{}} after </think>")
        
        return 1.0
    
    except Exception as e:
        if DEBUG:
            print(f"DEEPSEEK_THINKING_FORMAT ERROR: Exception during validation: {e}")
        return 0.0


def find_balanced_boxed(text):
    """
    Find the position of \\boxed{...} with balanced braces.
    Returns (start, end) positions of all boxed expressions.
    """
    boxed_positions = []
    
    # Look for both \\boxed and \\\\boxed patterns
    patterns = [r'\\\\boxed\{', r'\\boxed\{']
    
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            start = match.start()
            brace_start = match.end() - 1  # Position of the opening {
            
            # Find the matching closing brace
            brace_count = 1
            pos = brace_start + 1
            
            while pos < len(text) and brace_count > 0:
                if text[pos] == '{':
                    brace_count += 1
                elif text[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            if brace_count == 0:
                # Found matching closing brace
                end = pos
                boxed_positions.append((start, end))
    
    return boxed_positions


def rlvr_format_validity(solution_str):
    """
    RLVR_FORMAT validation: Strict format checking for mathematical solutions.
    
    Rules:
    1. Must have exactly one \\boxed{} or \\\\boxed{}  
    2. No content after the boxed answer (only whitespace allowed)
    
    Returns 1.0 if valid, 0.0 if invalid.
    """
    try:
        # Find all boxed expressions with balanced braces
        boxed_positions = find_balanced_boxed(solution_str)
        
        if len(boxed_positions) != 1:
            if DEBUG:
                print(f"RLVR_FORMAT INVALID: Found {len(boxed_positions)} boxed expressions (required exactly 1)")
            return 0.0
        
        # Get the end position of the single boxed answer
        start, end = boxed_positions[0]
        
        # Check content after the boxed answer - STRICT: only whitespace allowed
        content_after_boxed = solution_str[end:].strip()
        
        if content_after_boxed:
            if DEBUG:
                preview = content_after_boxed[:200] + "..." if len(content_after_boxed) > 200 else content_after_boxed
                print(f"RLVR_FORMAT INVALID: Extra content after boxed answer: '{preview}'")
            return 0.0
        
        if DEBUG:
            boxed_content = solution_str[start:end]
            print(f"RLVR_FORMAT VALID: Found single boxed answer: '{boxed_content}', no extra content")
        return 1.0
    
    except Exception as e:
        if DEBUG:
            print(f"RLVR_FORMAT ERROR: Exception during validation: {e}")
        return 0.0


def extract_valid_answer(solution_str):
    """Extract the valid portion of answer (returns full solution since no answer tags are used)."""
    return solution_str


# =============================================================================
# RANDOM REWARD SCORING
# =============================================================================

def compute_random_reward_score(solution_str, ground_truth):
    """
    RANDOM_REWARD: Format validation + Random 50/50 scoring.
    Returns 0.0 if format is invalid, otherwise returns random 0.0 or 1.0.
    """
    try:
        if format_validity(solution_str) == 0.0:
            if DEBUG:
                print("RANDOM_REWARD: Format invalid -> 0.0 reward")
            return 0.0
        
        random_score = random.choice([0.0, 1.0])
        
        if DEBUG:
            print(f"RANDOM_REWARD: Format valid -> Random score: {random_score}")
            print(f"   Solution format: VALID")
            print(f"   Ground truth: {ground_truth} (ignored for random scoring)")
            print(f"   Random result: {random_score}")
        
        return random_score
        
    except Exception as e:
        if DEBUG:
            print(f"RANDOM_REWARD: Error during processing: {e}")
        return 0.0


# =============================================================================
# RULE-BASED SCORING
# =============================================================================

def compute_rule_based_score(solution_str, ground_truth):
    """
    RULE_BASED: Uses math_verify for accurate math answer checking.
    """
    try:
        from verl.utils.reward_score import math_verify
        
        # Handle ground truth format: convert list to string if needed
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            processed_ground_truth = str(ground_truth[0])
        else:
            processed_ground_truth = str(ground_truth)
        
        score = math_verify.compute_score(solution_str, processed_ground_truth)
        
        if DEBUG:
            print("\n" + "="*80)
            print("RULE_BASED COMPUTE DEBUG")
            print("="*80)
            print(f"MODEL RESPONSE:\n{solution_str}")
            print(f"\nGROUND TRUTH: {ground_truth}")
            print(f"\nFINAL REWARD: {score}")
            print("="*80 + "\n")
        
        return score
    except Exception as e:
        if DEBUG:
            print(f"Error in math_verify scoring: {e}")
        return 0.0

def compute_verifier_based_score(solution_str, ground_truth, extra_info):
    from verl.utils.reward_score.reasoning_gym import compute_score
    """
    VERIFIER_BASED: Use reasoning-gym verifier for graph domain
    """
    score = compute_score(solution_str, ground_truth, extra_info)
    return score
# =============================================================================
# MAIN SCORING FUNCTIONS
# =============================================================================

def compute_score(data_source, solution_str, ground_truth, extra_info):
    """
    Main scoring function for individual samples.
    Supports RULE_BASED, RANDOM_REWARD, RLVR_FORMAT, RULE_BASED_THINKING_FORMAT, DEEPSEEK_RULE_BASED_THINKING_FORMAT, and MAJORITY_VOTE_FORMAT_PENALTY methods.
    """
    # DEEPSEEK_RULE_BASED_THINKING_FORMAT: DeepSeek thinking format validation + RULE_BASED scoring
    if REWARD_MODEL_TYPE == 'DEEPSEEK_RULE_BASED_THINKING_FORMAT':
        # First check DeepSeek thinking format validity (only </think> tag required)
        if deepseek_thinking_format_validity(solution_str) == 0.0:
            if DEBUG:
                print("\n" + "="*80)
                print("DEEPSEEK_RULE_BASED_THINKING_FORMAT DEBUG - FORMAT PENALTY")
                print("="*80)
                print(f"MODEL RESPONSE:\n{solution_str}")
                print(f"\nFORMAT VALIDITY: INVALID (missing </think> or no boxed answer)")
                print(f"\nGROUND TRUTH: {ground_truth}")
                print(f"\nFINAL REWARD: 0.0 (format penalty)")
                print("="*80 + "\n")
            return {"score": 0.0, "ground_truth": ground_truth, "reward_method": "DEEPSEEK_RULE_BASED_THINKING_FORMAT"}
        
        # Format is valid, use RULE_BASED scoring
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            processed_ground_truth = str(ground_truth[0])
        else:
            processed_ground_truth = str(ground_truth)
        
        reward_score = compute_rule_based_score(solution_str, processed_ground_truth)
        
        if DEBUG:
            print("\n" + "="*80)
            print("DEEPSEEK_RULE_BASED_THINKING_FORMAT DEBUG - VALID FORMAT")
            print("="*80)
            print(f"MODEL RESPONSE:\n{solution_str}")
            print(f"\nFORMAT VALIDITY: VALID (</think> tag and boxed answer present)")
            print(f"\nGROUND TRUTH: {ground_truth}")
            print(f"\nFINAL REWARD: {reward_score}")
            print("="*80 + "\n")
        
        return {"score": reward_score, "ground_truth": ground_truth, "reward_method": "DEEPSEEK_RULE_BASED_THINKING_FORMAT"}
    
    # RULE_BASED_THINKING_FORMAT: Thinking format validation + RULE_BASED scoring
    if REWARD_MODEL_TYPE == 'RULE_BASED_THINKING_FORMAT':
        # First check thinking format validity
        if thinking_format_validity(solution_str) == 0.0:
            if DEBUG:
                print("\n" + "="*80)
                print("RULE_BASED_THINKING_FORMAT DEBUG - FORMAT PENALTY")
                print("="*80)
                print(f"MODEL RESPONSE:\n{solution_str}")
                print(f"\nFORMAT VALIDITY: INVALID (missing <think>...</think> or no boxed answer)")
                print(f"\nGROUND TRUTH: {ground_truth}")
                print(f"\nFINAL REWARD: 0.0 (format penalty)")
                print("="*80 + "\n")
            return {"score": 0.0, "ground_truth": ground_truth, "reward_method": "RULE_BASED_THINKING_FORMAT"}
        
        # Format is valid, use RULE_BASED scoring
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            processed_ground_truth = str(ground_truth[0])
        else:
            processed_ground_truth = str(ground_truth)
        
        reward_score = compute_rule_based_score(solution_str, processed_ground_truth)
        
        if DEBUG:
            print("\n" + "="*80)
            print("RULE_BASED_THINKING_FORMAT DEBUG - VALID FORMAT")
            print("="*80)
            print(f"MODEL RESPONSE:\n{solution_str}")
            print(f"\nFORMAT VALIDITY: VALID (<think>...</think> tags and boxed answer present)")
            print(f"\nGROUND TRUTH: {ground_truth}")
            print(f"\nFINAL REWARD: {reward_score}")
            print("="*80 + "\n")
        
        return {"score": reward_score, "ground_truth": ground_truth, "reward_method": "RULE_BASED_THINKING_FORMAT"}
    
    # RLVR_FORMAT: Format validation + RULE_BASED scoring
    if REWARD_MODEL_TYPE == 'RLVR_FORMAT':
        # First check format validity
        if rlvr_format_validity(solution_str) == 0.0:
            if DEBUG:
                print("\n" + "="*80)
                print("RLVR_FORMAT DEBUG - FORMAT PENALTY")
                print("="*80)
                print(f"MODEL RESPONSE:\n{solution_str}")
                print(f"\nFORMAT VALIDITY: INVALID (extra content after boxed answer)")
                print(f"\nGROUND TRUTH: {ground_truth}")
                print(f"\nFINAL REWARD: 0.0 (format penalty)")
                print("="*80 + "\n")
            return {"score": 0.0, "ground_truth": ground_truth, "reward_method": "RLVR_FORMAT"}
        
        # Format is valid, use RULE_BASED scoring
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            processed_ground_truth = str(ground_truth[0])
        else:
            processed_ground_truth = str(ground_truth)
        
        reward_score = compute_rule_based_score(solution_str, processed_ground_truth)
        
        if DEBUG:
            print("\n" + "="*80)
            print("RLVR_FORMAT DEBUG - VALID FORMAT")
            print("="*80)
            print(f"MODEL RESPONSE:\n{solution_str}")
            print(f"\nFORMAT VALIDITY: VALID (no extra content after boxed answer)")
            print(f"\nGROUND TRUTH: {ground_truth}")
            print(f"\nFINAL REWARD: {reward_score}")
            print("="*80 + "\n")
        
        return {"score": reward_score, "ground_truth": ground_truth, "reward_method": "RLVR_FORMAT"}
    
    # RULE_BASED: Use math_verify for enhanced accuracy
    if REWARD_MODEL_TYPE == 'RULE_BASED':
        # Handle ground truth format: convert list to string if needed
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            processed_ground_truth = str(ground_truth[0])
        else:
            processed_ground_truth = str(ground_truth)
        
        reward_score = compute_rule_based_score(solution_str, processed_ground_truth)
        return {"score": reward_score, "ground_truth": ground_truth, "reward_method": "RULE_BASED"}

    # VERIFIER_BASED: Use reasoning-gym verifier for graph domain
    if 'VERIFIER_BASED' in REWARD_MODEL_TYPE:
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            processed_ground_truth = str(ground_truth[0])
        else:
            processed_ground_truth = str(ground_truth)
        reward_score = compute_verifier_based_score(solution_str, processed_ground_truth, extra_info)
        return {"score": reward_score, "ground_truth": processed_ground_truth, "reward_method": "VERIFIER_BASED"}
    
    # RANDOM_REWARD: Format validation + Random 50/50 scoring
    if REWARD_MODEL_TYPE == 'RANDOM_REWARD':
        # Handle ground truth format: convert list to string if needed
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            processed_ground_truth = str(ground_truth[0])
        else:
            processed_ground_truth = str(ground_truth)
        
        reward_score = compute_random_reward_score(solution_str, processed_ground_truth)
        
        if DEBUG:
            print("\n" + "="*80)
            print("RANDOM_REWARD DEBUG")
            print("="*80)
            print(f"MODEL RESPONSE:\n{solution_str}")
            print(f"\nVALIDITY: {'VALID' if format_validity(solution_str) == 1.0 else 'INVALID'}")
            print(f"\nGROUND TRUTH: {processed_ground_truth} (ignored)")
            print(f"\nFINAL REWARD: {reward_score} (random)")
            print("="*80 + "\n")
        
        return {"score": reward_score, "ground_truth": ground_truth, "reward_method": "RANDOM_REWARD"}
    
    # Fallback for unknown reward types
    print(f"WARNING: Unknown REWARD_MODEL_TYPE: {REWARD_MODEL_TYPE}")
    return {"score": 0.0, "ground_truth": ground_truth, "reward_method": "UNKNOWN"}


# =============================================================================
# BATCH SCORING FUNCTION
# =============================================================================

def is_validation_data(data_sources):
    """
    Check if the batch contains validation data based on data source patterns.
    """
    validation_patterns = [
        'test-math-aime24', 'test-math-aime25', 'huggingfaceh4/math-500', 
        'test-math-', 'test-amc', 'test-scp', 'test-graph-',
        'stem__gpqa', 'stem__supergpqa',  # STEM eval datasets
        'logic__arcagi', 'logic__zebra_puzzle',  # Logic eval datasets
        'mmlu_sci', 'stem__mmlu'  # MMLU science eval datasets
    ]
    return any(
        any(pattern in str(data_source).lower() for pattern in validation_patterns)
        for data_source in data_sources
    )


def compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, reward_method_name="VALIDATION"):
    """
    Common validation scoring function using math_verify.
    Used by all reward model types for validation data.
    """
    from verl.utils.reward_score import math_verify
    
    results = []
    for data_source, solution_str, ground_truth, extra_info in zip(
        data_sources, solution_strs, ground_truths, extra_infos, strict=True
    ):

        # For validation with MAJORITY_VOTE, use original ground truth if available
        if "MAJORITY_VOTE_VALIDATION" in reward_method_name and 'original_ground_truth' in extra_info:
            processed_gt = extra_info['original_ground_truth']
        else:
            processed_gt = ground_truth
        # Handle ground truth format: convert list to string if needed
        if isinstance(processed_gt, list) and len(processed_gt) > 0:
            processed_ground_truth = str(processed_gt[0])
        else:
            processed_ground_truth = str(processed_gt)

        if  'VERIFIER_BASED' in REWARD_MODEL_TYPE:
            score = compute_verifier_based_score(solution_str, processed_ground_truth, extra_info)
            if "MAJORITY_VOTE" in reward_method_name:
                results.append({
                    "score": score, 
                    "majority_vote_gt": ground_truth,                    # Original majority vote GT
                    "original_gt": processed_ground_truth,                         # Original dataset GT (used for validation)
                    "ground_truth": processed_ground_truth,                        # Keep for backward compatibility
                    "reward_method": reward_method_name
                })
            else:
                results.append({"score": score, 
                                "ground_truth": processed_ground_truth, 
                                "reward_method": reward_method_name})
        
        else:     
            score = math_verify.compute_score(solution_str, processed_ground_truth)
            
            # Enhanced return with both ground truths for MAJORITY_VOTE
            if "MAJORITY_VOTE" in reward_method_name:
                results.append({
                    "score": score, 
                    "majority_vote_gt": ground_truth,                    # Original majority vote GT
                    "original_gt": processed_gt,                         # Original dataset GT (used for validation)
                    "ground_truth": processed_gt,                        # Keep for backward compatibility
                    "reward_method": reward_method_name
                })
            else:
                results.append({
                    "score": score, 
                    "ground_truth": processed_gt,
                    "reward_method": reward_method_name
                })
        
    if DEBUG and results:
        correct_count = sum(1 for r in results if r["score"] > 0.5)
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"\n{reward_method_name} SUMMARY: {correct_count}/{total_count} = {accuracy:.3f} accuracy\n")
    
    return results


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    """
    Batch scoring function. Supports RULE_BASED, RANDOM_REWARD, MAJORITY_VOTE, MAJORITY_VOTE_FORMAT_PENALTY, SELF_CERTAINTY, RULE_BASED_THINKING_FORMAT, and DEEPSEEK_RULE_BASED_THINKING_FORMAT methods.
    For RANDOM_REWARD, MAJORITY_VOTE, MAJORITY_VOTE_FORMAT_PENALTY, and SELF_CERTAINTY, uses rule-based scoring for validation data.
    """
    if DEBUG:
        print(f"BATCH INPUT DEBUG: data_sources={len(data_sources)}, solution_strs={len(solution_strs)}, ground_truths={len(ground_truths)}, extra_infos={len(extra_infos)}")
        print(f"REWARD_MODEL_TYPE: {REWARD_MODEL_TYPE}")
    
    # DEEPSEEK_RULE_BASED_THINKING_FORMAT: DeepSeek thinking format validation + RULE_BASED scoring for training, RULE_BASED only for validation
    if REWARD_MODEL_TYPE == 'DEEPSEEK_RULE_BASED_THINKING_FORMAT':
        if is_validation_data(data_sources):
            # Validation: Use rule-based scoring without format checking for meaningful metrics
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "DEEPSEEK_RULE_BASED_THINKING_FORMAT_VALIDATION")
        else:
            # Training: Apply DeepSeek thinking format validation + RULE_BASED scoring
            from verl.utils.reward_score import math_verify
            
            results = []
            format_penalty_count = 0
            
            for data_source, solution_str, ground_truth, extra_info in zip(
                data_sources, solution_strs, ground_truths, extra_infos, strict=True
            ):
                # First check DeepSeek thinking format validity (only </think> tag required)
                if deepseek_thinking_format_validity(solution_str) == 0.0:
                    # Format penalty: 0.0 reward
                    results.append({"score": 0.0, "ground_truth": ground_truth, "reward_method": "DEEPSEEK_RULE_BASED_THINKING_FORMAT"})
                    format_penalty_count += 1
                else:
                    # Format is valid, use RULE_BASED scoring
                    if isinstance(ground_truth, list) and len(ground_truth) > 0:
                        processed_ground_truth = str(ground_truth[0])
                    else:
                        processed_ground_truth = str(ground_truth)
                    
                    score = math_verify.compute_score(solution_str, processed_ground_truth)
                    results.append({"score": score, "ground_truth": ground_truth, "reward_method": "DEEPSEEK_RULE_BASED_THINKING_FORMAT"})
            
            if DEBUG and results:
                correct_count = sum(1 for r in results if r["score"] > 0.5)
                total_count = len(results)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                format_valid_count = total_count - format_penalty_count
                print(f"\nDEEPSEEK_RULE_BASED_THINKING_FORMAT TRAINING SUMMARY:")
                print(f"   Format valid (</think> + boxed): {format_valid_count}/{total_count} = {(format_valid_count/total_count):.3f}")
                print(f"   Format penalties: {format_penalty_count}/{total_count} = {(format_penalty_count/total_count):.3f}")
                print(f"   Overall accuracy: {correct_count}/{total_count} = {accuracy:.3f}")
                print()
            
            return results
    
    # RULE_BASED_THINKING_FORMAT: Thinking format validation + RULE_BASED scoring for training, RULE_BASED only for validation
    if REWARD_MODEL_TYPE == 'RULE_BASED_THINKING_FORMAT':
        if is_validation_data(data_sources):
            # Validation: Use rule-based scoring without format checking for meaningful metrics
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "RULE_BASED_THINKING_FORMAT_VALIDATION")
        else:
            # Training: Apply thinking format validation + RULE_BASED scoring
            from verl.utils.reward_score import math_verify
            
            results = []
            format_penalty_count = 0
            
            for data_source, solution_str, ground_truth, extra_info in zip(
                data_sources, solution_strs, ground_truths, extra_infos, strict=True
            ):
                # First check thinking format validity
                if thinking_format_validity(solution_str) == 0.0:
                    # Format penalty: 0.0 reward
                    results.append({"score": 0.0, "ground_truth": ground_truth, "reward_method": "RULE_BASED_THINKING_FORMAT"})
                    format_penalty_count += 1
                else:
                    # Format is valid, use RULE_BASED scoring
                    if isinstance(ground_truth, list) and len(ground_truth) > 0:
                        processed_ground_truth = str(ground_truth[0])
                    else:
                        processed_ground_truth = str(ground_truth)
                    
                    score = math_verify.compute_score(solution_str, processed_ground_truth)
                    results.append({"score": score, "ground_truth": ground_truth, "reward_method": "RULE_BASED_THINKING_FORMAT"})
            
            if DEBUG and results:
                correct_count = sum(1 for r in results if r["score"] > 0.5)
                total_count = len(results)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                format_valid_count = total_count - format_penalty_count
                print(f"\nRULE_BASED_THINKING_FORMAT TRAINING SUMMARY:")
                print(f"   Format valid (<think>...</think> + boxed): {format_valid_count}/{total_count} = {(format_valid_count/total_count):.3f}")
                print(f"   Format penalties: {format_penalty_count}/{total_count} = {(format_penalty_count/total_count):.3f}")
                print(f"   Overall accuracy: {correct_count}/{total_count} = {accuracy:.3f}")
                print()
            
            return results
    
    # SELF_CERTAINTY: Self-certainty rewards for training, rule-based for validation
    if REWARD_MODEL_TYPE == 'SELF_CERTAINTY':
        if is_validation_data(data_sources):
            # Validation: Use rule-based scoring for meaningful metrics
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "SELF_CERTAINTY_VALIDATION")
        else:
            # Training: Mark for self-certainty processing (actual calculation in ray_trainer.py)
            results = []
            for data_source, solution_str, ground_truth, extra_info in zip(
                data_sources, solution_strs, ground_truths, extra_infos, strict=True
            ):
                results.append({
                    "score": 0.0,  # Placeholder - actual self-certainty applied in ray_trainer.py
                    "ground_truth": ground_truth,
                    "reward_method": "SELF_CERTAINTY"
                })
            
            if DEBUG and results:
                print(f"\nSELF_CERTAINTY TRAINING: {len(results)} examples marked for self-certainty processing\n")
            
            return results
    
    # MAJORITY_VOTE_FORMAT_PENALTY: Format validation + MAJORITY_VOTE scoring for training, RULE_BASED only for validation
    if REWARD_MODEL_TYPE == 'MAJORITY_VOTE_FORMAT_PENALTY':
        if is_validation_data(data_sources):
            # Validation: Use rule-based scoring without format checking for meaningful metrics
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "MAJORITY_VOTE_FORMAT_PENALTY_VALIDATION")
        else:
            # Training: Apply thinking format validation + MAJORITY_VOTE scoring
            from verl.utils.reward_score import math_verify
            
            results = []
            format_penalty_count = 0
            
            for data_source, solution_str, ground_truth, extra_info in zip(
                data_sources, solution_strs, ground_truths, extra_infos, strict=True
            ):
                # First check thinking format validity
                if thinking_format_validity(solution_str) == 0.0:
                    # Format penalty: 0.0 reward
                    original_gt = extra_info.get('original_ground_truth', None)
                    if isinstance(original_gt, list) and len(original_gt) > 0:
                        original_gt = str(original_gt[0])
                    elif original_gt is not None:
                        original_gt = str(original_gt)
                    
                    results.append({
                        "score": 0.0, 
                        "majority_vote_gt": ground_truth,
                        "original_gt": original_gt,
                        "ground_truth": ground_truth, 
                        "reward_method": "MAJORITY_VOTE_FORMAT_PENALTY"
                    })
                    format_penalty_count += 1
                else:
                    # Format is valid, use MAJORITY_VOTE scoring
                    if isinstance(ground_truth, list) and len(ground_truth) > 0:
                        processed_ground_truth = str(ground_truth[0])
                    else:
                        processed_ground_truth = str(ground_truth)
                    
                    # Score against majority vote ground truth
                    score = math_verify.compute_score(solution_str, processed_ground_truth)
                    
                    # Get original_gt from extra_info
                    original_gt = extra_info.get('original_ground_truth', None)
                    if isinstance(original_gt, list) and len(original_gt) > 0:
                        original_gt = str(original_gt[0])
                    elif original_gt is not None:
                        original_gt = str(original_gt)
                    
                    results.append({
                        "score": score, 
                        "majority_vote_gt": ground_truth,
                        "original_gt": original_gt,
                        "ground_truth": ground_truth,
                        "reward_method": "MAJORITY_VOTE_FORMAT_PENALTY"
                    })
            
            if DEBUG and results:
                correct_count = sum(1 for r in results if r["score"] > 0.5)
                total_count = len(results)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                format_valid_count = total_count - format_penalty_count
                print(f"\nMAJORITY_VOTE_FORMAT_PENALTY TRAINING SUMMARY:")
                print(f"   Format valid (<think>...</think> + boxed): {format_valid_count}/{total_count} = {(format_valid_count/total_count):.3f}")
                print(f"   Format penalties: {format_penalty_count}/{total_count} = {(format_penalty_count/total_count):.3f}")
                print(f"   Accuracy against majority vote GT: {correct_count}/{total_count} = {accuracy:.3f}")
                print()
            
            return results
    
    # MAJORITY_VOTE: TTRL-style majority voting rewards
    if REWARD_MODEL_TYPE == 'MAJORITY_VOTE':
        if is_validation_data(data_sources):
            # Validation: Use original ground truth (no majority voting)
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "MAJORITY_VOTE_VALIDATION")
        else:
            # Training: Use majority vote ground truth (already replaced by TTRL)
            from verl.utils.reward_score import math_verify
            
            results = []
            for data_source, solution_str, ground_truth, extra_info in zip(
                data_sources, solution_strs, ground_truths, extra_infos, strict=True
            ):
                # Handle ground truth format (majority vote is already set as ground_truth)
                if isinstance(ground_truth, list) and len(ground_truth) > 0:
                    processed_ground_truth = str(ground_truth[0])
                else:
                    processed_ground_truth = str(ground_truth)
                
                # Use math_verify to compare solution against majority vote ground truth
                score = math_verify.compute_score(solution_str, processed_ground_truth)
                
                # Get original_gt from extra_info (available in updated training data)
                original_gt = extra_info.get('original_ground_truth', None)
                
                # Handle ground truth format: convert list to string if needed
                if isinstance(original_gt, list) and len(original_gt) > 0:
                    original_gt = str(original_gt[0])
                elif original_gt is not None:
                    original_gt = str(original_gt)
                
                results.append({
                    "score": score, 
                    "majority_vote_gt": ground_truth,                    # Majority vote ground truth (clear naming)
                    "original_gt": original_gt,                          # Original dataset ground truth
                    "ground_truth": ground_truth,                        # Keep for backward compatibility
                    "reward_method": "MAJORITY_VOTE"
                })
            
            if DEBUG and results:
                correct_count = sum(1 for r in results if r["score"] > 0.5)
                total_count = len(results)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                
                # Show comparison between majority vote and original GT
                original_gts = [r.get('original_gt') for r in results if r.get('original_gt') is not None]
                majority_gts = [r.get('majority_vote_gt') for r in results]
                
                if original_gts and len(original_gts) == len(majority_gts):
                    matches = sum(1 for orig, maj in zip(original_gts, majority_gts) if str(orig) == str(maj))
                    agreement = matches / len(original_gts) if original_gts else 0.0
                    print(f"\nMAJORITY_VOTE TRAINING SUMMARY:")
                    print(f"   Accuracy against majority vote GT: {correct_count}/{total_count} = {accuracy:.3f}")
                    print(f"   Majority vote ↔ Original GT agreement: {matches}/{len(original_gts)} = {agreement:.3f}")
                else:
                    print(f"\nMAJORITY_VOTE TRAINING SUMMARY: {correct_count}/{total_count} = {accuracy:.3f} accuracy against majority vote GT")
                print()
            
            return results
    
    # RLVR_FORMAT: Format validation + RULE_BASED scoring for ALL data (training + validation)
    if REWARD_MODEL_TYPE == 'RLVR_FORMAT':
        from verl.utils.reward_score import math_verify
        
        results = []
        format_penalty_count = 0
        
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            # First check format validity
            if rlvr_format_validity(solution_str) == 0.0:
                # Format penalty: 0.0 reward
                results.append({"score": 0.0, "ground_truth": ground_truth, "reward_method": "RLVR_FORMAT"})
                format_penalty_count += 1
            else:
                # Format is valid, use RULE_BASED scoring
                if isinstance(ground_truth, list) and len(ground_truth) > 0:
                    processed_ground_truth = str(ground_truth[0])
                else:
                    processed_ground_truth = str(ground_truth)
                
                score = math_verify.compute_score(solution_str, processed_ground_truth)
                results.append({"score": score, "ground_truth": ground_truth, "reward_method": "RLVR_FORMAT"})
        
        if DEBUG and results:
            correct_count = sum(1 for r in results if r["score"] > 0.5)
            total_count = len(results)
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            format_valid_count = total_count - format_penalty_count
            print(f"\nRLVR_FORMAT BATCH SUMMARY:")
            print(f"   Format valid: {format_valid_count}/{total_count} = {(format_valid_count/total_count):.3f}")
            print(f"   Format penalties: {format_penalty_count}/{total_count} = {(format_penalty_count/total_count):.3f}")
            print(f"   Overall accuracy: {correct_count}/{total_count} = {accuracy:.3f}")
            print()
        
        return results
    
    # RULE_BASED: Force math_verify scoring for ALL data (training + validation)
    if REWARD_MODEL_TYPE == 'RULE_BASED':
        from verl.utils.reward_score import math_verify
        
        results = []
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            # Handle ground truth format: convert list to string if needed
            if isinstance(ground_truth, list) and len(ground_truth) > 0:
                processed_ground_truth = str(ground_truth[0])
            else:
                processed_ground_truth = str(ground_truth)
            
            score = math_verify.compute_score(solution_str, processed_ground_truth)
            results.append({"score": score, "ground_truth": ground_truth, "reward_method": "RULE_BASED"})
        
        return results
    
    # VERIFIER_BASED: Use reasoning-gym verifier for graph domain
    if REWARD_MODEL_TYPE == 'VERIFIER_BASED':
        results = []
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            if isinstance(ground_truth, list) and len(ground_truth) > 0:
                processed_ground_truth = str(ground_truth[0])
            else:
                processed_ground_truth = str(ground_truth)
            score = compute_verifier_based_score(solution_str, processed_ground_truth, extra_info)
            results.append({"score": score, "ground_truth": processed_ground_truth, "reward_method": "VERIFIER_BASED"})
        return results
    # RANDOM_REWARD: Format validation + Random 50/50 scoring for TRAINING, RULE_BASED for VALIDATION
    if REWARD_MODEL_TYPE == 'VERIFIER_BASED_MAJORITY_VOTE':
        if is_validation_data(data_sources):
            # Validation: Use original ground truth (no majority voting)
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "MAJORITY_VOTE_VALIDATION")
        else:
            # Training: Use majority vote ground truth (already replaced by TTRL)
            from verl.utils.reward_score import math_verify
            
            results = []
            for data_source, solution_str, ground_truth, extra_info in zip(
                data_sources, solution_strs, ground_truths, extra_infos, strict=True
            ):
                if isinstance(ground_truth, list) and len(ground_truth) > 0:
                    processed_ground_truth = str(ground_truth[0])
                else:
                    processed_ground_truth = str(ground_truth)
                score = compute_verifier_based_score(solution_str, processed_ground_truth, extra_info)
                
                # Get original_gt from extra_info (available in updated training data)
                original_gt = extra_info.get('original_ground_truth', None)
                
                # Handle ground truth format: convert list to string if needed
                if isinstance(original_gt, list) and len(original_gt) > 0:
                    original_gt = str(original_gt[0])
                elif original_gt is not None:
                    original_gt = str(original_gt)
                
                results.append({
                    "score": score, 
                    "majority_vote_gt": ground_truth,                    # Majority vote ground truth (clear naming)
                    "original_gt": original_gt,                          # Original dataset ground truth
                    "ground_truth": ground_truth,                        # Keep for backward compatibility
                    "reward_method": "MAJORITY_VOTE"
                })
            
            if DEBUG and results:
                correct_count = sum(1 for r in results if r["score"] > 0.5)
                total_count = len(results)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                
                # Show comparison between majority vote and original GT
                original_gts = [r.get('original_gt') for r in results if r.get('original_gt') is not None]
                majority_gts = [r.get('majority_vote_gt') for r in results]
                
                if original_gts and len(original_gts) == len(majority_gts):
                    matches = sum(1 for orig, maj in zip(original_gts, majority_gts) if str(orig) == str(maj))
                    agreement = matches / len(original_gts) if original_gts else 0.0
                    print(f"\nMAJORITY_VOTE TRAINING SUMMARY:")
                    print(f"   Accuracy against majority vote GT: {correct_count}/{total_count} = {accuracy:.3f}")
                    print(f"   Majority vote ↔ Original GT agreement: {matches}/{len(original_gts)} = {agreement:.3f}")
                else:
                    print(f"\nMAJORITY_VOTE TRAINING SUMMARY: {correct_count}/{total_count} = {accuracy:.3f} accuracy against majority vote GT")
                print()
            
            return results
    if REWARD_MODEL_TYPE == 'RANDOM_REWARD':
        if is_validation_data(data_sources):
            # For validation: Use RULE_BASED scoring for meaningful metrics
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "RANDOM_REWARD_VALIDATION")
        else:
            # For training: Use random scoring
            results = []
            for data_source, solution_str, ground_truth, extra_info in zip(
                data_sources, solution_strs, ground_truths, extra_infos, strict=True
            ):
                if isinstance(ground_truth, list) and len(ground_truth) > 0:
                    processed_ground_truth = str(ground_truth[0])
                else:
                    processed_ground_truth = str(ground_truth)
                
                score = compute_random_reward_score(solution_str, processed_ground_truth)
                results.append({"score": score, "ground_truth": ground_truth, "reward_method": "RANDOM_REWARD"})
            
            if DEBUG and results:
                correct_count = sum(1 for r in results if r["score"] > 0.5)
                total_count = len(results)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                print(f"\nRANDOM_REWARD TRAINING: {correct_count}/{total_count} = {accuracy:.3f} random accuracy (should be ~0.5)\n")
            
            return results
    
    # Fallback for unknown reward types
    print(f"WARNING: Unknown REWARD_MODEL_TYPE in batch: {REWARD_MODEL_TYPE}")
    return [{"score": 0.0, "ground_truth": gt, "reward_method": "UNKNOWN"} for gt in ground_truths]
