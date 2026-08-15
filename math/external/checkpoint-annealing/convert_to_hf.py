"""Convert an annealed native OLMo-core checkpoint to HuggingFace format.

Thin wrapper around ``olmo_core.nn.hf.convert_checkpoint_to_hf``. The annealed
checkpoint written by ``anneal.py`` already stores its model + tokenizer config in the
current OLMo-core schema, so this reads that config and runs the standard converter
(with numerical validation on by default).

The resulting HF directory is what the downstream SFT + RL stages consume.

Usage:
    python convert_to_hf.py -i $OUT_DIR/annealed/swafix-step6000/stepNNNN -o $OUT_DIR/hf/swafix-step6000
"""

from argparse import ArgumentParser

import torch

from olmo_core.config import DType
from olmo_core.nn.hf import convert_checkpoint_to_hf, load_config
from olmo_core.utils import prepare_cli_environment


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True, help="Annealed OLMo-core checkpoint dir (the stepNNNN folder).")
    p.add_argument("-o", "--output", required=True, help="Output dir for the HF checkpoint.")
    p.add_argument("-s", "--max-sequence-length", type=int, help="Defaults to the tokenizer's model_max_length.")
    p.add_argument("-t", "--tokenizer", help="HF tokenizer id to save with (defaults to the experiment config's).")
    p.add_argument("--dtype", type=DType, default=DType.bfloat16, help="Saved weight dtype (default bfloat16).")
    p.add_argument("--skip-validation", dest="validate", action="store_false",
                   help="Skip the numerical check that the HF model matches the original.")
    p.add_argument("--device", type=torch.device, help="Device for conversion (default CPU).")
    return p.parse_args()


def main():
    args = parse_args()

    experiment_config = load_config(args.input)
    if experiment_config is None:
        raise RuntimeError(f"No experiment config found in {args.input}; cannot convert.")

    transformer_config_dict = experiment_config["model"]
    tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")
    assert transformer_config_dict is not None and tokenizer_config_dict is not None

    convert_checkpoint_to_hf(
        original_checkpoint_path=args.input,
        output_path=args.output,
        transformer_config_dict=transformer_config_dict,
        tokenizer_config_dict=tokenizer_config_dict,
        dtype=args.dtype,
        max_sequence_length=args.max_sequence_length,
        tokenizer_id=args.tokenizer,
        validate=args.validate,
        device=args.device,
        validation_device=args.device,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
