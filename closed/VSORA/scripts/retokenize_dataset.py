"""Re-tokenize a preprocessed MLPerf LLM dataset for a different model's tokenizer (dev models only).

The official files are tokenized for the benchmark model. Prompts are rebuilt from the file's own template
column filled with its row fields, so nothing benchmark-specific lives here. All original columns (targets
for the official scorer) are kept; only tok_input is replaced.
"""
import argparse
import ast

import pandas as pd
from transformers import AutoTokenizer


def read_table(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    return pd.read_json(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="official preprocessed file, e.g. data/cnn_eval.json")
    p.add_argument("--dst", required=True, help="output .json/.pkl/.parquet")
    p.add_argument("--model", required=True, help="HF id or local path of the tokenizer")
    p.add_argument("--template-column", default="instruction")
    p.add_argument("--template-key", default="llama", help="key inside the template column's dict")
    p.add_argument(
        "--chat-template",
        action="store_true",
        help="wrap prompts in the model's chat template (small instruct models answer EOS to raw prompts)",
    )
    args = p.parse_args()

    df = read_table(args.src)
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for row in df.to_dict("records"):
        template = row[args.template_column]
        if isinstance(template, str):
            template = ast.literal_eval(template)
        prompts.append(template[args.template_key].format_map(row))
    if args.chat_template:
        prompts = [
            tok.apply_chat_template([{"role": "user", "content": x}], add_generation_prompt=True, tokenize=False)
            for x in prompts
        ]
    df["tok_input"] = tok(prompts, add_special_tokens=not args.chat_template)["input_ids"]

    if args.dst.endswith(".parquet"):
        df.to_parquet(args.dst)
    elif args.dst.endswith((".pkl", ".pickle")):
        df.to_pickle(args.dst)
    else:
        df.to_json(args.dst, orient="records", force_ascii=False)
    print(f"{len(df)} samples -> {args.dst}")


if __name__ == "__main__":
    main()
