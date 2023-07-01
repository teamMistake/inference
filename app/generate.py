import torch
import sys
from pathlib import Path
import argparse
import glob
import utils
import os
from jamo import JAMO, Tokenizer


@torch.no_grad()
def generate(
        model: JAMO,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        max_seq_length=None,
        temperature: float = 1.0,
        top_k=None,
        eos_id=None,
) -> torch.Tensor:
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    T_new = T + max_new_tokens
    if max_seq_length is None:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    # generate max_new_tokens tokens
    for _ in range(max_new_tokens):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, max_seq_length, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1
        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)
        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description='Train My Custom GPT 🚀!!!')
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--model_path", type=str, default="/home/jovyan/jamo_llm/tmp/checkpoint/")
    parser.add_argument("--chat", action="store_true")
    args = parser.parse_args()

    # Loading the pretrained model.
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_path)
    model = utils.load_model(model_path, model_size="small", device=device)
    model.eval()

    # Loading the tokenizer.
    if model.config.vocab_size == 20000:
        tokenizer = Tokenizer("./tokenizer/corpus.model")
    elif model.config.vocab_size == 8000:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("hg_tokenizer")
    print("⭐️ Loading LLM Done! ⭐️")

    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    EOS_ID = tokenizer.encode(EOS_TOKEN)[0]

    chat_parser = (
        "명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### 명령어:\n{instruction}\n\n### 응답:"
    )

    while True:
        user_prompt = input(">>> ")
        if args.chat: user_prompt = chat_parser.format_map({"instruction":user_prompt})
        user_prompt = f"{SOS_TOKEN} {user_prompt}"

        if user_prompt == "q":
            break

        idx = tokenizer.encode(user_prompt)
        token = torch.tensor(idx, dtype=torch.long, device=device)

        cur = len(SOS_TOKEN)
        for idx in generate(model, token, max_new_tokens=128, temperature=1, top_k=40, eos_id=EOS_TOKEN):
            target = tokenizer.decode(idx)

            for char in target[cur:]:
               sys.stdout.write(char)
               sys.stdout.flush()
               cur = len(target)

        model.reset_cache()
        print("\n")
