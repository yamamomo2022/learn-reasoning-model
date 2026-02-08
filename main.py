from src.generate_text_functions import get_device, generate_text_basic
import torch
from src.qwen3 import download_qwen3_small
from pathlib import Path
from src.qwen3 import Qwen3Tokenizer
from src.qwen3 import Qwen3Model, QWEN_CONFIG_06_B


if __name__ == "__main__":

    download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
    tokenizer_path = Path("qwen3") / "tokenizer-base.json"
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    model_path = Path("qwen3") / "qwen3-0.6B-base.pth"

    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))

    device = get_device()
    model.to(get_device())

    prompt = "Explain large language models in a single sentence."
    input_token_ids_tensor = torch.tensor(
        tokenizer.encode(prompt), device=device
    ).unsqueeze(0)

    max_new_tokens = 100
    output_token_ids_tensor = generate_text_basic(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=max_new_tokens,
    )
    output_text = tokenizer.decode(output_token_ids_tensor.squeeze(0).tolist())
    print(output_text)
