from transformers import AutoTokenizer
from functools import cache


@cache
def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer(model_name)
    return tokenizer
