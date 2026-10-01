"""Shared Adamas dispatch for Hugging Face accuracy evaluations."""


def enable_adamas_attention_eval(model, args):
    if args.token_budget is None or args.token_budget <= 0:
        raise ValueError("Adamas requires a positive token budget")
    if model.config.model_type == "llama":
        from evaluation import adamas_attention
        adamas_attention.layer_id = model.config.num_hidden_layers
        adamas_attention.enable_adamas_attention_eval(model, args)
    elif model.config.model_type == "qwen3":
        from evaluation.adamas_attention_qwen3 import enable_adamas_attention_eval as enable
        enable(model, args)
    else:
        raise ValueError(f"Unsupported Adamas model type: {model.config.model_type}")
