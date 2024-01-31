def freeze(model, n_freeze, freeze_embed, module_name="layers"):
    if n_freeze > 0:
        def _find_mod(model, module_name):
            for name, mod in model.named_modules():
                if name.endswith(module_name):
                    return mod
        # freeze layers (disable gradients)
        for param in model.parameters(): param.requires_grad = False

        # never freeze the head
        for param in model.lm_head.parameters(): param.requires_grad = True
    
        layers = _find_mod(model, module_name)
        for param in layers[n_freeze:].parameters(): param.requires_grad = True
    
    # Freeze embeddings for small memory decrease
    if freeze_embed:
        embed_tokens = _find_mod(model, "embed_tokens")
        embed_tokens.weight.requires_grad_(False);