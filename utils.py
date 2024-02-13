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


def map_state_dict(original_model, layer_ids=[0,1,30,31], layer_naming="layers"):
    "We will map the parameters of the original model layer_ids to the new model layer_ids"
    name_mapping = {}
    layer_mapping = {layer_id: i for i, layer_id in enumerate(layer_ids)}
    print(f"Layer mapping: {layer_mapping}")
    for name, _ in original_model.named_parameters():
        if layer_naming in name:
            layer_id = int(name.split(".")[2])
            if layer_id in layer_ids:
                new_name = name.replace(f"{layer_naming}.{layer_id}", f"{layer_naming}.{layer_mapping[layer_id]}")
                name_mapping[name] = new_name
        else:
            name_mapping[name] = name
    return name_mapping