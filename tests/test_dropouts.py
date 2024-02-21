import torch
import lightning as L
from io import StringIO
from unittest.mock import Mock
from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.utils import chunked_cross_entropy, load_checkpoint
from contextlib import redirect_stdout
from torch.utils.data import DataLoader


def test_dropouts(tmp_path):
    """
    We test using two methods.
    
    1.Train a single step on 3 models, one with dropout and two without. The 
    two without should be identical, and the one with should be different.
    
    Two are trained without because then we can check that no difference in 
    output is a result of random number generation or something.
    
    2. Manually probe the model and calculate what the expected logits would be.
    """
    config = Config(block_size=128, vocab_size=16, n_layer=1, n_head=4, 
                    n_embd=8, parallel_residual=False)
    model_no_dropout_1 = GPT(config)    
    model_no_dropout_2 = GPT(config)
    model_with_dropout = GPT(config)

    for model in [model_no_dropout_1, model_no_dropout_2, model_with_dropout]:
        model.max_seq_length = 16
        model.set_kv_cache(batch_size=1)

    fabric = L.Fabric(accelerator="cpu")
    model_no_dropout_1 = fabric.setup(model_no_dropout_1)
    model_no_dropout_2 = fabric.setup(model_no_dropout_2)
    model_with_dropout = fabric.setup(model_with_dropout)
    model_with_dropout.patch_block(0, dropout=True, pdrop=1)
    
    # Make the models identical
    fabric.save(tmp_path / "model.pt", {"model": model_no_dropout_1})
    load_checkpoint(fabric, model_no_dropout_2, tmp_path / "model.pt")
    load_checkpoint(fabric, model_with_dropout, tmp_path / "model.pt")

    optimizer_no_dropout_1 = torch.optim.SGD(model_no_dropout_1.parameters())
    optimizer_no_dropout_2 = torch.optim.SGD(model_no_dropout_2.parameters())
    optimizer_with_dropout = torch.optim.SGD(model_with_dropout.parameters())

    optimizer_no_dropout_1 = fabric.setup_optimizers(optimizer_no_dropout_1)
    optimizer_no_dropout_2 = fabric.setup_optimizers(optimizer_no_dropout_2)
    optimizer_with_dropout = fabric.setup_optimizers(optimizer_with_dropout)

    input_ids = torch.Tensor([[1, 2]]).to(torch.int64)
    targets = torch.Tensor([[1, 9]]).to(torch.int64)

    # Check dropouts work and only occur during training
    captured_outputs = {}
    def capture_output_hook(module, input, output):
        captured_outputs["attn_output"] = output.detach()
    hook = model_with_dropout.transformer.h[0].attn.register_forward_hook(capture_output_hook)
    embed_logits = model_with_dropout.transformer.wte(input_ids)
    identity_logits = model_with_dropout(input_ids)
    expected_logits = model_with_dropout.lm_head(model_with_dropout.transformer.ln_f(captured_outputs["attn_output"] + embed_logits))
    hook.remove()
    assert torch.equal(identity_logits, expected_logits)
    model_with_dropout.eval()
    assert not torch.equal(model_with_dropout(input_ids), identity_logits), f"MLP dropouts test mode not working"

    # Train models
    logits_no_dropout_1 = train_single_step(model_no_dropout_1, fabric, input_ids, targets)
    logits_no_dropout_2 = train_single_step(model_no_dropout_2, fabric, input_ids, targets)
    logits_with_dropout = train_single_step(model_with_dropout, fabric, input_ids, targets)

    # Check that the outputs of the two models without dropout are identical
    assert torch.equal(logits_no_dropout_1, logits_no_dropout_2), "Outputs of the models without dropout should be the same."

    # Check that the output with dropout is different
    assert not torch.equal(logits_no_dropout_1, logits_with_dropout), "Output with dropout should be different from the ones without dropout."


def train_single_step(model, fabric, input_ids, targets):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer = fabric.setup_optimizers(optimizer)

    with fabric.no_backward_sync(model):
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        fabric.backward(loss)
        optimizer.step()
        
    return logits.detach()
