# minlora-reproduction

A hands-on reproduction of [minLoRA](https://github.com/ceylanmesut/minLoRA) by changjonathanc, implemented to understand the mechanics of Low-Rank Adaptation (LoRA) for transformer fine-tuning.

## What it covers

- Reproduced the core minLoRA implementation from scratch by reading through the original codebase
- Traced through the low-rank matrix decomposition logic (W = W₀ + BA) to understand how LoRA injects trainable parameters
- Identified and fixed a bug: the original `remove_lora` function modified a dictionary while iterating over it, causing a `RuntimeError`. Fixed by wrapping the keys in `list()` before iteration

## Bug fix

```python
# original — raises RuntimeError
for name, module in model.named_modules():
    ...

# fixed — iterate over a copied list
for name, module in list(model.named_modules()):
    ...
```

## Key takeaways

- How LoRA freezes the original weights and only trains two small matrices A and B
- Why LoRA is memory-efficient compared to full fine-tuning
- How to inject and remove LoRA layers from an existing PyTorch model

## Credit

Based on [minLoRA](https://github.com/ceylanmesut/minLoRA) by changjonathanc, licensed under MIT.
