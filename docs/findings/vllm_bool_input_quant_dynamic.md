# vLLM `compressed_tensors.py` — `bool()` wrap on `is_static_input_scheme`

**Status:** Surfaced during sibling-repo W4A16 serve smoke testing on H200. One-liner PR; file against `vllm-project/vllm` today, in parallel with calibration.

## Symptom

`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` lines 683 and 704 set:

```python
is_static_input_scheme = input_quant and not input_quant.dynamic
```

When `input_quant` is `None` (the weight-only case — our recipe configures `input_quant=None` for the FP8_BLOCK attn and NVFP4 expert schemes), this expression short-circuits and returns `None`, not `False`. Downstream code that does `if is_static_input_scheme: ...` is fine (Python truthy), but anything that does `is_static_input_scheme is False` or passes the value to a typed function expecting `bool` quietly breaks.

## Fix

```python
is_static_input_scheme = bool(input_quant and not input_quant.dynamic)
```

at both call sites (lines 683 and 704).

## Why this is worth filing

- One-line change, low review surface.
- Removes a class of subtle type bugs in any caller that compares with `is False` or passes the value as `bool`.
- Establishes contribution lineage for the larger sharded-MoE save coordination PR we'll file later — reviewers see the contributor name on a clean simple PR first.

## PR design

Title: `[Bugfix] CompressedTensorsScheme: wrap is_static_input_scheme with bool()`

Body:
- Symptom: with `input_quant=None`, `input_quant and not input_quant.dynamic` evaluates to `None`, not `False`. Most call sites get the right behavior accidentally via Python truthiness, but `is False`/`is True` comparisons and `bool`-typed call sites quietly do the wrong thing.
- Reproducer: recipe with `QuantizationScheme(format="float-quantized", weights=..., input_activations=None)` — common for weight-only schemes — produces a model whose serve-side schema introspection returns `is_static_input_scheme=None`.
- Fix: wrap with `bool()` at both sites.
- Test: extend the existing `tests/model_executor/quantization/test_compressed_tensors.py` (if present; otherwise add) with a unit test asserting the returned value is `bool` not `NoneType`.

Identity: file under `paul@sunday.market` / `canada-quant` org footer. Reference the prior contact via vLLM #41511 in the PR body so reviewers know the contributor.
