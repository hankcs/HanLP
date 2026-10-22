# Model loading and trust

`hanlp.load()` trusts classes and functions defined in the first-party `hanlp`,
`hanlp_common`, and `hanlp_trie` packages. Existing model configurations keep their classpaths;
no model conversion or republication is required.

Classpaths outside these packages are rejected before importing their modules.
The same rule applies to nested components, vocabularies, and pipeline functions
resolved during the load. Re-exporting an external class through a HanLP module
does not make it trusted.

For custom models that use other installed Python packages, explicitly opt in:

```python
model = hanlp.load(path, trust_remote_code=True)
```

This trusts the external code selected by that model, including its constructors
and nested components. It applies only to the current load. Custom models that
previously relied on external classpaths need this argument.

This policy is not a sandbox for untrusted models. Trusted HanLP components can
download dependencies from URLs specified in model configurations. Only load
models from sources you trust. Direct component loading
and general-purpose reflection outside `hanlp.load()` retain their existing
behavior and do not enter this restricted loading context.

# Embedding caches

Word2vec vocabulary and NumPy pickle caches are read with a restricted unpickler.
Unsupported or malformed caches are rebuilt from the source embeddings. Tensor
caches use `weights_only=True` on stable PyTorch versions starting at 2.10. Older
or unrecognized versions rebuild tensors from word2vec data instead, avoiding
known vulnerabilities in older PyTorch readers. No dependency upgrade or model
republication is required, though rebuilding tensors may make loading slower.

These checks prevent unrestricted deserialization in the embedding cache paths;
they do not authenticate model contents or protect against resource exhaustion.
