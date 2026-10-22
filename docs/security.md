# Embedding caches

Word2vec vocabulary and NumPy pickle caches are read with a restricted unpickler.
Unsupported or malformed caches are rebuilt from the source embeddings. Tensor
caches use `weights_only=True` on stable PyTorch versions starting at 2.10. Older
or unrecognized versions rebuild tensors from word2vec data instead, avoiding
known vulnerabilities in older PyTorch readers. No dependency upgrade or model
republication is required, though rebuilding tensors may make loading slower.

These checks prevent unrestricted deserialization in the embedding cache paths;
they do not authenticate model contents or protect against resource exhaustion.
