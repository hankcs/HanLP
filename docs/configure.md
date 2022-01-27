# Configuration

## Customize ``HANLP_HOME``

All resources HanLP use will be cached into a directory called `HANLP_HOME`. 
It is an environment variable which you can customize to any path you like. 
By default, `HANLP_HOME` resolves to `~/.hanlp` and `%appdata%\hanlp` on *nix and Windows respectively. 
If you want to redirect `HANLP_HOME` to a different location, say `/data/hanlp`, the following shell command can be very helpful.

```bash
export HANLP_HOME=/data/hanlp
```

## Use GPUs

By default, HanLP tries to use the least occupied GPU so that mostly you don't need to worry about it, HanLP makes the best choice for you. This behavior is very useful when you're using a public server shared across your lab or company with your colleagues. 

HanLP also honors the ``CUDA_VISIBLE_DEVICES`` used by PyTorch and TensorFlow to limit which devices HanLP can choose from. For example, the following command will only keep the `0`th and `1`st GPUs.

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

```{eval-rst}
If you need fine grained control over each component, ``hanlp.load(..., devices=...)`` is what you're looking for.
See documents for :meth:`hanlp.load`.
```

### External Resources

For deep learning beginners, you might need to learn how to set up a working GPU environment first. Here are some 
resources.

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
    - It's a good practice to install the driver shipped with a CUDA package. 
- [PyTorch](https://pytorch.org/get-started/locally/)
    - If no existing PyTorch found, `pip install hanlp` will have the CPU-only PyTorch installed, which is universal and assumes no GPU or CUDA dependencies. 
    - You will need to install a GPU-enabled PyTorch according to your CUDA and OS versions.
- Cloud servers
    - There are many cloud services providing out-of-the-box deep learning images. HanLP works fine on these platforms. 
        They could save your time and efforts.
- Google Colab
    - Colab allows you to write excutable notebooks with full GPU support. PyTorch and TensorFlow have been pre-installed and configured to the best state.
    - In fact, you can click [![Open In Colab](https://file.hankcs.com/img/colab-badge.svg)](https://colab.research.google.com/drive/1KPX6t1y36TOzRIeB4Kt3uJ1twuj6WuFv?usp=sharing) to play with the GPU-enabled HanLP tutorial right now.


## Use Mirror Sites

By default, models are downloaded from a global CDN we maintain. However, in some regions the downloading speed can 
be slow occasionally. If you happen to be in one of those regions, you can find some third party mirror sites 
on our [bbs](https://bbs.hankcs.com/). When you find a working URL, say 
[https://ftp.hankcs.com/hanlp/](https://ftp.hankcs.com/hanlp/), you can set a `HANLP_URL` 
environment variable and HanLP will pick it up at the next startup.

```bash
export HANLP_URL=https://ftp.hankcs.com/hanlp/
```

## Control Verbosity

By default, HanLP will print progressive message to the console when you load a model. If you want to silence it, use the 
following environment variable.

```bash
export HANLP_VERBOSE=0
```
