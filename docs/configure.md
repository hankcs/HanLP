# Configuration

## Customize ``HANLP_HOME``

All resources HanLP use will be cached into a directory called `HANLP_HOME`. 
It is an environment variable which you can customize to any path you like. 
By default, `HANLP_HOME` resolves to `~/.hanlp` and `%appdata%\hanlp` on *nix and Windows respectively. 
If you want to temporally redirect `HANLP_HOME` to a different location, say `/data/hanlp`, the following shell command can be very helpful.

```bash
export HANLP_HOME=/data/hanlp
```

## Using GPUs

By default, HanLP tries to use the least occupied GPU so that mostly you don't need to worry about it, HanLP makes the best choice for you. This behavior is very useful when you're using a public server shared across your lab or company with your collegues. 

HanLP also honors the ``CUDA_VISIBLE_DEVICES`` used by PyTorch and TensorFlow to limit which devices HanLP can choose from. For example, the following command will only keep the `0`th and `1`th GPU.

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

```{eval-rst}
If you need fine grained control over each component, ``hanlp.load(..., devices=...)`` is what you're looking for.
See documents for :meth:`hanlp.load`.
```

:::{seealso}

For deep learning beginners, you might need to learn how to set up a working GPU environment first. Here are some 
resources.

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
    - It's a good practice to install the driver inside a CUDA package. 
- [PyTorch](https://pytorch.org/get-started/locally/)
    - If no existing PyTorch found, `pip install hanlp` will have the CPU-only PyTorch installed, which is universal and assumes no GPU or CUDA dependencies. 
    - You will need to install a GPU-enabled PyTorch according to your CUDA and OS versions.
- Cloud servers
    - There are many cloud service providing out-of-box deep learning images. HanLP works fine on these platforms. 
        They could save your time and efforts.

:::

## Using mirror sites

By default, we maintain a global CDN to host the models. However, in some regions the downloading speed can 
be slow occasionally. If you happen to be in one of those regions, you can find some third party mirror sites 
on our [bbs](https://bbs.hankcs.com/). When you find a working URL, say 
[http://mirrors-hk.miduchina.com/hanlp/](http://mirrors-hk.miduchina.com/hanlp/) , you can set a `HANLP_URL` 
environment variable and HanLP will pick it up at the next startup.

```bash
export HANLP_URL=http://mirrors-hk.miduchina.com/hanlp/
```

## Control Verbosity

By default, HanLP will print progressive message to console when you load a model. If you want to silence it, use the 
following environment variable.

```bash
export HANLP_VERBOSE=0
```
