# Install

## Install Native Package

The native package running locally can be installed via pip.

```
pip install hanlp
```

HanLP requires Python 3.6 or later. GPU/TPU is suggested but not mandatory. Depending on your preference, HanLP offers the following flavors:

````{margin} **Windows Support**
```{note}
Installation on Windows is **perfectly** supported. The full version `hanlp[full]` additionally requires 
[Microsoft Visual C++ Build Tools](http://go.microsoft.com/fwlink/?LinkId=691126) to compile C++ extensions. 
```
````

| Flavor  | Description                                                  |
| ------- | ------------------------------------------------------------ |
| default | This installs the default version which delivers the most commonly used functionalities. However, some heavy dependencies like TensorFlow are not installed. |
| full    | For experts who seek to maximize the efficiency via TensorFlow and C++ extensions, `pip install hanlp[full]` installs every dependency HanLP will use in production. |

## Install Models

In short, you don't need to manually install any model. Instead, they are automatically downloaded to a directory called `HANLP_HOME` when you call `hanlp.load`.
Occasionally, some errors might occur the first time you load a model, in which case you can refer to the following tips.

### Download Error

If the auto-download fails, you can either:

1. Retry as our file server might be busy serving users from all over the world.
1. Follow the message on your terminal, which often guides you to manually download a `zip` file to a particular path. 
1. Use a [mirror site](https://hanlp.hankcs.com/docs/configure.html#use-mirror-sites) which could be faster and stabler in your region.

### Server without Internet

If your server has no Internet access at all, just debug your codes on your local PC and copy the following directory to your server via a USB disk.

1. `~/.hanlp`: the home directory for HanLP models.
1. `~/.cache/huggingface`: the home directory for Hugging Face 🤗 Transformers.

### Import Error

Some TensorFlow/fastText models will ask you to install the missing TensorFlow/fastText modules, in which case you'll need to install the full version:

```shell script
pip install hanlp[full]
```  


```{caution}
DO NOT install TensorFlow/fastText by yourself, as higher or lower versions of TensorFlow have not been tested and might not work properly. 
```