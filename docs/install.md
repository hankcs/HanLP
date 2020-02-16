# Install

## Install Native Package

The native package runs locally which can be installed via pip.

```
pip install hanlp
```

HanLP requires Python 3.6 or later. GPU/TPU is suggested but not mandatory. Depending on your preference, HanLP offers the following flavors:

| Flavor  | Description                                                  |
| ------- | ------------------------------------------------------------ |
| default | This installs the default version which delivers the most commonly used functionalities. However, some heavy dependencies like TensorFlow are not installed. |
| full    | For experts who seek to maximize the efficiency via TensorFlow, `pip install hanlp[full]` installs every dependency HanLP will use in production. |

## Install models

In short, you don't need to manually install any models. Instead, they are automatically  downloaded to a directory called `HANLP_HOME` when you call `hanlp.load`.
