# datasets

```{eval-rst}
NLP datasets grouped by tasks. For each task, we provide at least one ``torch.utils.data.Dataset`` compatible class
and several open-source resources. Their file format and description can be found in their ``Dataset.load_file`` 
documents. Their contents are split into ``TRAIN``, ``DEV`` and ``TEST`` portions, each of them is stored in
a Python constant which can be fetched using :meth:`~hanlp.utils.io_util.get_resource`.  
``` 

````{margin} **Professionals use Linux**
```{note}
Many preprocessing scripts written by professionals make heavy use of Linux/Unix tool chains like shell, perl, gcc, 
etc., which is not available or buggy on Windows. You may need a *nix evironment to run these scripts.
```
````

```{toctree}
eos/index
tok/index
pos/index
ner/index
dep/index
srl/index
con/index
```

