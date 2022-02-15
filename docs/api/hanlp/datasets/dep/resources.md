# resources

## PKU Multiview Treebank

PKU Multi-view Chinese Treebank, released by PKU-ICL. It contains the sentences from People's Daily(19980101-19980110).
The number of sentences in it is 14463.

```{eval-rst}

.. automodule:: hanlp.datasets.parsing.pmt1
    :members:

```

## Chinese Treebank

### CTB5

```{eval-rst}

.. automodule:: hanlp.datasets.parsing.ctb5
    :members:

```

### CTB7

```{eval-rst}

.. automodule:: hanlp.datasets.parsing.ctb7
    :members:

```

### CTB8

```{eval-rst}

.. Attention::

    We propose a new data split for CTB which is different from the academia conventions with the following 3 advantages.
    
    - Easy to reproduce. Files ending with ``8`` go to dev set, ending with ``9`` go to the test set, otherwise go to the training set.
    - Full use of CTB8. The academia conventional split omits 50 gold files while we recall them.
    - More balanced split across genres. Proportions of samples in each genres are similar.
    
    We also use Stanford Dependencies 3.3.0 which offers fine-grained relations and more grammars than the conventional
    head finding rules introduced by :cite:`zhang-clark-2008-tale`.
    
    Therefore, scores on our preprocessed CTB8 are not directly comparable to those in most literatures. We have 
    experimented the same model on the conventionally baked CTB8 and the scores could be 4~5 points higher. 
    We believe it's worthy since HanLP is made for practical purposes, not just for producing pretty numbers.
    
```

````{margin} **Discussion**
```{seealso}
We have a discussion on [our forum](https://bbs.hankcs.com/t/topic/3024).
```
````

```{eval-rst}


.. autodata:: hanlp.datasets.parsing.ctb8.CTB8_SD330_TRAIN
.. autodata:: hanlp.datasets.parsing.ctb8.CTB8_SD330_DEV
.. autodata:: hanlp.datasets.parsing.ctb8.CTB8_SD330_TEST

```

### CTB9

```{eval-rst}

.. Attention::

    Similar preprocessing and splits with CTB8 are applied. See the notice above.
    
```

```{eval-rst}


.. autodata:: hanlp.datasets.parsing.ctb9.CTB9_SD330_TRAIN
.. autodata:: hanlp.datasets.parsing.ctb9.CTB9_SD330_DEV
.. autodata:: hanlp.datasets.parsing.ctb9.CTB9_SD330_TEST

```

## English Treebank

### PTB

```{eval-rst}

.. autodata:: hanlp.datasets.parsing.ptb.PTB_SD330_TRAIN
.. autodata:: hanlp.datasets.parsing.ptb.PTB_SD330_DEV
.. autodata:: hanlp.datasets.parsing.ptb.PTB_SD330_TEST

```

## Universal Dependencies

### Languages

```{eval-rst}

.. automodule:: hanlp.datasets.parsing.ud.ud27
    :members:

```

### Multilingual

```{eval-rst}

.. automodule:: hanlp.datasets.parsing.ud.ud27m
    :members:

```
