package com.hankcs.hanlp.model.crf.crfpp;

/**
 * @author zhifac
 */
public abstract class Model
{

    public boolean open(String[] args)
    {
        return true;
    }

    public boolean open(String arg)
    {
        return true;
    }

    public boolean close()
    {
        return true;
    }

    public Tagger createTagger()
    {
        return null;
    }
}
