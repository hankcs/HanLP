package com.hankcs.hanlp.model.crf.crfpp;


import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.model.perceptron.cli.Args;
import com.hankcs.hanlp.model.perceptron.cli.Argument;

import java.io.InputStream;

/**
 * @author zhifac
 */
public class ModelImpl extends Model
{
    private int nbest_;
    private int vlevel_;
    private DecoderFeatureIndex featureIndex_;

    public ModelImpl()
    {
        nbest_ = vlevel_ = 0;
        featureIndex_ = null;
    }

    public Tagger createTagger()
    {
        if (featureIndex_ == null)
        {
            return null;
        }
        TaggerImpl tagger = new TaggerImpl(TaggerImpl.Mode.TEST);
        tagger.open(featureIndex_, nbest_, vlevel_);
        return tagger;
    }

    public boolean open(String arg)
    {
        return open(arg.split(" ", -1));
    }

    private static class Option
    {
        @Argument(description = "set FILE for model file", alias = "m", required = true)
        String model;
        @Argument(description = "output n-best results", alias = "n")
        Integer nbest = 0;
        @Argument(description = "set INT for verbose level", alias = "v")
        Integer verbose = 0;
        @Argument(description = "set cost factor", alias = "c")
        Double cost_factor = 1.0;
    }

    public boolean open(String[] args)
    {
        Option cmd = new Option();
        try
        {
            Args.parse(cmd, args);
        }
        catch (IllegalArgumentException e)
        {
            System.err.println("invalid arguments");
            return false;
        }
        String model = cmd.model;
        int nbest = cmd.nbest;
        int vlevel = cmd.verbose;
        double costFactor = cmd.cost_factor;
        return open(model, nbest, vlevel, costFactor);
    }

    public boolean open(InputStream stream, int nbest, int vlevel, double costFactor)
    {
        featureIndex_ = new DecoderFeatureIndex();
        nbest_ = nbest;
        vlevel_ = vlevel;
        if (costFactor > 0)
        {
            featureIndex_.setCostFactor_(costFactor);
        }
        return featureIndex_.open(stream);
    }

    public boolean open(String model, int nbest, int vlevel, double costFactor)
    {
        try
        {
            InputStream stream = IOUtil.newInputStream(model);
            return open(stream, nbest, vlevel, costFactor);
        }
        catch (Exception e)
        {
            return false;
        }
    }

    public String getTemplate()
    {
        if (featureIndex_ != null)
        {
            return featureIndex_.getTemplate();
        }
        else
        {
            return null;
        }
    }

    public int getNbest_()
    {
        return nbest_;
    }

    public void setNbest_(int nbest_)
    {
        this.nbest_ = nbest_;
    }

    public int getVlevel_()
    {
        return vlevel_;
    }

    public void setVlevel_(int vlevel_)
    {
        this.vlevel_ = vlevel_;
    }

    public DecoderFeatureIndex getFeatureIndex_()
    {
        return featureIndex_;
    }

    public void setFeatureIndex_(DecoderFeatureIndex featureIndex_)
    {
        this.featureIndex_ = featureIndex_;
    }
}
