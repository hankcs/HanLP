package com.hankcs.hanlp.model.crf.crfpp;

import java.util.List;

/**
 * @author zhifac
 */
public abstract class Tagger
{
    public boolean open(String[] args)
    {
        return true;
    }

    public boolean open(FeatureIndex featureIndex, int nbest, int vlevel, double costFactor)
    {
        return true;
    }

    public boolean open(FeatureIndex featureIndex, int nbest, int vlevel)
    {
        return true;
    }

    public boolean open(String arg)
    {
        return true;
    }

    public boolean add(String[] strArr)
    {
        return true;
    }

    public void close()
    {
    }

    public float[] weightVector()
    {
        return null;
    }

    public boolean add(String str)
    {
        return true;
    }

    public int size()
    {
        return 0;
    }

    public int xsize()
    {
        return 0;
    }

    public int dsize()
    {
        return 0;
    }

    public int result(int i)
    {
        return 0;
    }

    public int answer(int i)
    {
        return 0;
    }

    public int y(int i)
    {
        return result(i);
    }

    public String y2(int i)
    {
        return "";
    }

    public String yname(int i)
    {
        return "";
    }

    public String x(int i, int j)
    {
        return "";
    }

    public int ysize()
    {
        return 0;
    }

    public double prob(int i, int j)
    {
        return 0.0;
    }

    public double prob(int i)
    {
        return 0.0;
    }

    public double prob()
    {
        return 0.0;
    }

    public double alpha(int i, int j)
    {
        return 0.0;
    }

    public double beta(int i, int j)
    {
        return 0.0;
    }

    public double emissionCost(int i, int j)
    {
        return 0.0;
    }

    public double nextTransitionCost(int i, int j, int k)
    {
        return 0.0;
    }

    public double prevTransitionCost(int i, int j, int k)
    {
        return 0.0;
    }

    public double bestCost(int i, int j)
    {
        return 0.0;
    }

    public List<Integer> emissionVector(int i, int j)
    {
        return null;
    }

    public List<Integer> nextTransitionVector(int i, int j, int k)
    {
        return null;
    }

    public List<Integer> prevTransitionVector(int i, int j, int k)
    {
        return null;
    }

    public double Z()
    {
        return 0.0;
    }

    public boolean parse()
    {
        return true;
    }

    public boolean empty()
    {
        return true;
    }

    public boolean clear()
    {
        return true;
    }

    public boolean next()
    {
        return true;
    }

    public String parse(String str)
    {
        return "";
    }

    public String toString()
    {
        return "";
    }

    public String toString(String result, int size)
    {
        return "";
    }

    public String parse(String str, int size)
    {
        return "";
    }

    public String parse(String str, int size1, String result, int size2)
    {
        return "";
    }

    // set token-level penalty. It would be useful for implementing
    // Dual decompositon decoding.
    // e.g.
    // "Dual Decomposition for Parsing with Non-Projective Head Automata"
    // Terry Koo Alexander M. Rush Michael Collins Tommi Jaakkola David Sontag
    public void setPenalty(int i, int j, double penalty)
    {
    }

    public double penalty(int i, int j)
    {
        return 0.0;
    }
}
