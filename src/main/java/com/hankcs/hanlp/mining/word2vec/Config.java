package com.hankcs.hanlp.mining.word2vec;

public class Config
{

    static final int DEF_ITER = 5;
    static final int DEF_WINDOW = 5;
    static final int DEF_MIN_COUNT = 5;
    static final int DEF_NEGATIVE = 5;
    static final int DEF_LAYER1_SIZE = 100;
    static final int DEF_NUM_THREADS = Runtime.getRuntime().availableProcessors();
    static final float DEF_SAMPLE = 0.001f;

    protected String outputFile;
    protected int iter = DEF_ITER, window = DEF_WINDOW, minCount = DEF_MIN_COUNT, negative = DEF_NEGATIVE,
            layer1Size = DEF_LAYER1_SIZE, numThreads = DEF_NUM_THREADS;
    protected boolean hs, cbow;
    protected float sample = DEF_SAMPLE, alpha = 0.025f;
    private String inputFile;
    private TrainingCallback callback;

    public Config setOutputFile(String outputFile)
    {
        this.outputFile = outputFile;
        return this;
    }

    public String getOutputFile()
    {
        return outputFile;
    }

    public Config setIter(int iter)
    {
        this.iter = iter;
        return this;
    }

    public int getIter()
    {
        return iter;
    }

    public Config setWindow(int window)
    {
        this.window = window;
        return this;
    }

    public int getWindow()
    {
        return window;
    }

    public Config setMinCount(int minCount)
    {
        this.minCount = minCount;
        return this;
    }

    public int getMinCount()
    {
        return minCount;
    }

    public Config setNegative(int negative)
    {
        this.negative = negative;
        return this;
    }

    public int getNegative()
    {
        return negative;
    }

    public Config setLayer1Size(int layer1Size)
    {
        this.layer1Size = layer1Size;
        return this;
    }

    public int getLayer1Size()
    {
        return layer1Size;
    }

    public Config setNumThreads(int numThreads)
    {
        this.numThreads = numThreads;
        return this;
    }

    public int getNumThreads()
    {
        return numThreads;
    }

    public Config setUseHierarchicalSoftmax(boolean hs)
    {
        this.hs = hs;
        return this;
    }

    public boolean useHierarchicalSoftmax()
    {
        return hs;
    }

    public Config setUseContinuousBagOfWords(boolean cbow)
    {
        this.cbow = cbow;
        return this;
    }

    public boolean useContinuousBagOfWords()
    {
        return cbow;
    }

    public Config setSample(float sample)
    {
        this.sample = sample;
        return this;
    }

    public float getSample()
    {
        return sample;
    }

    public Config setAlpha(float alpha)
    {
        this.alpha = alpha;
        return this;
    }

    public float getAlpha()
    {
        return alpha;
    }

    public Config setInputFile(String inputFile)
    {
        this.inputFile = inputFile;
        return this;
    }

    public String getInputFile()
    {
        return inputFile;
    }

    public TrainingCallback getCallback()
    {
        return callback;
    }

    public void setCallback(TrainingCallback callback)
    {
        this.callback = callback;
    }
}
