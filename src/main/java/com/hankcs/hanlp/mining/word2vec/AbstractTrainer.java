package com.hankcs.hanlp.mining.word2vec;



public abstract class AbstractTrainer
{

    protected abstract void localUsage();

    protected void paramDesc(String param, String desc)
    {
        System.err.printf("\t%s\n\t\t%s\n", param, desc);
    }

    protected void usage()
    {
        System.err.printf("word2vec Java toolkit v 0.1c\n\n");
        System.err.printf("Options:\n");
        System.err.printf("Parameters for training:\n");
        paramDesc("-output <file>", "Use <file> to save the resulting word vectors / word clusters");
        paramDesc("-size <int>", "Set size of word vectors; default is 100");
        paramDesc("-window <int>", "Set max skip length between words; default is 5");
        paramDesc("-sample <float>", "Set threshold for occurrence of words. Those that appear with higher frequency in the training data" +
                " will be randomly down-sampled; default is 0.001, useful range is (0, 0.00001)");
        paramDesc("-hs", "Use Hierarchical Softmax; default is not used");
        paramDesc("-negative <int>", "Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)");
        paramDesc("-threads <int>", "Use <int> threads (default is the cores of local machine)");
        paramDesc("-iter <int>", "Run more training iterations (default 5)");
        paramDesc("-min-count <int>", "This will discard words that appear less than <int> times; default is 5");
        paramDesc("-alpha <float>", "Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW");
        paramDesc("-cbow", "Use the continuous bag of words model; default is skip-gram model");

        localUsage();

        System.exit(0);
    }

    protected int argPos(String param, String[] args)
    {
        return argPos(param, args, true);
    }

    protected int argPos(String param, String[] args, boolean checkArgNum)
    {
        for (int i = 0; i < args.length; i++)
        {
            if (param.equals(args[i]))
            {
                if (checkArgNum && (i == args.length - 1))
                    throw new IllegalArgumentException(String.format("Argument missing for %s", param));
                return i;
            }
        }
        return -1;
    }

    protected void setConfig(String[] args, Config config)
    {
        int i;
        if ((i = argPos("-size", args)) >= 0) config.setLayer1Size(Integer.parseInt(args[i + 1]));
        if ((i = argPos("-output", args)) >= 0) config.setOutputFile(args[i + 1]);
        if ((i = argPos("-cbow", args)) >= 0) config.setUseContinuousBagOfWords(Integer.parseInt(args[i + 1]) == 1);
        if (config.useContinuousBagOfWords()) config.setAlpha(0.05f);
        if ((i = argPos("-alpha", args)) >= 0) config.setAlpha(Float.parseFloat(args[i + 1]));
        if ((i = argPos("-window", args)) >= 0) config.setWindow(Integer.parseInt(args[i + 1]));
        if ((i = argPos("-sample", args)) >= 0) config.setSample(Float.parseFloat(args[i + 1]));
        if ((i = argPos("-hs", args)) >= 0) config.setUseHierarchicalSoftmax(Integer.parseInt(args[i + 1]) == 1);
        if ((i = argPos("-negative", args)) >= 0) config.setNegative(Integer.parseInt(args[i + 1]));
        if ((i = argPos("-threads", args)) >= 0) config.setNumThreads(Integer.parseInt(args[i + 1]));
        if ((i = argPos("-iter", args)) >= 0) config.setIter(Integer.parseInt(args[i + 1]));
        if ((i = argPos("-min-count", args)) >= 0) config.setMinCount(Integer.parseInt(args[i + 1]));
    }
}
