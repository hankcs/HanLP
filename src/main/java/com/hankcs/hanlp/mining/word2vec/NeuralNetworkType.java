package com.hankcs.hanlp.mining.word2vec;


/**
 * 神经网络类型
 */
public enum NeuralNetworkType
{
    /**
     * 更快，对高频词的准确率更高
     * Faster, slightly better accuracy for frequent words
     */
    CBOW()
            {
                @Override
                public float getDefaultInitialLearningRate()
                {
                    return 0.05f;
                }
            },
    /**
     * 较慢，对低频词的准确率更高
     */
    SKIP_GRAM()
            {
                @Override
                public float getDefaultInitialLearningRate()
                {
                    return 0.025f;
                }
            };

    /**
     * @return 默认学习率
     */
    public abstract float getDefaultInitialLearningRate();
}