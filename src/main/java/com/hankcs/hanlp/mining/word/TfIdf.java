package com.hankcs.hanlp.mining.word;


import java.util.*;

/**
 * 词频-倒排文档词频统计
 */
public class TfIdf
{

    /**
     * 词频统计方式
     */
    public enum TfType
    {
        /**
         * 普通词频
         */
        NATURAL,
        /**
         * 词频的对数并加1
         */
        LOGARITHM,
        /**
         * 01词频
         */
        BOOLEAN
    }

    /**
     * tf-idf 向量的正规化算法
     */
    public enum Normalization
    {
        /**
         * 不正规化
         */
        NONE,
        /**
         * cosine正规化
         */
        COSINE
    }

    /**
     * 单文档词频
     *
     * @param document 词袋
     * @param type     词频计算方式
     * @param <TERM>   词语类型
     * @return 一个包含词频的Map
     */
    public static <TERM> Map<TERM, Double> tf(Collection<TERM> document, TfType type)
    {
        Map<TERM, Double> tf = new HashMap<TERM, Double>();
        for (TERM term : document)
        {
            Double f = tf.get(term);
            if (f == null) f = 0.0;
            tf.put(term, f + 1);
        }
        if (type != TfType.NATURAL)
        {
            for (TERM term : tf.keySet())
            {
                switch (type)
                {
                    case LOGARITHM:
                        tf.put(term, 1 + Math.log(tf.get(term)));
                        break;
                    case BOOLEAN:
                        tf.put(term, tf.get(term) == 0.0 ? 0.0 : 1.0);
                        break;
                }
            }
        }
        return tf;
    }

    /**
     * 单文档词频
     *
     * @param document 词袋
     * @param <TERM>   词语类型
     * @return 一个包含词频的Map
     */
    public static <TERM> Map<TERM, Double> tf(Collection<TERM> document)
    {
        return tf(document, TfType.NATURAL);
    }

    /**
     * 多文档词频
     *
     * @param documents 多个文档，每个文档都是一个词袋
     * @param type      词频计算方式
     * @param <TERM>    词语类型
     * @return 一个包含词频的Map的列表
     */
    public static <TERM> Iterable<Map<TERM, Double>> tfs(Iterable<Collection<TERM>> documents, TfType type)
    {
        List<Map<TERM, Double>> tfs = new ArrayList<Map<TERM, Double>>();
        for (Collection<TERM> document : documents)
        {
            tfs.add(tf(document, type));
        }
        return tfs;
    }

    /**
     * 多文档词频
     *
     * @param documents 多个文档，每个文档都是一个词袋
     * @param <TERM>    词语类型
     * @return 一个包含词频的Map的列表
     */
    public static <TERM> Iterable<Map<TERM, Double>> tfs(Iterable<Collection<TERM>> documents)
    {
        return tfs(documents, TfType.NATURAL);
    }

    /**
     * 一系列文档的倒排词频
     *
     * @param documentVocabularies 词表
     * @param smooth               平滑参数，视作额外有一个文档，该文档含有smooth个每个词语
     * @param addOne               tf-idf加一平滑
     * @param <TERM>               词语类型
     * @return 一个词语->倒排文档的Map
     */
    public static <TERM> Map<TERM, Double> idf(Iterable<Iterable<TERM>> documentVocabularies,
                                               boolean smooth, boolean addOne)
    {
        Map<TERM, Integer> df = new HashMap<TERM, Integer>();
        int d = smooth ? 1 : 0;
        int a = addOne ? 1 : 0;
        int n = d;
        for (Iterable<TERM> documentVocabulary : documentVocabularies)
        {
            n += 1;
            for (TERM term : documentVocabulary)
            {
                Integer t = df.get(term);
                if (t == null) t = d;
                df.put(term, t + 1);
            }
        }
        Map<TERM, Double> idf = new HashMap<TERM, Double>();
        for (Map.Entry<TERM, Integer> e : df.entrySet())
        {
            TERM term = e.getKey();
            double f = e.getValue();
            idf.put(term, Math.log(n / f) + a);
        }
        return idf;
    }

    /**
     * 平滑处理后的一系列文档的倒排词频
     *
     * @param documentVocabularies 词表
     * @param <TERM>               词语类型
     * @return 一个词语->倒排文档的Map
     */
    public static <TERM> Map<TERM, Double> idf(Iterable<Iterable<TERM>> documentVocabularies)
    {
        return idf(documentVocabularies, true, true);
    }

    /**
     * 计算文档的tf-idf
     *
     * @param tf            词频
     * @param idf           倒排频率
     * @param normalization 正规化
     * @param <TERM>        词语类型
     * @return 一个词语->tf-idf的Map
     */
    public static <TERM> Map<TERM, Double> tfIdf(Map<TERM, Double> tf, Map<TERM, Double> idf,
                                                 Normalization normalization)
    {
        Map<TERM, Double> tfIdf = new HashMap<TERM, Double>();
        for (TERM term : tf.keySet())
        {
            Double TF = tf.get(term);
            if (TF == null) TF = 1.;
            Double IDF = idf.get(term);
            if (IDF == null) IDF = 1.;
            tfIdf.put(term, TF * IDF);
        }
        if (normalization == Normalization.COSINE)
        {
            double n = 0.0;
            for (double x : tfIdf.values())
            {
                n += x * x;
            }
            n = Math.sqrt(n);

            for (TERM term : tfIdf.keySet())
            {
                tfIdf.put(term, tfIdf.get(term) / n);
            }
        }
        return tfIdf;
    }

    /**
     * 计算文档的tf-idf（不正规化）
     *
     * @param tf     词频
     * @param idf    倒排频率
     * @param <TERM> 词语类型
     * @return 一个词语->tf-idf的Map
     */
    public static <TERM> Map<TERM, Double> tfIdf(Map<TERM, Double> tf, Map<TERM, Double> idf)
    {
        return tfIdf(tf, idf, Normalization.NONE);
    }

    /**
     * 从词频集合建立倒排频率
     *
     * @param tfs    次品集合
     * @param smooth 平滑参数，视作额外有一个文档，该文档含有smooth个每个词语
     * @param addOne tf-idf加一平滑
     * @param <TERM> 词语类型
     * @return 一个词语->倒排文档的Map
     */
    public static <TERM> Map<TERM, Double> idfFromTfs(Iterable<Map<TERM, Double>> tfs, boolean smooth, boolean addOne)
    {
        return idf(new KeySetIterable<TERM, Double>(tfs), smooth, addOne);
    }

    /**
     * 从词频集合建立倒排频率（默认平滑词频，且加一平滑tf-idf）
     *
     * @param tfs    次品集合
     * @param <TERM> 词语类型
     * @return 一个词语->倒排文档的Map
     */
    public static <TERM> Map<TERM, Double> idfFromTfs(Iterable<Map<TERM, Double>> tfs)
    {
        return idfFromTfs(tfs, true, true);
    }

    /**
     * Map的迭代器
     *
     * @param <KEY>   map 键类型
     * @param <VALUE> map 值类型
     */
    static private class KeySetIterable<KEY, VALUE> implements Iterable<Iterable<KEY>>
    {
        final private Iterator<Map<KEY, VALUE>> maps;

        public KeySetIterable(Iterable<Map<KEY, VALUE>> maps)
        {
            this.maps = maps.iterator();
        }

        @Override
        public Iterator<Iterable<KEY>> iterator()
        {
            return new Iterator<Iterable<KEY>>()
            {
                @Override
                public boolean hasNext()
                {
                    return maps.hasNext();
                }

                @Override
                public Iterable<KEY> next()
                {
                    return maps.next().keySet();
                }

                @Override
                public void remove()
                {

                }
            };
        }
    }
}
