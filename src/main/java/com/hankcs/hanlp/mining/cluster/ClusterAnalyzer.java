/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-08-12 6:37 PM</create-date>
 *
 * <copyright file="ClusterAnalyzer.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.mining.cluster;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.classification.utilities.TextProcessUtility;
import com.hankcs.hanlp.collection.trie.datrie.MutableDoubleArrayTrieInteger;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.utility.MathUtility;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static com.hankcs.hanlp.classification.utilities.io.ConsoleLogger.logger;

/**
 * 文本聚类
 *
 * @param <K> 文档的id类型
 * @author hankcs
 */
public class ClusterAnalyzer<K>
{
    protected HashMap<K, Document<K>> documents_;
    protected Segment segment;
    protected MutableDoubleArrayTrieInteger vocabulary;
    static final int NUM_REFINE_LOOP = 30;

    public ClusterAnalyzer()
    {
        documents_ = new HashMap<K, Document<K>>();
        segment = HanLP.newSegment();
        vocabulary = new MutableDoubleArrayTrieInteger();
    }

    protected int id(String word)
    {
        int id = vocabulary.get(word);
        if (id == -1)
        {
            id = vocabulary.size();
            vocabulary.put(word, id);
        }
        return id;
    }

    /**
     * 重载此方法实现自己的预处理逻辑（预处理、分词、去除停用词）
     *
     * @param document 文档
     * @return 单词列表
     */
    protected List<String> preprocess(String document)
    {
        List<Term> termList = segment.seg(document);
        ListIterator<Term> listIterator = termList.listIterator();
        while (listIterator.hasNext())
        {
            Term term = listIterator.next();
            if (CoreStopWordDictionary.contains(term.word) ||
                term.nature.startsWith("w")
            )
            {
                listIterator.remove();
            }
        }
        List<String> wordList = new ArrayList<String>(termList.size());
        for (Term term : termList)
        {
            wordList.add(term.word);
        }
        return wordList;
    }

    protected SparseVector toVector(List<String> wordList)
    {
        SparseVector vector = new SparseVector();
        for (String word : wordList)
        {
            int id = id(word);
            Double f = vector.get(id);
            if (f == null)
            {
                f = 1.;
                vector.put(id, f);
            }
            else
            {
                vector.put(id, ++f);
            }
        }
        return vector;
    }

    /**
     * 添加文档
     *
     * @param id       文档id
     * @param document 文档内容
     * @return 文档对象
     */
    public Document<K> addDocument(K id, String document)
    {
        return addDocument(id, preprocess(document));
    }

    /**
     * 添加文档
     *
     * @param id       文档id
     * @param document 文档内容
     * @return 文档对象
     */
    public Document<K> addDocument(K id, List<String> document)
    {
        SparseVector vector = toVector(document);
        Document<K> d = new Document<K>(id, vector);
        return documents_.put(id, d);
    }

    /**
     * k-means聚类
     *
     * @param nclusters 簇的数量
     * @return 指定数量的簇（Set）构成的集合
     */
    public List<Set<K>> kmeans(int nclusters)
    {
        if (nclusters > size())
        {
            logger.err("传入聚类数目%d大于文档数量%d，已纠正为文档数量\n", nclusters, size());
            nclusters = size();
        }
        Cluster<K> cluster = new Cluster<K>();
        for (Document<K> document : documents_.values())
        {
            cluster.add_document(document);
        }
        cluster.section(nclusters);
        refine_clusters(cluster.sectioned_clusters());
        List<Cluster<K>> clusters_ = new ArrayList<Cluster<K>>(nclusters);
        for (Cluster<K> s : cluster.sectioned_clusters())
        {
            s.refresh();
            clusters_.add(s);
        }
        return toResult(clusters_);
    }

    /**
     * 已向聚类分析器添加的文档数量
     *
     * @return 文档数量
     */
    public int size()
    {
        return this.documents_.size();
    }

    private List<Set<K>> toResult(List<Cluster<K>> clusters_)
    {
        List<Set<K>> result = new ArrayList<Set<K>>(clusters_.size());
        for (Cluster<K> c : clusters_)
        {
            Set<K> s = new HashSet<K>();
            for (Document<K> d : c.documents_)
            {
                s.add(d.id_);
            }
            result.add(s);
        }
        return result;
    }

    /**
     * repeated bisection 聚类
     *
     * @param nclusters 簇的数量
     * @return 指定数量的簇（Set）构成的集合
     */
    public List<Set<K>> repeatedBisection(int nclusters)
    {
        return repeatedBisection(nclusters, 0);
    }

    /**
     * repeated bisection 聚类
     *
     * @param limit_eval 准则函数增幅阈值
     * @return 指定数量的簇（Set）构成的集合
     */
    public List<Set<K>> repeatedBisection(double limit_eval)
    {
        return repeatedBisection(0, limit_eval);
    }

    /**
     * repeated bisection 聚类
     *
     * @param nclusters  簇的数量
     * @param limit_eval 准则函数增幅阈值
     * @return 指定数量的簇（Set）构成的集合
     */
    public List<Set<K>> repeatedBisection(int nclusters, double limit_eval)
    {
        if (nclusters > size())
        {
            logger.err("传入聚类数目%d大于文档数量%d，已纠正为文档数量\n", nclusters, size());
            nclusters = size();
        }
        Cluster<K> cluster = new Cluster<K>();
        List<Cluster<K>> clusters_ = new ArrayList<Cluster<K>>(nclusters > 0 ? nclusters : 16);
        for (Document<K> document : documents_.values())
        {
            cluster.add_document(document);
        }

        PriorityQueue<Cluster<K>> que = new PriorityQueue<Cluster<K>>();
        cluster.section(2);
        refine_clusters(cluster.sectioned_clusters());
        cluster.set_sectioned_gain();
        cluster.composite_vector().clear();
        que.add(cluster);

        while (!que.isEmpty())
        {
            if (nclusters > 0 && que.size() >= nclusters)
                break;
            cluster = que.peek();
            if (cluster.sectioned_clusters().size() < 1)
                break;
            if (limit_eval > 0 && cluster.sectioned_gain() < limit_eval)
                break;
            que.poll();
            List<Cluster<K>> sectioned = cluster.sectioned_clusters();

            for (Cluster<K> c : sectioned)
            {
                if (c.size() >= 2)
                {
                    c.section(2);
                    refine_clusters(c.sectioned_clusters());
                    c.set_sectioned_gain();
                    if (c.sectioned_gain() < limit_eval)
                    {
                        for (Cluster<K> sub : c.sectioned_clusters())
                        {
                            sub.clear();
                        }
                    }
                    c.composite_vector().clear();
                }
                que.add(c);
            }
        }
        while (!que.isEmpty())
        {
            clusters_.add(0, que.poll());
        }
        return toResult(clusters_);
    }

    /**
     * 根据k-means算法迭代优化聚类
     *
     * @param clusters 簇
     * @return 准则函数的值
     */
    double refine_clusters(List<Cluster<K>> clusters)
    {
        double[] norms = new double[clusters.size()];
        int offset = 0;
        for (Cluster cluster : clusters)
        {
            norms[offset++] = cluster.composite_vector().norm();
        }

        double eval_cluster = 0.0;
        int loop_count = 0;
        while (loop_count++ < NUM_REFINE_LOOP)
        {
            List<int[]> items = new ArrayList<int[]>(size());
            for (int i = 0; i < clusters.size(); i++)
            {
                for (int j = 0; j < clusters.get(i).documents().size(); j++)
                {
                    items.add(new int[]{i, j});
                }
            }
            Collections.shuffle(items);

            boolean changed = false;
            for (int[] item : items)
            {
                int cluster_id = item[0];
                int item_id = item[1];
                Cluster<K> cluster = clusters.get(cluster_id);
                Document<K> doc = cluster.documents().get(item_id);
                double value_base = refined_vector_value(cluster.composite_vector(), doc.feature(), -1);
                double norm_base_moved = Math.pow(norms[cluster_id], 2) + value_base;
                norm_base_moved = norm_base_moved > 0 ? Math.sqrt(norm_base_moved) : 0.0;

                double eval_max = -1.0;
                double norm_max = 0.0;
                int max_index = 0;
                for (int j = 0; j < clusters.size(); j++)
                {
                    if (cluster_id == j)
                        continue;
                    Cluster<K> other = clusters.get(j);
                    double value_target = refined_vector_value(other.composite_vector(), doc.feature(), 1);
                    double norm_target_moved = Math.pow(norms[j], 2) + value_target;
                    norm_target_moved = norm_target_moved > 0 ? Math.sqrt(norm_target_moved) : 0.0;
                    double eval_moved = norm_base_moved + norm_target_moved - norms[cluster_id] - norms[j];
                    if (eval_max < eval_moved)
                    {
                        eval_max = eval_moved;
                        norm_max = norm_target_moved;
                        max_index = j;
                    }
                }
                if (eval_max > 0)
                {
                    eval_cluster += eval_max;
                    clusters.get(max_index).add_document(doc);
                    clusters.get(cluster_id).remove_document(item_id);
                    norms[cluster_id] = norm_base_moved;
                    norms[max_index] = norm_max;
                    changed = true;
                }
            }
            if (!changed)
                break;
            for (Cluster<K> cluster : clusters)
            {
                cluster.refresh();
            }
        }
        return eval_cluster;
    }

    /**
     * c^2 - 2c(a + c) + d^2 - 2d(b + d)
     *
     * @param composite (a+c,b+d)
     * @param vec       (c,d)
     * @param sign
     * @return
     */
    double refined_vector_value(SparseVector composite, SparseVector vec, int sign)
    {
        double sum = 0.0;
        for (Map.Entry<Integer, Double> entry : vec.entrySet())
        {
            sum += Math.pow(entry.getValue(), 2) + sign * 2 * composite.get(entry.getKey()) * entry.getValue();
        }
        return sum;
    }

    /**
     * 训练模型
     *
     * @param folderPath 分类语料的根目录.目录必须满足如下结构:<br>
     *                   根目录<br>
     *                   ├── 分类A<br>
     *                   │   └── 1.txt<br>
     *                   │   └── 2.txt<br>
     *                   │   └── 3.txt<br>
     *                   ├── 分类B<br>
     *                   │   └── 1.txt<br>
     *                   │   └── ...<br>
     *                   └── ...<br>
     *                   文件不一定需要用数字命名,也不需要以txt作为后缀名,但一定需要是文本文件.
     * @param algorithm  kmeans 或 repeated bisection
     * @throws IOException 任何可能的IO异常
     */
    public static double evaluate(String folderPath, String algorithm)
    {
        if (folderPath == null) throw new IllegalArgumentException("参数 folderPath == null");
        File root = new File(folderPath);
        if (!root.exists()) throw new IllegalArgumentException(String.format("目录 %s 不存在", root.getAbsolutePath()));
        if (!root.isDirectory())
            throw new IllegalArgumentException(String.format("目录 %s 不是一个目录", root.getAbsolutePath()));

        ClusterAnalyzer<String> analyzer = new ClusterAnalyzer<String>();
        File[] folders = root.listFiles();
        if (folders == null) return 1.;
        logger.start("根目录:%s\n加载中...\n", folderPath);
        int docSize = 0;
        int[] ni = new int[folders.length];
        String[] cat = new String[folders.length];
        int offset = 0;
        for (File folder : folders)
        {
            if (folder.isFile()) continue;
            File[] files = folder.listFiles();
            if (files == null) continue;
            String category = folder.getName();
            cat[offset] = category;
            logger.out("[%s]...", category);
            int b = 0;
            int e = files.length;

            int logEvery = (int) Math.ceil((e - b) / 10000f);
            for (int i = b; i < e; i++)
            {
                analyzer.addDocument(folder.getName() + " " + files[i].getName(), IOUtil.readTxt(files[i].getAbsolutePath()));
                if (i % logEvery == 0)
                {
                    logger.out("%c[%s]...%.2f%%", 13, category, MathUtility.percentage(i - b + 1, e - b));
                }
                ++docSize;
                ++ni[offset];
            }
            logger.out(" %d 篇文档\n", e - b);
            ++offset;
        }
        logger.finish(" 加载了 %d 个类目,共 %d 篇文档\n", folders.length, docSize);
        logger.start(algorithm + "聚类中...");
        List<Set<String>> clusterList = algorithm.replaceAll("[-\\s]", "").toLowerCase().equals("kmeans") ?
            analyzer.kmeans(ni.length) : analyzer.repeatedBisection(ni.length);
        logger.finish(" 完毕。\n");
        double[] fi = new double[ni.length];
        for (int i = 0; i < ni.length; i++)
        {
            for (Set<String> j : clusterList)
            {
                int nij = 0;
                for (String d : j)
                {
                    if (d.startsWith(cat[i]))
                        ++nij;
                }
                if (nij == 0) continue;
                double p = nij / (double) (j.size());
                double r = nij / (double) (ni[i]);
                double f = 2 * p * r / (p + r);
                fi[i] = Math.max(fi[i], f);
            }
        }
        double f = 0;
        for (int i = 0; i < fi.length; i++)
        {
            f += fi[i] * ni[i] / docSize;
        }
        return f;
    }
}
