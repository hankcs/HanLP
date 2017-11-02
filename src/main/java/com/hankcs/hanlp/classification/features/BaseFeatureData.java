package com.hankcs.hanlp.classification.features;

import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.classification.corpus.Catalog;
import com.hankcs.hanlp.classification.corpus.Document;
import com.hankcs.hanlp.classification.corpus.IDataSet;
import com.hankcs.hanlp.classification.corpus.Lexicon;

import java.util.Map;

/**
 * 储存所有必需的统计数据,尽量不要存太多东西在这里,因为多个分类器都用这个结构,所以里面的数据仅保留必需的数据
 */
public class BaseFeatureData
{
    /**
     * 样本数量
     */
    public int n;

    /**
     * 一个特征在类目中分别出现几次(键是特征,值的键是类目)
     */
    public int[][] featureCategoryJointCount;

    /**
     * 每个类目中的文档数量
     */
    public int[] categoryCounts;

    /**
     * 新的特征映射表
     */
    public BinTrie<Integer> wordIdTrie;

    /**
     * 构造一个空白的统计对象
     */
    public BaseFeatureData(IDataSet dataSet)
    {
        Catalog catalog = dataSet.getCatalog();
        Lexicon lexicon = dataSet.getLexicon();
        n = dataSet.size();
        featureCategoryJointCount = new int[lexicon.size()][catalog.size()];
        categoryCounts = new int[catalog.size()];

        // 执行统计
        for (Document document : dataSet)
        {
            ++categoryCounts[document.category];

            for (Map.Entry<Integer, int[]> entry : document.tfMap.entrySet())
            {
                featureCategoryJointCount[entry.getKey()][document.category] += 1;
            }
        }
    }
}
