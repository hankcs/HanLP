package com.hankcs.hanlp.seg.Viterbi;

import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.File;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 可自定义用户词典的维特比分词器
 */
public class ExtendViterbiSegment extends ViterbiSegment
{
    public ExtendViterbiSegment()
    {
        super();
    }

    /**
     * @param customPath 自定义字典路径（绝对路径，多词典使用英文分号隔开）
     */
    public ExtendViterbiSegment(String customPath)
    {
        super();
        loadCustomDic(customPath, true);
    }

    /**
     * @param customPath customPath 自定义字典路径（绝对路径，多词典使用英文分号隔开）
     * @param cache      是否缓存词典
     */
    public ExtendViterbiSegment(String customPath, boolean cache)
    {
        super();
        loadCustomDic(customPath, cache);
    }

    private void loadCustomDic(String customPath, boolean isCache)
    {
        if (TextUtility.isBlank(customPath))
        {
            return;
        }
        logger.info("自定义词典开始加载:" + customPath);
        DoubleArrayTrie<CoreDictionary.Attribute> dat = new DoubleArrayTrie<CoreDictionary.Attribute>();
        String path[] = customPath.split(";");
        String mainPath = path[0];
        StringBuilder combinePath = new StringBuilder();
        for (String aPath : path)
        {
            combinePath.append(aPath.trim());
        }
        File file = new File(mainPath);
        mainPath = file.getParent() + "/" + Math.abs(combinePath.toString().hashCode());
        mainPath = mainPath.replace("\\", "/");
        if (CustomDictionary.loadMainDictionary(mainPath, path, dat, isCache))
        {
            super.setDat(dat);
        }
    }
}