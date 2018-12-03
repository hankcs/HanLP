package com.hankcs.hanlp.seg.Viterbi;

import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.utility.LexiconUtility;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 可自定义用户词典的维特比分词器
 */
public class ExtendViterbiSegment extends ViterbiSegment
{
    public ExtendViterbiSegment()
    {
        super();
        super.setDat(CustomDictionary.dat);
    }

    /**
     * @param customPath 自定义字典路径（绝对路径，多词典使用英文分号隔开）
     */
    public ExtendViterbiSegment(String customPath)
    {
        super();
        if (!TextUtility.isBlank(customPath))
        {
            loadCustomDic(customPath, true);
        }
    }

    /**
     * @param customPath customPath 自定义字典路径（绝对路径，多词典使用英文分号隔开）
     * @param cache      是否缓存词典
     */
    public ExtendViterbiSegment(String customPath, boolean cache)
    {
        super();
        if (!TextUtility.isBlank(customPath))
        {
            loadCustomDic(customPath, cache);
        }
    }

    private void loadCustomDic(String customPath, boolean cache)
    {
        logger.info("自定义词典开始加载:" + customPath);
        DoubleArrayTrie<CoreDictionary.Attribute> dat = new DoubleArrayTrie<CoreDictionary.Attribute>();
        TreeMap<String, CoreDictionary.Attribute> map = new TreeMap<String, CoreDictionary.Attribute>();
        LinkedHashSet<Nature> customNatureCollector = new LinkedHashSet<Nature>();
        try
        {
            String path[] = customPath.split(";");
            String mainPath = "";
            for (String p : path)
            {
                p = p.trim();
                Nature defaultNature = Nature.n;
                File file = new File(p);
                String fileName = file.getName();
                int cut = fileName.lastIndexOf(' ');
                if (cut > 0)
                {
                    // 有默认词性
                    String nature = fileName.substring(cut + 1).trim();
                    p = file.getParent() + File.separator + fileName.substring(0, cut);
                    try
                    {
                        defaultNature = LexiconUtility.convertStringToNature(nature, customNatureCollector);
                    }
                    catch (Exception e)
                    {
                        logger.severe("配置文件【" + p + "】写错了！" + e);
                        continue;
                    }
                }
                if (TextUtility.isBlank(mainPath))
                {
                    mainPath = p;
                    if (CustomDictionary.loadDat(mainPath, path, dat))
                    {
                        super.setDat(dat);
                        return;
                    }
                }
                logger.info("以默认词性[" + defaultNature + "]加载自定义词典" + p + "中……");
                boolean success = CustomDictionary.load(p, defaultNature, map, customNatureCollector);
                if (!success) logger.warning("失败：" + p);
            }
            if (map.size() == 0)
            {
                logger.warning("没有加载到任何词条，使用默认词典");
                super.setDat(CustomDictionary.dat);
                return;
            }
            logger.info("正在构建DoubleArrayTrie……");
            dat.build(map);
            if (cache)
            {
                // 缓存成dat文件，下次加载会快很多
                logger.info("正在缓存词典为dat文件……");
                // 缓存值文件
                List<CoreDictionary.Attribute> attributeList = new LinkedList<CoreDictionary.Attribute>();
                for (Map.Entry<String, CoreDictionary.Attribute> entry : map.entrySet())
                {
                    attributeList.add(entry.getValue());
                }
                DataOutputStream out = new DataOutputStream(new BufferedOutputStream(IOUtil.newOutputStream(mainPath + Predefine.BIN_EXT)));
                // 缓存用户词性
                if (customNatureCollector.isEmpty()) // 热更新
                {
                    for (int i = Nature.begin.ordinal() + 1; i < Nature.values().length; ++i)
                    {
                        customNatureCollector.add(Nature.values()[i]);
                    }
                }
                IOUtil.writeCustomNature(out, customNatureCollector);
                // 缓存正文
                out.writeInt(attributeList.size());
                for (CoreDictionary.Attribute attribute : attributeList)
                {
                    attribute.save(out);
                }
                dat.save(out);
                out.close();
            }
            super.setDat(dat);
        }
        catch (Exception e)
        {
            logger.warning("自定义词典" + customPath + "缓存失败！\n" + TextUtility.exceptionToString(e));
            super.setDat(CustomDictionary.dat);
        }
    }
}