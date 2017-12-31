/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/10/30 19:24</create-date>
 *
 * <copyright file="parser_dll.java" company="码农场">
 * Copyright (c) 2008-2015, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser;

import com.hankcs.hanlp.dependency.nnparser.option.ConfigOption;
import com.hankcs.hanlp.dependency.nnparser.option.SpecialOption;
import com.hankcs.hanlp.utility.GlobalObjectPool;

import java.util.List;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 静态调用的伪 Windows “dll”
 * @author hankcs
 */
public class parser_dll
{
    private NeuralNetworkParser parser;

    public parser_dll()
    {
        this(ConfigOption.PATH);
    }

    public parser_dll(String modelPath)
    {
        parser = GlobalObjectPool.get(modelPath);
        if (parser != null) return;
        parser = new NeuralNetworkParser();
        long start = System.currentTimeMillis();
        logger.info("开始加载神经网络依存句法模型：" + modelPath);
        if (!parser.load(modelPath))
        {
            throw new IllegalArgumentException("加载神经网络依存句法模型[" + modelPath + "]失败！");
        }
        logger.info("加载神经网络依存句法模型[" + modelPath + "]成功，耗时 " + (System.currentTimeMillis() - start) + " ms");
        parser.setup_system();
        parser.build_feature_space();
        GlobalObjectPool.put(modelPath, parser);
    }

    /**
     * 分析句法
     *
     * @param words   词语列表
     * @param postags 词性列表
     * @param heads   输出依存指向列表
     * @param deprels 输出依存名称列表
     * @return 节点的个数
     */
    public int parse(List<String> words, List<String> postags, List<Integer> heads, List<String> deprels)
    {
        Instance inst = new Instance();
        inst.forms.add(SpecialOption.ROOT);
        inst.postags.add(SpecialOption.ROOT);

        for (int i = 0; i < words.size(); i++)
        {
            inst.forms.add(words.get(i));
            inst.postags.add(postags.get(i));
        }

        parser.predict(inst, heads, deprels);
        heads.remove(0);
        deprels.remove(0);

        return heads.size();
    }
}
