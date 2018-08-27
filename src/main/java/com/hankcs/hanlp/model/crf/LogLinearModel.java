/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-28 7:37 PM</create-date>
 *
 * <copyright file="LogLinearModel.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.FileIOAdapter;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.model.perceptron.common.TaskType;
import com.hankcs.hanlp.model.perceptron.feature.FeatureMap;
import com.hankcs.hanlp.model.perceptron.feature.MutableFeatureMap;
import com.hankcs.hanlp.model.perceptron.model.LinearModel;
import com.hankcs.hanlp.model.perceptron.tagset.CWSTagSet;
import com.hankcs.hanlp.model.perceptron.tagset.NERTagSet;
import com.hankcs.hanlp.model.perceptron.tagset.TagSet;
import com.hankcs.hanlp.utility.Predefine;

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.BIN_EXT;
import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 对数线性模型形式的CRF模型
 *
 * @author hankcs
 */
public class LogLinearModel extends LinearModel
{
    /**
     * 特征模板
     */
    private FeatureTemplate[] featureTemplateArray;

    private LogLinearModel(FeatureMap featureMap, float[] parameter)
    {
        super(featureMap, parameter);
    }

    private LogLinearModel(FeatureMap featureMap)
    {
        super(featureMap);
    }

    @Override
    public boolean load(ByteArray byteArray)
    {
        if (!super.load(byteArray)) return false;
        int size = byteArray.nextInt();
        featureTemplateArray = new FeatureTemplate[size];
        for (int i = 0; i < size; ++i)
        {
            FeatureTemplate featureTemplate = new FeatureTemplate();
            featureTemplate.load(byteArray);
            featureTemplateArray[i] = featureTemplate;
        }
        if (!byteArray.hasMore())
            byteArray.close();
        return true;
    }

    /**
     * 加载CRF模型
     *
     * @param modelFile HanLP的.bin格式，或CRF++的.txt格式（将会自动转换为model.txt.bin，下次会直接加载.txt.bin）
     * @throws IOException
     */
    public LogLinearModel(String modelFile) throws IOException
    {
        super(null, null);
        if (modelFile.endsWith(BIN_EXT))
        {
            load(modelFile); // model.bin
            return;
        }
        String binPath = modelFile + Predefine.BIN_EXT;

        if (!((HanLP.Config.IOAdapter == null || HanLP.Config.IOAdapter instanceof FileIOAdapter) && !IOUtil.isFileExisted(binPath)))
        {
            try
            {
                load(binPath); // model.txt -> model.bin
                return;
            }
            catch (Exception e)
            {
                // ignore
            }
        }

        convert(modelFile, binPath);
    }

    /**
     * 加载txt，转换为bin
     *
     * @param txtFile txt
     * @param binFile bin
     * @throws IOException
     */
    public LogLinearModel(String txtFile, String binFile) throws IOException
    {
        super(null, null);
        convert(txtFile, binFile);
    }

    private void convert(String txtFile, String binFile) throws IOException
    {
        TagSet tagSet = new TagSet(TaskType.CLASSIFICATION);
        IOUtil.LineIterator lineIterator = new IOUtil.LineIterator(txtFile);
        if (!lineIterator.hasNext()) throw new IOException("空白文件");
        logger.info(lineIterator.next());   // verson
        logger.info(lineIterator.next());   // cost-factor
        int maxid = Integer.parseInt(lineIterator.next().substring("maxid:".length()).trim());
        logger.info(lineIterator.next());   // xsize
        lineIterator.next();    // blank
        String line;
        while ((line = lineIterator.next()).length() != 0)
        {
            tagSet.add(line);
        }
        tagSet.type = guessModelType(tagSet);
        switch (tagSet.type)
        {
            case CWS:
                tagSet = new CWSTagSet(tagSet.idOf("B"), tagSet.idOf("M"), tagSet.idOf("E"), tagSet.idOf("S"));
                break;
            case NER:
                tagSet = new NERTagSet(tagSet.idOf("O"), tagSet.tags());
                break;
        }
        tagSet.lock();
        this.featureMap = new MutableFeatureMap(tagSet);
        FeatureMap featureMap = this.featureMap;
        final int sizeOfTagSet = tagSet.size();
        TreeMap<String, FeatureFunction> featureFunctionMap = new TreeMap<String, FeatureFunction>();  // 构建trie树的时候用
        TreeMap<Integer, FeatureFunction> featureFunctionList = new TreeMap<Integer, FeatureFunction>(); // 读取权值的时候用
        ArrayList<FeatureTemplate> featureTemplateList = new ArrayList<FeatureTemplate>();
        float[][] matrix = null;
        while ((line = lineIterator.next()).length() != 0)
        {
            if (!"B".equals(line))
            {
                FeatureTemplate featureTemplate = FeatureTemplate.create(line);
                featureTemplateList.add(featureTemplate);
            }
            else
            {
                matrix = new float[sizeOfTagSet][sizeOfTagSet];
            }
        }
        this.featureTemplateArray = featureTemplateList.toArray(new FeatureTemplate[0]);

        int b = -1;// 转换矩阵的权重位置
        if (matrix != null)
        {
            String[] args = lineIterator.next().split(" ", 2);    // 0 B
            b = Integer.valueOf(args[0]);
            featureFunctionList.put(b, null);
        }

        while ((line = lineIterator.next()).length() != 0)
        {
            String[] args = line.split(" ", 2);
            char[] charArray = args[1].toCharArray();
            FeatureFunction featureFunction = new FeatureFunction(charArray, sizeOfTagSet);
            featureFunctionMap.put(args[1], featureFunction);
            featureFunctionList.put(Integer.parseInt(args[0]), featureFunction);
        }

        for (Map.Entry<Integer, FeatureFunction> entry : featureFunctionList.entrySet())
        {
            int fid = entry.getKey();
            FeatureFunction featureFunction = entry.getValue();
            if (fid == b)
            {
                for (int i = 0; i < sizeOfTagSet; i++)
                {
                    for (int j = 0; j < sizeOfTagSet; j++)
                    {
                        matrix[i][j] = Float.parseFloat(lineIterator.next());
                    }
                }
            }
            else
            {
                for (int i = 0; i < sizeOfTagSet; i++)
                {
                    featureFunction.w[i] = Double.parseDouble(lineIterator.next());
                }
            }
        }
        if (lineIterator.hasNext())
        {
            logger.warning("文本读取有残留，可能会出问题！" + txtFile);
        }
        lineIterator.close();
        logger.info("文本读取结束，开始转换模型");
        int transitionFeatureOffset = (sizeOfTagSet + 1) * sizeOfTagSet;
        parameter = new float[transitionFeatureOffset + featureFunctionMap.size() * sizeOfTagSet];
        if (matrix != null)
        {
            for (int i = 0; i < sizeOfTagSet; ++i)
            {
                for (int j = 0; j < sizeOfTagSet; ++j)
                {
                    parameter[i * sizeOfTagSet + j] = matrix[i][j];
                }
            }
        }
        for (Map.Entry<Integer, FeatureFunction> entry : featureFunctionList.entrySet())
        {
            int id = entry.getKey();
            FeatureFunction f = entry.getValue();
            if (f == null) continue;
            String feature = new String(f.o);
            for (int tid = 0; tid < featureTemplateList.size(); tid++)
            {
                FeatureTemplate template = featureTemplateList.get(tid);
                Iterator<String> iterator = template.delimiterList.iterator();
                String header = iterator.next();
                if (feature.startsWith(header))
                {
                    int fid = featureMap.idOf(feature.substring(header.length()) + tid);
//                    assert id == sizeOfTagSet * sizeOfTagSet + (fid - sizeOfTagSet - 1) * sizeOfTagSet;
                    for (int i = 0; i < sizeOfTagSet; ++i)
                    {
                        parameter[fid * sizeOfTagSet + i] = (float) f.w[i];
                    }
                    break;
                }
            }
        }
        DataOutputStream out = new DataOutputStream(IOUtil.newOutputStream(binFile));
        save(out);
        out.writeInt(featureTemplateList.size());
        for (FeatureTemplate template : featureTemplateList)
        {
            template.save(out);
        }
        out.close();
    }


    private TaskType guessModelType(TagSet tagSet)
    {
        if (tagSet.size() == 4 &&
            tagSet.idOf("B") != -1 &&
            tagSet.idOf("M") != -1 &&
            tagSet.idOf("E") != -1 &&
            tagSet.idOf("S") != -1
            )
        {
            return TaskType.CWS;
        }
        if (tagSet.idOf("O") != -1)
        {
            for (String tag : tagSet.tags())
            {
                String[] parts = tag.split("-");
                if (parts.length > 1)
                {
                    if (parts[0].length() == 1 && "BMES".contains(parts[0]))
                        return TaskType.NER;
                }
            }
        }
        return TaskType.POS;
    }

    public FeatureTemplate[] getFeatureTemplateArray()
    {
        return featureTemplateArray;
    }
}
