/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/9 21:00</create-date>
 *
 * <copyright file="FeatureTemplate.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.model.crf;

import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.corpus.io.ICacheAble;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 特征模板
 * @author hankcs
 */
public class FeatureTemplate implements ICacheAble
{
    /**
     * 用来解析模板的正则表达式
     */
    static final Pattern pattern = Pattern.compile("%x\\[(-?\\d*),(\\d*)]");
    String template;
    /**
     * 每个部分%x[-2,0]的位移，其中int[0]储存第一个数（-2），int[1]储存第二个数（0）
     */
    ArrayList<int[]> offsetList;
    List<String> delimiterList;

    public FeatureTemplate()
    {
    }

    public static FeatureTemplate create(String template)
    {
        FeatureTemplate featureTemplate = new FeatureTemplate();
        featureTemplate.delimiterList = new LinkedList<String>();
        featureTemplate.offsetList = new ArrayList<int[]>(3);
        featureTemplate.template = template;
        Matcher matcher = pattern.matcher(template);
        int start = 0;
        while (matcher.find())
        {
            featureTemplate.delimiterList.add(template.substring(start, matcher.start()));
            start = matcher.end();
            featureTemplate.offsetList.add(new int[]{Integer.parseInt(matcher.group(1)), Integer.parseInt(matcher.group(2))});
        }
        return featureTemplate;
    }

    public char[] generateParameter(Table table, int current)
    {
        StringBuilder sb = new StringBuilder();
        int i = 0;
        for (String d : delimiterList)
        {
            sb.append(d);
            int[] offset = offsetList.get(i++);
            sb.append(table.get(current + offset[0], offset[1]));
        }

        char[] o = new char[sb.length()];
        sb.getChars(0, sb.length(), o, 0);

        return o;
    }

    @Override
    public void save(DataOutputStream out) throws IOException
    {
        out.writeUTF(template);
        out.writeInt(offsetList.size());
        for (int[] offset : offsetList)
        {
            out.writeInt(offset[0]);
            out.writeInt(offset[1]);
        }
        out.writeInt(delimiterList.size());
        for (String s : delimiterList)
        {
            out.writeUTF(s);
        }
    }

    @Override
    public boolean load(ByteArray byteArray)
    {
        template = byteArray.nextUTF();
        int size = byteArray.nextInt();
        offsetList = new ArrayList<int[]>(size);
        for (int i = 0; i < size; ++i)
        {
            offsetList.add(new int[]{byteArray.nextInt(), byteArray.nextInt()});
        }
        size = byteArray.nextInt();
        delimiterList = new ArrayList<String>(size);
        for (int i = 0; i < size; ++i)
        {
            delimiterList.add(byteArray.nextUTF());
        }
        return true;
    }

    @Override
    public String toString()
    {
        final StringBuilder sb = new StringBuilder("FeatureTemplate{");
        sb.append("template='").append(template).append('\'');
        sb.append(", delimiterList=").append(delimiterList);
        sb.append('}');
        return sb.toString();
    }

    public String getTemplate()
    {
        return template;
    }
}
