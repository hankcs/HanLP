/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/9 20:55</create-date>
 *
 * <copyright file="ISaveAble.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.dictionary;

/**
 * @author hankcs
 */
public interface ISaveAble
{
    /**
     * 将自己以文本文档的方式保存到磁盘
     * @param path 保存位置，包含文件名，不一定包含后缀
     * @return 是否成功
     */
    public boolean saveTxtTo(String path);
}
