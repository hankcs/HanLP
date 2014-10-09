/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/11 18:04</create-date>
 *
 * <copyright file="NameDictionaryMaker.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.nr;

import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.corpus.tag.NR;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;

/**
 * @author hankcs
 */
public class NameDictionaryMaker
{
    static Logger L = LoggerFactory.getLogger(NRCorpusLoader.class);
    public static DictionaryMaker create(String path)
    {
        DictionaryMaker dictionaryMaker = new DictionaryMaker();
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null)
            {
                if (line.matches(".*[\\p{P}+~$`^=|<>～`$^+=|<>￥×|\\s|a-z0-9A-Z]+.*")) continue;
                // 只载入两字和三字的名字
                Integer length = line.length();
                switch (length)
                {
                    case 2:
                    {
                        Word wordB = new Word(line.substring(0, 1), NR.B.toString());
                        if (!FamilyName.contains(wordB.value)) break;
                        Word wordE = new Word(line.substring(1), NR.E.toString());
                        dictionaryMaker.add(wordB);
                        dictionaryMaker.add(wordE);
                        break;
                    }
                    case 3:
                    {
                        Word wordB = new Word(line.substring(0, 1), NR.B.toString());
                        if (!FamilyName.contains(wordB.value)) break;
                        Word wordC = new Word(line.substring(1, 2), NR.C.toString());
                        Word wordD = new Word(line.substring(2, 3), NR.D.toString());
//                        Word wordC = new Word(line.substring(1, 2), NR.E.toString());
//                        Word wordD = new Word(line.substring(2, 3), NR.E.toString());
                        dictionaryMaker.add(wordB);
                        dictionaryMaker.add(wordC);
                        dictionaryMaker.add(wordD);
                        break;
                    }
                    default:
//                        L.trace("放弃【{}】", line);
                        break;
                }
            }
            br.close();
            L.info(dictionaryMaker.toString());
        }
        catch (Exception e)
        {
            L.warn("读取{}发生错误", path);
            return null;
        }

        return dictionaryMaker;
    }

    public static void main(String[] args)
    {
        DictionaryMaker dictionaryMaker = NameDictionaryMaker.create("data/corpus/authornames.txt");
        dictionaryMaker.saveTxtTo("data/dictionary/person/authornames.txt");
    }
}
