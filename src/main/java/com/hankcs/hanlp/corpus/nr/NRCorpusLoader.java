/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/11 12:58</create-date>
 *
 * <copyright file="NRCorpusLoader.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.nr;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.DictionaryMaker;
import com.hankcs.hanlp.corpus.dictionary.item.Item;
import com.hankcs.hanlp.corpus.document.sentence.word.Word;
import com.hankcs.hanlp.corpus.tag.NR;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import static com.hankcs.hanlp.utility.Predefine.logger;
/**
 * 对人名语料的解析，并且生成词典
 * @author hankcs
 */
public class NRCorpusLoader
{
    public static boolean load(String path)
    {
        try
        {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"));
            String line;
            DictionaryMaker dictionaryMaker = new DictionaryMaker();
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
                        Word wordE = new Word(line.substring(1), NR.E.toString());
                        dictionaryMaker.add(wordB);
                        dictionaryMaker.add(wordE);
                        break;
                    }
                    case 3:
                    {
                        Word wordB = new Word(line.substring(0, 1), NR.B.toString());
                        Word wordC = new Word(line.substring(1, 2), NR.C.toString());
                        Word wordD = new Word(line.substring(2, 3), NR.D.toString());
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
            logger.info(dictionaryMaker.toString());
            dictionaryMaker.saveTxtTo("data/dictionary/person/name.txt", new DictionaryMaker.Filter()
            {
                @Override
                public boolean onSave(Item item)
                {
                    return false;
                }
            });
        }
        catch (Exception e)
        {
            logger.warning("读取" + path + "发生错误");
            return false;
        }

        return true;
    }

    public static void combine()
    {
        DictionaryMaker dictionaryMaker = DictionaryMaker.combine(HanLP.Config.CoreDictionaryPath, "XXXDictionary.txt");
        dictionaryMaker.saveTxtTo(HanLP.Config.CoreDictionaryPath);
    }
}
