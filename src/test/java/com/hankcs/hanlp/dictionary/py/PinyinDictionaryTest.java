package com.hankcs.hanlp.dictionary.py;

import com.hankcs.hanlp.HanLP;
import junit.framework.TestCase;

import java.util.Arrays;

public class PinyinDictionaryTest extends TestCase
{

    public void testGet()
    {
        System.out.println(Arrays.toString(PinyinDictionary.get("鼖")));
        System.out.println(PinyinDictionary.convertToPinyin("\uD867\uDF7E\uD867\uDF8C"));
        System.out.println(HanLP.convertToPinyinList("\uD867\uDF7E\uD867\uDF8C"));
    }
}