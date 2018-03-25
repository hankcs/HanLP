package com.hankcs.hanlp.corpus.document.sentence;

import com.hankcs.hanlp.corpus.document.sentence.word.WordFactory;
import junit.framework.TestCase;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SentenceTest extends TestCase
{
    public void testToAnnotation() throws Exception
    {
        Sentence sentence = Sentence.create("[上海/ns 华安/nz 工业/n （/w 集团/n ）/w 公司/n]/nt 董事长/n 谭旭光/nr 和/c 秘书/n 胡花蕊/nr 来到/v [美国/ns 纽约/ns 现代/t 艺术/n 博物馆/n]/ns 参观/v");
        System.out.println(sentence.toStandoff());
    }

    public void testText() throws Exception
    {
        assertEquals("人民网纽约时报", Sentence.create("人民网/nz [纽约/nsf 时报/n]/nz").text());
    }

    public void testCreate() throws Exception
    {
        String text = "人民网/nz 1月1日/t 讯/ng 据/p 《/w [纽约/nsf 时报/n]/nz 》/w 报道/v ，/w";
        Pattern pattern = Pattern.compile("(\\[(.+/[a-z]+)]/[a-z]+)|([^\\s]+/[a-z]+)");
        Matcher matcher = pattern.matcher(text);
        while (matcher.find())
        {
            String param = matcher.group();
            assertEquals(param, WordFactory.create(param).toString());
        }
        assertEquals(text, Sentence.create(text).toString());
    }

    public void testCreateNoTag() throws Exception
    {
        String text = "商品 和 服务";
        Sentence sentence = Sentence.create(text);
        System.out.println(sentence);
    }
}