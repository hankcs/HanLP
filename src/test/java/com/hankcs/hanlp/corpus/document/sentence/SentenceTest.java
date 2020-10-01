package com.hankcs.hanlp.corpus.document.sentence;

import com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord;
import com.hankcs.hanlp.corpus.document.sentence.word.IWord;
import com.hankcs.hanlp.corpus.document.sentence.word.WordFactory;
import junit.framework.TestCase;

import java.util.ListIterator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SentenceTest extends TestCase
{
    public void testFindFirstWordIteratorByLabel() throws Exception
    {
        Sentence sentence = Sentence.create("[上海/ns 华安/nz 工业/n （/w 集团/n ）/w 公司/n]/nt 董事长/n 谭旭光/nr 和/c 秘书/n 胡花蕊/nr 来到/v [美国/ns 纽约/ns 现代/t 艺术/n 博物馆/n]/ns 参观/v");
        ListIterator<IWord> nt = sentence.findFirstWordIteratorByLabel("nt");
        assertNotNull(nt);
        assertEquals("[上海/ns 华安/nz 工业/n （/w 集团/n ）/w 公司/n]/nt", nt.previous().toString());
        CompoundWord apple = CompoundWord.create("[苹果/n 公司/n]/nt");
        nt.set(apple);
        assertEquals(sentence.findFirstWordByLabel("nt"), apple);
        nt.remove();
        assertEquals("董事长/n 谭旭光/nr 和/c 秘书/n 胡花蕊/nr 来到/v [美国/ns 纽约/ns 现代/t 艺术/n 博物馆/n]/ns 参观/v", sentence.toString());
        ListIterator<IWord> ns = sentence.findFirstWordIteratorByLabel("ns");
        assertEquals("参观/v", ns.next().toString());
    }

    public void testToStandoff() throws Exception
    {
        Sentence sentence = Sentence.create("[上海/ns 华安/nz 工业/n （/w 集团/n ）/w 公司/n]/nt 董事长/n 谭旭光/nr 和/c 秘书/n 胡花蕊/nr 来到/v [美国/ns 纽约/ns 现代/t 艺术/n 博物馆/n]/ns 参观/v");
        System.out.println(sentence.toStandoff(true));
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

    public void testMerge() throws Exception
    {
        Sentence sentence = Sentence.create("晚９时４０分/TIME ，/v 鸟/n 迷/v 、/v 专家/n 托尼/PERSON 率领/v 的/u [英国/ns “/w 野翅膀/nz ”/w 观/Vg 鸟/n 团/n]/ORGANIZATION 一行/n ２９/INTEGER 人/n ，/v 才/d 吃/v 完/v 晚饭/n 回到/v [金山/nz 宾馆/n]/ORGANIZATION 的/u 大/a 酒吧间/n ，/v 他们/r 一边/d 喝/v 着/u 青岛/LOCATION 啤酒/n ，/v 一边/d 兴致勃勃/i 地/u 回答/v 记者/n 的/u 提问/vn 。/w");
        System.out.println(sentence.mergeCompoundWords());
    }

    public void testRemoveBracket()
    {
        Sentence sentence = Sentence.create("[关塔那摩/ns]/ns 问题/n 上/f ，/w 美国/nsf 的/ude1 [双重/b 标准/n]/nz");
        for (IWord word : sentence)
        {
            String text = word.getValue();
            if (text.contains("/"))
            {
                System.out.println(word);
                fail();
            }
        }
        assertNotNull(Sentence.create("各/rz [2/m //w 3/m 以上/f]/mq 议员/nnt 赞成/v"));
    }
}