package com.hankcs.hanlp.dictionary.ts;

import com.hankcs.hanlp.HanLP;
import junit.framework.TestCase;

public class TraditionalChineseDictionaryTest extends TestCase
{
    public void testF2J() throws Exception
    {
        assertEquals("草莓是红色的", TraditionalChineseDictionary.convertToSimplifiedChinese("士多啤梨是紅色的"));
    }

    public void testJ2F() throws Exception
    {
        assertEquals("草莓是紅色的", SimplifiedChineseDictionary.convertToTraditionalChinese("草莓是红色的"));
    }

    public void testInterface() throws Exception
    {
        assertEquals("“以后等你当上皇后，就能买草莓庆祝了”", HanLP.convertToSimplifiedChinese("「以後等妳當上皇后，就能買士多啤梨慶祝了」"));
        assertEquals("「以後等你當上皇后，就能買草莓慶祝了」", HanLP.convertToTraditionalChinese("“以后等你当上皇后，就能买草莓庆祝了”"));
    }

    public void testIssue1182() throws Exception
    {
        String content = "直面现实,直面人生,手工捏面人";
        System.out.println(HanLP.s2hk(content));
        System.out.println(HanLP.s2tw(content));
    }

    public void testIssue1184()
    {
        String table = "，斟酒的人翻过大金斗【猛】击代君，一下就砸死 |\t猛 |   勐\n" +
            "校及科研单位挂钩，并【建】立了长期的协作关系 |\t建 |\t创\n" +
            "寇夫人 他自拣一搭金【堦】死。”亦省作“ \u2064|\t堦|\t階\n" +
            "综合兼容性 　　二、【大】众娱乐性 　　三、|\t大|\t福\n" +
            "进行有效的传播控制和【整】合管理。2007年|\t整|\t集\n" +
            "行有效的传播控制和整【合】管理。2007年，|\t合|\t成\n" +
            "有物饮碧水，高林挂青【蜺】。”\",\"ts\":|\t蜺|\t霓\n" +
            "西安市莲湖城内，共计【房】屋231户。\",\"|\t房|\t住\n" +
            "；行程万里的“世界屋【脊】汽车挑战赛”等成功|\t脊|\t嵴\n" +
            "成“全国性”、“全程【式】”的技术创新公共服|\t式|\t序";
        for (String line : table.split("\n"))
        {
            String[] cells = line.split("\\|");
            String text = cells[0].trim().replaceAll("[【】]", "");
            String right = cells[1].trim();
            String wrong = cells[2].trim();
            String hanlpOutput = HanLP.convertToTraditionalChinese(text);
            assertTrue(hanlpOutput.contains(right));
            assertFalse(hanlpOutput.contains(wrong));
        }
    }
}