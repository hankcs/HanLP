/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/12 14:03</create-date>
 *
 * <copyright file="TestMakeTranslateName.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.dictionary.nr.TranslatedPersonDictionary;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.StandardTokenizer;
import junit.framework.TestCase;

import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author hankcs
 */
public class TestMakeTranslateName extends TestCase
{
    public void testCombineOuterDictionary() throws Exception
    {
        String root = "D:\\JavaProjects\\SougouDownload\\data\\";
        String[] pathArray = new String[]{"常用外国人名.txt", "外国人名", "外国姓名大全.txt", "外国诗人名.txt", "英语姓名词典.txt", "俄罗斯人名.txt"};
        Set<String> wordSet = new TreeSet<String>();
        for (String path : pathArray)
        {
            path = root + path;
            for (String word : IOUtil.readLineList(path))
            {
                word = word.replaceAll("[a-z]", "");
                if (CoreDictionary.contains(word) || CustomDictionary.contains(word)) continue;
                wordSet.add(word);
            }
        }
        IOUtil.saveCollectionToTxt(wordSet, "data/dictionary/person/nrf.txt");
    }

    public void testSpiltToChar() throws Exception
    {
        String commonChar = "·-—阿埃艾爱安昂敖奥澳笆芭巴白拜班邦保堡鲍北贝本比毕彼别波玻博勃伯泊卜布才采仓查差柴彻川茨慈次达大戴代丹旦但当道德得登迪狄蒂帝丁东杜敦多额俄厄鄂恩尔伐法范菲芬费佛夫福弗甫噶盖干冈哥戈革葛格各根古瓜哈海罕翰汗汉豪合河赫亨侯呼胡华霍基吉及加贾坚简杰金京久居君喀卡凯坎康考柯科可克肯库奎拉喇莱来兰郎朗劳勒雷累楞黎理李里莉丽历利立力连廉良列烈林隆卢虏鲁路伦仑罗洛玛马买麦迈曼茅茂梅门蒙盟米蜜密敏明摩莫墨默姆木穆那娜纳乃奈南内尼年涅宁纽努诺欧帕潘畔庞培佩彭皮平泼普其契恰强乔切钦沁泉让热荣肉儒瑞若萨塞赛桑瑟森莎沙山善绍舍圣施诗石什史士守斯司丝苏素索塔泰坦汤唐陶特提汀图土吐托陀瓦万王旺威韦维魏温文翁沃乌吾武伍西锡希喜夏相香歇谢辛新牙雅亚彦尧叶依伊衣宜义因音英雍尤于约宰泽增詹珍治中仲朱诸卓孜祖佐伽娅尕腓滕济嘉津赖莲琳律略慕妮聂裴浦奇齐琴茹珊卫欣逊札哲智兹芙汶迦珀琪梵斐胥黛" +
                "·-阿安奥巴比彼波布察茨大德得丁杜尔法夫伏甫盖格哈基加坚捷金卡科可克库拉莱兰勒雷里历利连列卢鲁罗洛马梅蒙米姆娜涅宁诺帕泼普奇齐乔切日萨色山申什斯索塔坦特托娃维文乌西希谢亚耶叶依伊以扎佐柴达登蒂戈果海赫华霍吉季津柯理琳玛曼穆纳尼契钦丘桑沙舍泰图瓦万雅卓兹" +
                "-·—丁万丘东丝中丹丽乃久义乌乔买于亚亨京什仑仓代以仲伊伍伏伐伦伯伽但佐佛佩依侯俄保儒克兰其兹内冈凯切列利别力加努劳勃勒北华卓南博卜卡卢卫厄历及古可史叶司各合吉吐君吾呼哈哥哲唐喀善喇喜嘉噶因图土圣坎坚坦埃培基堡塔塞增墨士夏多大夫奇奈奎契奥妮姆威娃娅娜孜季宁守安宜宰密察尔尕尤尧尼居山川差巴布希帕帝干平年库庞康廉弗强当彦彭彻彼律得德恩恰慈慕戈戴才扎托拉拜捷提摩敏敖敦文斐斯新施日旦旺昂明普智曼朗木本札朱李杜来杰林果查柯柴根格桑梅梵森楞次欣欧歇武比毕汀汉汗汤汶沁沃沙河治泉泊法波泰泼泽洛津济浦海涅温滕潘澳烈热爱牙特狄王玛玻珀珊珍班理琪琳琴瑞瑟瓜瓦甫申畔略登白皮盖盟相石祖福科穆立笆简米素索累约纳纽绍维罕罗翁翰考耶聂肉肯胡胥腓舍良色艾芙芬芭苏若英茂范茅茨茹荣莉莎莫莱莲菲萨葛蒂蒙虏蜜衣裴西詹让诗诸诺谢豪贝费贾赖赛赫路辛达迈连迦迪逊道那邦郎鄂采里金钦锡门阿陀陶隆雅雍雷霍革韦音额香马魏鲁鲍麦黎默黛齐" +
                "·—阿埃艾爱安昂敖奥澳笆芭巴白拜班邦保堡鲍北贝本比毕彼别波玻博勃伯泊卜布才采仓查差柴彻川茨慈次达大戴代丹旦但当道德得的登迪狄蒂帝丁东杜敦多额俄厄鄂恩尔伐法范菲芬费佛夫福弗甫噶盖干冈哥戈革葛格各根古瓜哈海罕翰汗汉豪合河赫亨侯呼胡华霍基吉及加贾坚简杰金京久居君喀卡凯坎康考柯科可克肯库奎拉喇莱来兰郎朗劳勒雷累楞黎理李里莉丽历利立力连廉良列烈林隆卢虏鲁路伦仑罗洛玛马买麦迈曼茅茂梅门蒙盟米蜜密敏明摩莫墨默姆木穆那娜纳乃奈南内尼年涅宁纽努诺欧帕潘畔庞培佩彭皮平泼普其契恰强乔切钦沁泉让热荣肉儒瑞若萨塞赛桑瑟森莎沙山善绍舍圣施诗石什史士守斯司丝苏素索塔泰坦汤唐陶特提汀图土吐托陀瓦万王旺威韦维魏温文翁沃乌吾武伍西锡希喜夏相香歇谢辛新牙雅亚彦尧叶依伊衣宜义因音英雍尤于约宰泽增詹珍治中仲朱诸卓孜祖佐伽娅尕腓滕济嘉津赖莲琳律略慕妮聂裴浦奇齐琴茹珊卫欣逊札哲智兹芙汶迦珀琪梵斐胥黛" +
                "·阿安奥巴比彼波布察茨大德得丁杜尔法夫伏甫盖格哈基加坚捷金卡科可克库拉莱兰勒雷里历利连列卢鲁罗洛马梅蒙米姆娜涅宁诺帕泼普奇齐乔切日萨色山申什斯索塔坦特托娃维文乌西希谢亚耶叶依伊以扎佐柴达登蒂戈果海赫华霍吉季津柯理琳玛曼穆纳尼契钦丘桑沙舍泰图瓦万雅卓兹";
        Set<String> wordSet = new TreeSet<String>();
        LinkedList<String> wordList = IOUtil.readLineList("data/dictionary/person/nrf.txt");
        wordList.add(commonChar);
        for (String word : wordList)
        {
            word = word.replaceAll("\\s", "");
            for (char c : word.toCharArray())
            {
                wordSet.add(String.valueOf(c));
            }
        }
        IOUtil.saveCollectionToTxt(wordSet, "data/dictionary/person/音译用字.txt");
    }

    public void testQuery() throws Exception
    {
        HanLP.Config.enableDebug();
        System.out.println(TranslatedPersonDictionary.containsKey("汤姆"));
        System.out.println(TranslatedPersonDictionary.containsKey("汤"));
        System.out.println(TranslatedPersonDictionary.containsKey("姆"));
        System.out.println(TranslatedPersonDictionary.containsKey("点"));
        System.out.println(TranslatedPersonDictionary.containsKey("·"));
    }

    public void testSeg() throws Exception
    {
        HanLP.Config.enableDebug();
        System.out.println(StandardTokenizer.segment("齐格林斯基"));
    }

    public void testNonRec() throws Exception
    {
        HanLP.Config.enableDebug();
        DijkstraSegment segment = new DijkstraSegment();
        segment.enableTranslatedNameRecognize(true);
        System.out.println(segment.seg("汤姆和杰克逊"));
    }

    public void testHeadNRF() throws Exception
    {
        DijkstraSegment segment = new DijkstraSegment();
        segment.enableTranslatedNameRecognize(false);
        for (String name : IOUtil.readLineList("data/dictionary/person/nrf.txt"))
        {
            List<Term> termList = segment.seg(name);
            if (termList.get(0).nature != Nature.nrf)
            {
                System.out.println(name + " : " + termList);
            }
        }
    }

    public void testDot() throws Exception
    {
        char c1 = '·';
        char c2 = '·';
        System.out.println(c1 == c2);
    }

    public void testMakeDictionary() throws Exception
    {
        Set<String> wordSet = new TreeSet<String>();
        Pattern pattern = Pattern.compile("^[a-zA-Z]+ *(\\[.*?])? *([\\u4E00-\\u9FA5]+) ?[:：。]");
        int found = 0;
        for (String line : IOUtil.readLineList("D:\\Doc\\语料库\\英语姓名词典.txt"))
        {
            Matcher matcher = pattern.matcher(line);
            if (matcher.find())
            {
                wordSet.add(matcher.group(2));
                ++found;
            }
        }
        System.out.println("一共找到" + found + "条");
        IOUtil.saveCollectionToTxt(wordSet, "data/dictionary/person/英语姓名词典.txt");
    }

    public void testRegex() throws Exception
    {
        Pattern pattern = Pattern.compile("^[a-zA-Z]+ (\\[.*?])? ?([\\u4E00-\\u9FA5]+) ?[:：。]");
        String text = "Adey 阿迪：Adam的昵称，英格兰人姓氏 \n" +
                "Adkin 阿德金:Adarn的昵称，英格兰人姓氏。 \n" +
                "Adkins 阿德金斯:取自父名，源自Adkin,含义“阿德金之子”(son of Adkin)，英格兰人姓氏 \n" +
                "Adlam [英格兰人姓氏] 阿德拉姆。来源于日耳曼语人名，含义是“高贵的+保护，头盔”(noble＋protection，helmet) \n" +
                "Zena [女子名] 齐娜。来源于波斯语，含义是“女人”(woman)。 \n" +
                "Zenas [男子名] 泽纳斯。来源于希腊语，含义是“希腊主神宙斯的礼物”(gift of Zeus，the chief Greek god)。 \n" +
                "Zenia [女子名]齐尼娅：Xeniq的变体。 \n" +
                "Zenobia [女子名] 泽诺比娅。来源于希腊语，含义是“希腊主神宙斯+生命”(the chief Greek god Zeus+life)。 \n" +
                "Zillah [女子名] 齐拉。来源于希伯来语，含义是“荫”(shade)。 \n" +
                "Zoe [女子名]佐伊：来源于希腊语，含义是“生命”（life）。 \n" +
                "Zouch [英格兰人姓氏] 朱什。Such的变体。 ";

        Matcher matcher = pattern.matcher(text);
        if (matcher.find())
        {
            System.out.println(matcher.group(2));
        }
    }

    public void testCombineCharAndName() throws Exception
    {
        TreeSet<String> wordSet = new TreeSet<String>();
        wordSet.addAll(IOUtil.readLineList("data/dictionary/person/音译用字.txt"));
        wordSet.addAll(IOUtil.readLineList("data/dictionary/person/nrf.txt"));
        IOUtil.saveCollectionToTxt(wordSet, "data/dictionary/person/nrf.txt");
    }
}
