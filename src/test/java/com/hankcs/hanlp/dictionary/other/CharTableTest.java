package com.hankcs.hanlp.dictionary.other;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.io.IOUtil;
import junit.framework.TestCase;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.Map;

public class CharTableTest extends TestCase
{
    public void testNormalization() throws Exception
    {
        System.out.println(CharTable.convert('？'));
        assertEquals('(', CharTable.convert('（'));
    }

    public void testNormalizeSpace() throws Exception
    {
        assertEquals(CharTable.convert('\t'), ' ');
        assertEquals(CharTable.convert('\n'), ' ');
        assertEquals(CharTable.convert('\f'), ' ');
    }

    public void testIssue1615()
    {
        new File("data/dictionary/other/CharTable.txt.bin").delete();
        Map<String, String> normalizationBadCase = new HashMap<String, String>();
        normalizationBadCase.put("猛", "猛");
        normalizationBadCase.put("蜺", "蜺");
        normalizationBadCase.put("脊", "脊");
        normalizationBadCase.put("骼", "骼");
        normalizationBadCase.put("拾", "拾");
        normalizationBadCase.put("劈", "劈");
        normalizationBadCase.put("溜", "溜");
        normalizationBadCase.put("呱", "呱");
        normalizationBadCase.put("怵", "怵");
        normalizationBadCase.put("糸", "丝");
        normalizationBadCase.put("乾", "乾");
        normalizationBadCase.put("艸", "草");
        for (Map.Entry<String, String> entry : normalizationBadCase.entrySet())
        {
            String input = entry.getKey();
            String result = CharTable.convert(input);
            String expected = entry.getValue();
            System.out.println(input + "=" + expected);
            assert result.equals(expected);
        }
    }
//    public void testConvert() throws Exception
//    {
//        System.out.println(CharTable.CONVERT['關']);
//        System.out.println(CharTable.CONVERT['Ａ']);
//        System.out.println(CharTable.CONVERT['“']);
//        System.out.println(CharTable.CONVERT['．']);
//    }
//
//    public void testEnd() throws Exception
//    {
//        System.out.println(CharTable.CONVERT['，']);
//        System.out.println(CharTable.CONVERT['。']);
//        System.out.println(CharTable.CONVERT['！']);
//        System.out.println(CharTable.CONVERT['…']);
//    }
//
//    public void testFix() throws Exception
//    {
//        char[] CONVERT = CharTable.CONVERT;
//        CONVERT['.'] = '.';
//        CONVERT['．'] = '.';
//        CONVERT['。'] = '.';
//        CONVERT['！'] = '!';
//        CONVERT['，'] = ',';
//        CONVERT['!'] = '!';
//        CONVERT['#'] = '#';
//        CONVERT['&'] = '&';
//        CONVERT['*'] = '*';
//        CONVERT[','] = ',';
//        CONVERT['/'] = '/';
//        CONVERT[';'] = ';';
//        CONVERT['?'] = '?';
//        CONVERT['\\'] = '\\';
//        CONVERT['^'] = '^';
//        CONVERT['_'] = '_';
//        CONVERT['`'] = '`';
//        CONVERT['|'] = '|';
//        CONVERT['~'] = '~';
//        CONVERT['¡'] = '¡';
//        CONVERT['¦'] = '¦';
//        CONVERT['´'] = '´';
//        CONVERT['¸'] = '¸';
//        CONVERT['¿'] = '¿';
//        CONVERT['ˇ'] = 'ˇ';
//        CONVERT['ˉ'] = 'ˉ';
//        CONVERT['ˊ'] = 'ˊ';
//        CONVERT['ˋ'] = 'ˋ';
//        CONVERT['˜'] = '˜';
//        CONVERT['—'] = '—';
//        CONVERT['―'] = '―';
//        CONVERT['‖'] = '‖';
//        CONVERT['…'] = '…';
//        CONVERT['∕'] = '∕';
//        CONVERT['︳'] = '︳';
//        CONVERT['︴'] = '︴';
//        CONVERT['﹉'] = '﹉';
//        CONVERT['﹊'] = '﹊';
//        CONVERT['﹋'] = '﹋';
//        CONVERT['﹌'] = '﹌';
//        CONVERT['﹍'] = '﹍';
//        CONVERT['﹎'] = '﹎';
//        CONVERT['﹏'] = '﹏';
//        CONVERT['﹐'] = '﹐';
//        CONVERT['﹑'] = '﹑';
//        CONVERT['﹔'] = '﹔';
//        CONVERT['﹖'] = '﹖';
//        CONVERT['﹟'] = '﹟';
//        CONVERT['﹠'] = '﹠';
//        CONVERT['﹡'] = '﹡';
//        CONVERT['﹨'] = '﹨';
//        CONVERT['＇'] = '＇';
//        CONVERT['；'] = '；';
//        CONVERT['？'] = '？';
//        CONVERT['幣'] = '币';
//        CONVERT['繫'] = '系';
//        CONVERT['眾'] = '众';
//        CONVERT['龕'] = '龛';
//        CONVERT['製'] = '制';
//        for (int i = 0; i < CONVERT.length; i++)
//        {
//            if (CONVERT[i] == '\u0000')
//            {
//                if (i != '\u0000') CONVERT[i] = (char) i;
//                else CONVERT[i] = ' ';
//            }
//        }
//        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(HanLP.Config.CharTablePath));
//        out.writeObject(CONVERT);
//        out.close();
//    }
//
//    public void testImportSingleCharFromTraditionalChineseDictionary() throws Exception
//    {
////        char[] CONVERT = CharTable.CONVERT;
////        StringDictionary dictionary = new StringDictionary("=");
////        dictionary.load(HanLP.Config.t2sDictionaryPath);
////        for (Map.Entry<String, String> entry : dictionary.entrySet())
////        {
////            String key = entry.getKey();
////            if (key.length() != 1) continue;
////            String value = entry.getValue();
////            char t = key.charAt(0);
////            char s = value.charAt(0);
//////            if (CONVERT[t] != s)
//////            {
//////                System.out.printf("%s\t%c=%c\n", entry, t, CONVERT[t]);
//////            }
////            CONVERT[t] = s;
////        }
////
////        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(HanLP.Config.CharTablePath));
////        out.writeObject(CONVERT);
////        out.close();
//    }
//
//    public void testDumpCharTable() throws Exception
//    {
//        BufferedWriter bw = IOUtil.newBufferedWriter(HanLP.Config.CharTablePath.replace(".bin.yes", ".txt"));
//        char[] CONVERT = CharTable.CONVERT;
//        for (int i = 0; i < CONVERT.length; i++)
//        {
//            if (i != CONVERT[i])
//            {
//                bw.write(String.format("%c=%c\n", i, CONVERT[i]));
//            }
//        }
//        bw.close();
//    }
//
//    public void testLoadCharTableFromTxt() throws Exception
//    {
////        CharTable.load(HanLP.Config.CharTablePath.replace(".bin.yes", ".txt"));
//    }
}
