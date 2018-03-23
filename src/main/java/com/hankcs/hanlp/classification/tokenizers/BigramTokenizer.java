package com.hankcs.hanlp.classification.tokenizers;

import com.hankcs.hanlp.dictionary.other.CharTable;
import com.hankcs.hanlp.dictionary.other.CharType;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class BigramTokenizer implements ITokenizer
{
    public String[] segment(String text)
    {
        if (text.length() == 0) return new String[0];
        char[] charArray = text.toCharArray();
        CharTable.normalization(charArray);

        // 先拆成字
        List<int[]> atomList = new LinkedList<int[]>();
        int start = 0;
        int end = charArray.length;
        int offsetAtom = start;
        byte preType = CharType.get(charArray[offsetAtom]);
        byte curType;
        while (++offsetAtom < end)
        {
            curType = CharType.get(charArray[offsetAtom]);
            if (preType == CharType.CT_CHINESE)
            {
                atomList.add(new int[]{start, offsetAtom - start});
                start = offsetAtom;
            }
            else if (curType != preType)
            {
                // 浮点数识别
                if (charArray[offsetAtom] == '.' && preType == CharType.CT_NUM)
                {
                    while (++offsetAtom < end)
                    {
                        curType = CharType.get(charArray[offsetAtom]);
                        if (curType != CharType.CT_NUM) break;
                    }
                }
                if (preType == CharType.CT_NUM || preType == CharType.CT_LETTER) atomList.add(new int[]{start, offsetAtom - start});
                start = offsetAtom;
            }
            preType = curType;
        }
        if (offsetAtom == end)
            if (preType == CharType.CT_NUM || preType == CharType.CT_LETTER) atomList.add(new int[]{start, offsetAtom - start});
        if (atomList.isEmpty()) return new String[0];
        // 输出
        String[] termArray = new String[atomList.size() - 1];
        Iterator<int[]> iterator = atomList.iterator();
        int[] pre = iterator.next();
        int p = -1;
        while (iterator.hasNext())
        {
            int[] cur = iterator.next();
            termArray[++p] = new StringBuilder(pre[1] + cur[1]).append(charArray, pre[0], pre[1]).append(charArray, cur[0], cur[1]).toString();
            pre = cur;
        }

        return termArray;
    }

//    public static void main(String args[])
//    {
//        BigramTokenizer bws = new BigramTokenizer();
//        String[] result = bws.segment("@hankcs你好，广阔的世界2０16！\u0000\u0000\t\n\r\n慶祝Coding worlds!");
//        for (String str : result)
//        {
//            System.out.println(str);
//        }
//    }

}
