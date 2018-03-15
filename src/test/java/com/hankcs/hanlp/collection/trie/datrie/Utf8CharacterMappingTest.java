package com.hankcs.hanlp.collection.trie.datrie;

import junit.framework.TestCase;

public class Utf8CharacterMappingTest extends TestCase
{
    public void testToIdList() throws Exception
    {
        Utf8CharacterMapping ucm = new Utf8CharacterMapping();
        String s = "汉字\uD801\uDC00\uD801\uDC00ab\uD801\uDC00\uD801\uDC00cd";
        int[] bytes1 = ucm.toIdList(s);
        System.out.println("UTF-8: " + bytes1.length);
        {
            int charCount = 1;
            int start = 0;
            for (int i = 0; i < s.length(); i += charCount)
            {
                int codePoint = s.codePointAt(i);
                charCount = Character.charCount(codePoint);

                int[] arr = ucm.toIdList(codePoint);
                for (int j = 0; j < arr.length; j++, start++)
                {
                    if (bytes1[start] != arr[j])
                    {
                        System.out.println("error: " + start + "," + j);
                        System.exit(-1);
                    }
                }
            }
            if (start != bytes1.length)
            {
                System.out.println("error: " + start + "," + bytes1.length);
            }
        }
    }
}