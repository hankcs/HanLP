package com.hankcs.hanlp.utility;

import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.seg.common.Term;

import java.util.LinkedList;
import java.util.List;

/**
 * 文本断句
 *
 */
public class SentencesUtil
{
    /**
     * 将文本切割为句子
     * @param content
     * @return
     */
    public static List<String> toSentenceList(String content)
    {
        return toSentenceList(content.toCharArray());
    }

    public static List<String> toSentenceList(char[] chars)
    {

        StringBuilder sb = new StringBuilder();

        List<String> sentences = new LinkedList<String>();

        for (int i = 0; i < chars.length; ++i)
        {
            if (sb.length() == 0 && (Character.isWhitespace(chars[i]) || chars[i] == ' '))
            {
                continue;
            }

            sb.append(chars[i]);
            switch (chars[i])
            {
                case '.':
                    if (i < chars.length - 1 && chars[i + 1] > 128)
                    {
                        insertIntoList(sb, sentences);
                        sb = new StringBuilder();
                    }
                    break;
                case '…':
                {
                    if (i < chars.length - 1 && chars[i + 1] == '…')
                    {
                        sb.append('…');
                        ++i;
                        insertIntoList(sb, sentences);
                        sb = new StringBuilder();
                    }
                }break;
                case ' ':
                case '	':
                case ' ':
                case '。':
                case '，':
                case ',':
                    insertIntoList(sb, sentences);
                    sb = new StringBuilder();
                    break;
                case ';':
                case '；':
                    insertIntoList(sb, sentences);
                    sb = new StringBuilder();
                    break;
                case '!':
                case '！':
                    insertIntoList(sb, sentences);
                    sb = new StringBuilder();
                    break;
                case '?':
                case '？':
                    insertIntoList(sb, sentences);
                    sb = new StringBuilder();
                    break;
                case '\n':
                case '\r':
                    insertIntoList(sb, sentences);
                    sb = new StringBuilder();
                    break;
            }
        }

        if (sb.length() > 0)
        {
            insertIntoList(sb, sentences);
        }

        return sentences;
    }

    private static void insertIntoList(StringBuilder sb, List<String> sentences)
    {
        String content = sb.toString().trim();
        if (content.length() > 0)
        {
            sentences.add(content);
        }
    }

    /**
     * 句子中是否含有词性
     * @param sentence
     * @param nature
     * @return
     */
    public static boolean hasNature(List<Term> sentence, Nature nature)
    {
        for (Term term : sentence)
        {
            if (term.nature == nature)
            {
                return true;
            }
        }

        return false;
    }
}
