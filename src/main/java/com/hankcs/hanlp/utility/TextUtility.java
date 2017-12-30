package com.hankcs.hanlp.utility;


import java.io.*;
import java.util.Collection;

/**
 * 文本工具类
 */
public class TextUtility
{

    /**
     * 单字节
     */
    public static final int CT_SINGLE = 5;// SINGLE byte

    /**
     * 分隔符"!,.?()[]{}+=
     */
    public static final int CT_DELIMITER = CT_SINGLE + 1;// delimiter

    /**
     * 中文字符
     */
    public static final int CT_CHINESE = CT_SINGLE + 2;// Chinese Char

    /**
     * 字母
     */
    public static final int CT_LETTER = CT_SINGLE + 3;// HanYu Pinyin

    /**
     * 数字
     */
    public static final int CT_NUM = CT_SINGLE + 4;// HanYu Pinyin

    /**
     * 序号
     */
    public static final int CT_INDEX = CT_SINGLE + 5;// HanYu Pinyin

    /**
     * 中文数字
     */
    public static final int CT_CNUM = CT_SINGLE + 6;

    /**
     * 其他
     */
    public static final int CT_OTHER = CT_SINGLE + 12;// Other

    public static int charType(char c)
    {
        return charType(String.valueOf(c));
    }

    /**
     * 判断字符类型
     * @param str
     * @return
     */
    public static int charType(String str)
    {
        if (str != null && str.length() > 0)
        {
            if ("零○〇一二两三四五六七八九十廿百千万亿壹贰叁肆伍陆柒捌玖拾佰仟".contains(str)) return CT_CNUM;
            byte[] b;
            try
            {
                b = str.getBytes("GBK");
            }
            catch (UnsupportedEncodingException e)
            {
                b = str.getBytes();
                e.printStackTrace();
            }
            byte b1 = b[0];
            byte b2 = b.length > 1 ? b[1] : 0;
            int ub1 = getUnsigned(b1);
            int ub2 = getUnsigned(b2);
            if (ub1 < 128)
            {
                if (ub1 < 32) return CT_DELIMITER; // NON PRINTABLE CHARACTERS
                if (' ' == b1) return CT_OTHER;
                if ('\n' == b1) return CT_DELIMITER;
                if ("*\"!,.?()[]{}+=/\\;:|".indexOf((char) b1) != -1)
                    return CT_DELIMITER;
                if ("0123456789".indexOf((char)b1) != -1)
                    return CT_NUM;
                return CT_SINGLE;
            }
            else if (ub1 == 162)
                return CT_INDEX;
            else if (ub1 == 163 && ub2 > 175 && ub2 < 186)
                return CT_NUM;
            else if (ub1 == 163
                    && (ub2 >= 193 && ub2 <= 218 || ub2 >= 225
                    && ub2 <= 250))
                return CT_LETTER;
            else if (ub1 == 161 || ub1 == 163)
                return CT_DELIMITER;
            else if (ub1 >= 176 && ub1 <= 247)
                return CT_CHINESE;

        }
        return CT_OTHER;
    }

    /**
     * 是否全是中文
     * @param str
     * @return
     */
    public static boolean isAllChinese(String str)
    {
        return str.matches("[\\u4E00-\\u9FA5]+");
    }
    /**
     * 是否全部不是中文
     * @param sString
     * @return
     */
    public static boolean isAllNonChinese(byte[] sString)
    {
        int nLen = sString.length;
        int i = 0;

        while (i < nLen)
        {
            if (getUnsigned(sString[i]) < 248 && getUnsigned(sString[i]) > 175)
                return false;
            if (sString[i] < 0)
                i += 2;
            else
                i += 1;
        }
        return true;
    }

    /**
     * 是否全是单字节
     * @param str
     * @return
     */
    public static boolean isAllSingleByte(String str)
    {
        assert str != null;
        for (int i = 0; i < str.length(); i++)
        {
            if (str.charAt(i) >128)
            {
                return false;
            }
        }
        return true;
    }

    /**
     * 把表示数字含义的字符串转成整形
     *
     * @param str 要转换的字符串
     * @return 如果是有意义的整数，则返回此整数值。否则，返回-1。
     */
    public static int cint(String str)
    {
        if (str != null)
            try
            {
                int i = new Integer(str).intValue();
                return i;
            }
            catch (NumberFormatException e)
            {

            }

        return -1;
    }
    /**
     * 是否全是数字
     * @param str
     * @return
     */
    public static boolean isAllNum(String str)
    {
        if (str == null)
            return false;

        int i = 0;
        /** 判断开头是否是+-之类的符号 */
        if ("±+-＋－—".indexOf(str.charAt(0)) != -1)
            i++;
        /** 如果是全角的０１２３４５６７８９ 字符* */
        while (i < str.length() && "０１２３４５６７８９".indexOf(str.charAt(i)) != -1)
            i++;
        // Get middle delimiter such as .
        if (i > 0 && i < str.length())
        {
            char ch = str.charAt(i);
            if ("·∶:，,．.／/".indexOf(ch) != -1)
            {// 98．1％
                i++;
                while (i < str.length() && "０１２３４５６７８９".indexOf(str.charAt(i)) != -1)
                    i++;
            }
        }
        if (i >= str.length())
            return true;

        /** 如果是半角的0123456789字符* */
        while (i < str.length() && "0123456789".indexOf(str.charAt(i)) != -1)
            i++;
        // Get middle delimiter such as .
        if (i > 0 && i < str.length())
        {
            char ch = str.charAt(i);
            if (',' == ch || '.' == ch || '/' == ch  || ':' == ch || "∶·，．／".indexOf(ch) != -1)
            {// 98．1％
                i++;
                while (i < str.length() && "0123456789".indexOf(str.charAt(i)) != -1)
                    i++;
            }
        }

        if (i < str.length())
        {
            if ("百千万亿佰仟%％‰".indexOf(str.charAt(i)) != -1)
                i++;
        }
        if (i >= str.length())
            return true;

        return false;
    }

    /**
     * 是否全是序号
     * @param sString
     * @return
     */
    public static boolean isAllIndex(byte[] sString)
    {
        int nLen = sString.length;
        int i = 0;

        while (i < nLen - 1 && getUnsigned(sString[i]) == 162)
        {
            i += 2;
        }
        if (i >= nLen)
            return true;
        while (i < nLen && (sString[i] > 'A' - 1 && sString[i] < 'Z' + 1)
                || (sString[i] > 'a' - 1 && sString[i] < 'z' + 1))
        {// single
            // byte
            // number
            // char
            i += 1;
        }

        if (i < nLen)
            return false;
        return true;

    }

    /**
     * 是否全为英文
     *
     * @param text
     * @return
     */
    public static boolean isAllLetter(String text)
    {
        for (int i = 0; i < text.length(); ++i)
        {
            char c = text.charAt(i);
            if ((((c < 'a' || c > 'z')) && ((c < 'A' || c > 'Z'))))
            {
                return false;
            }
        }

        return true;
    }

    /**
     * 是否全为英文或字母
     *
     * @param text
     * @return
     */
    public static boolean isAllLetterOrNum(String text)
    {
        for (int i = 0; i < text.length(); ++i)
        {
            char c = text.charAt(i);
            if ((((c < 'a' || c > 'z')) && ((c < 'A' || c > 'Z')) && ((c < '0' || c > '9'))))
            {
                return false;
            }
        }

        return true;
    }

    /**
     * 是否全是分隔符
     * @param sString
     * @return
     */
    public static boolean isAllDelimiter(byte[] sString)
    {
        int nLen = sString.length;
        int i = 0;

        while (i < nLen - 1 && (getUnsigned(sString[i]) == 161 || getUnsigned(sString[i]) == 163))
        {
            i += 2;
        }
        if (i < nLen)
            return false;
        return true;
    }

    /**
     * 是否全是中国数字
     * @param word
     * @return
     */
    public static boolean isAllChineseNum(String word)
    {// 百分之五点六的人早上八点十八分起床

        String chineseNum = "零○一二两三四五六七八九十廿百千万亿壹贰叁肆伍陆柒捌玖拾佰仟∶·．／点";//
        String prefix = "几数上第";
        String surfix = "几多余来成倍";
        boolean round = false;

        if (word == null)
            return false;

        char[] temp = word.toCharArray();
        for (int i = 0; i < temp.length; i++)
        {
            if (word.startsWith("分之", i))// 百分之五
            {
                i += 1;
                continue;
            }
            char tchar = temp[i];
            if (i == 0 && prefix.indexOf(tchar) != -1)
            {
                round = true;
            }
            else if (i == temp.length-1 && !round && surfix.indexOf(tchar) != -1)
            {
                round = true;
            }
            else if (chineseNum.indexOf(tchar) == -1)
                return false;
        }
        return true;
    }


    /**
     * 得到字符集的字符在字符串中出现的次数
     *
     * @param charSet
     * @param word
     * @return
     */
    public static int getCharCount(String charSet, String word)
    {
        int nCount = 0;

        if (word != null)
        {
            String temp = word + " ";
            for (int i = 0; i < word.length(); i++)
            {
                String s = temp.substring(i, i + 1);
                if (charSet.indexOf(s) != -1)
                    nCount++;
            }
        }

        return nCount;
    }


    /**
     * 获取字节对应的无符号整型数
     *
     * @param b
     * @return
     */
    public static int getUnsigned(byte b)
    {
        if (b > 0)
            return (int) b;
        else
            return (b & 0x7F + 128);
    }

    /**
     * 判断字符串是否是年份
     *
     * @param snum
     * @return
     */
    public static boolean isYearTime(String snum)
    {
        if (snum != null)
        {
            int len = snum.length();
            String first = snum.substring(0, 1);

            // 1992年, 98年,06年
            if (isAllSingleByte(snum)
                    && (len == 4 || len == 2 && (cint(first) > 4 || cint(first) == 0)))
                return true;
            if (isAllNum(snum) && (len >= 3 || len == 2 && "０５６７８９".indexOf(first) != -1))
                return true;
            if (getCharCount("零○一二三四五六七八九壹贰叁肆伍陆柒捌玖", snum) == len && len >= 2)
                return true;
            if (len == 4 && getCharCount("千仟零○", snum) == 2)// 二仟零二年
                return true;
            if (len == 1 && getCharCount("千仟", snum) == 1)
                return true;
            if (len == 2 && getCharCount("甲乙丙丁戊己庚辛壬癸", snum) == 1
                    && getCharCount("子丑寅卯辰巳午未申酉戌亥", snum.substring(1)) == 1)
                return true;
        }
        return false;
    }

    /**
     * 判断一个字符串的所有字符是否在另一个字符串集合中
     *
     * @param aggr 字符串集合
     * @param str  需要判断的字符串
     * @return
     */
    public static boolean isInAggregate(String aggr, String str)
    {
        if (aggr != null && str != null)
        {
            str += "1";
            for (int i = 0; i < str.length(); i++)
            {
                String s = str.substring(i, i + 1);
                if (aggr.indexOf(s) == -1)
                    return false;
            }
            return true;
        }

        return false;
    }

    /**
     * 判断该字符串是否是半角字符
     *
     * @param str
     * @return
     */
    public static boolean isDBCCase(String str)
    {
        if (str != null)
        {
            str += " ";
            for (int i = 0; i < str.length(); i++)
            {
                String s = str.substring(i, i + 1);
                int length = 0;
                try
                {
                    length = s.getBytes("GBK").length;
                }
                catch (UnsupportedEncodingException e)
                {
                    e.printStackTrace();
                    length = s.getBytes().length;
                }
                if (length != 1)
                    return false;
            }

            return true;
        }

        return false;
    }

    /**
     * 判断该字符串是否是全角字符
     *
     * @param str
     * @return
     */
    public static boolean isSBCCase(String str)
    {
        if (str != null)
        {
            str += " ";
            for (int i = 0; i < str.length(); i++)
            {
                String s = str.substring(i, i + 1);
                int length = 0;
                try
                {
                    length = s.getBytes("GBK").length;
                }
                catch (UnsupportedEncodingException e)
                {
                    e.printStackTrace();
                    length = s.getBytes().length;
                }
                if (length != 2)
                    return false;
            }

            return true;
        }

        return false;
    }

    /**
     * 判断是否是一个连字符（分隔符）
     *
     * @param str
     * @return
     */
    public static boolean isDelimiter(String str)
    {
        if (str != null && ("-".equals(str) || "－".equals(str)))
            return true;
        else
            return false;
    }

    public static boolean isUnknownWord(String word)
    {
        if (word != null && word.indexOf("未##") == 0)
            return true;
        else
            return false;
    }

    /**
     * 防止频率为0发生除零错误
     *
     * @param frequency
     * @return
     */
    public static double nonZero(double frequency)
    {
        if (frequency == 0) return 1e-3;

        return frequency;
    }

    /**
     * 转换long型为char数组
     *
     * @param x
     */
    public static char[] long2char(long x)
    {
        char[] c = new char[4];
        c[0] = (char) (x >> 48);
        c[1] = (char) (x >> 32);
        c[2] = (char) (x >> 16);
        c[3] = (char) (x);
        return c;
    }

    /**
     * 转换long类型为string
     *
     * @param x
     * @return
     */
    public static String long2String(long x)
    {
        char[] cArray = long2char(x);
        StringBuilder sbResult = new StringBuilder(cArray.length);
        for (char c : cArray)
        {
            sbResult.append(c);
        }
        return sbResult.toString();
    }

    /**
     * 将异常转为字符串
     *
     * @param e
     * @return
     */
    public static String exceptionToString(Exception e)
    {
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        e.printStackTrace(pw);
        return sw.toString();
    }

    /**
     * 判断某个字符是否为汉字
     *
     * @param c 需要判断的字符
     * @return 是汉字返回true，否则返回false
     */
    public static boolean isChinese(char c)
    {
        String regex = "[\\u4e00-\\u9fa5]";
        return String.valueOf(c).matches(regex);
    }

    /**
     * 统计 keyword 在 srcText 中的出现次数
     *
     * @param keyword
     * @param srcText
     * @return
     */
    public static int count(String keyword, String srcText)
    {
        int count = 0;
        int leng = srcText.length();
        int j = 0;
        for (int i = 0; i < leng; i++)
        {
            if (srcText.charAt(i) == keyword.charAt(j))
            {
                j++;
                if (j == keyword.length())
                {
                    count++;
                    j = 0;
                }
            }
            else
            {
                i = i - j;// should rollback when not match
                j = 0;
            }
        }

        return count;
    }

    /**
     * 简单好用的写String方式
     *
     * @param s
     * @param out
     * @throws IOException
     */
    public static void writeString(String s, DataOutputStream out) throws IOException
    {
        out.writeInt(s.length());
        for (char c : s.toCharArray())
        {
            out.writeChar(c);
        }
    }

    /**
     * 判断字符串是否为空（null和空格）
     *
     * @param cs
     * @return
     */
    public static boolean isBlank(CharSequence cs)
    {
        int strLen;
        if (cs == null || (strLen = cs.length()) == 0)
        {
            return true;
        }
        for (int i = 0; i < strLen; i++)
        {
            if (!Character.isWhitespace(cs.charAt(i)))
            {
                return false;
            }
        }
        return true;
    }

    public static String join(String delimiter, Collection<String> stringCollection)
    {
        StringBuilder sb = new StringBuilder(stringCollection.size() * (16 + delimiter.length()));
        for (String str : stringCollection)
        {
            sb.append(str).append(delimiter);
        }

        return sb.toString();
    }
}
