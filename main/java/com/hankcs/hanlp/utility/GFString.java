/*
 * Created on 2004-5-31
 *
 * To change the template for this generated file go to
 * Window&gt;Preferences&gt;Java&gt;Code Generation&gt;Code and Comments
 */
package com.hankcs.hanlp.utility;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;


/**
 * 和字符串相关的常用操作
 *
 * @author SINBOY
 * @version V1.1
 */
public class GFString
{
    private static final HashMap<String, String> map_hex2bin = new HashMap<String, String>(16);

    {
        map_hex2bin.put("0", "0000");
        map_hex2bin.put("1", "0001");
        map_hex2bin.put("2", "0010");
        map_hex2bin.put("3", "0011");
        map_hex2bin.put("4", "0100");
        map_hex2bin.put("5", "0101");
        map_hex2bin.put("6", "0110");
        map_hex2bin.put("7", "0111");
        map_hex2bin.put("8", "1000");
        map_hex2bin.put("9", "1001");
        map_hex2bin.put("A", "1010");
        map_hex2bin.put("B", "1011");
        map_hex2bin.put("C", "1100");
        map_hex2bin.put("D", "1101");
        map_hex2bin.put("E", "1110");
        map_hex2bin.put("F", "1111");
    }

    public final static HashMap<String, String> map_bin2hex = new HashMap<String, String>(16);

    {
        map_bin2hex.put("0000", "0");
        map_bin2hex.put("0001", "1");
        map_bin2hex.put("0010", "2");
        map_bin2hex.put("0011", "3");
        map_bin2hex.put("0100", "4");
        map_bin2hex.put("0101", "5");
        map_bin2hex.put("0110", "6");
        map_bin2hex.put("0111", "7");
        map_bin2hex.put("1000", "8");
        map_bin2hex.put("1001", "9");
        map_bin2hex.put("1010", "A");
        map_bin2hex.put("1011", "B");
        map_bin2hex.put("1100", "C");
        map_bin2hex.put("1101", "D");
        map_bin2hex.put("1110", "E");
        map_bin2hex.put("1111", "F");
    }

    private final static LinkedHashMap<String, Integer> bopoMap = new LinkedHashMap<String, Integer>();

    {
        bopoMap.put("a", 1);
        bopoMap.put("a", -20319);
        bopoMap.put("ai", -20317);
        bopoMap.put("an", -20304);
        bopoMap.put("ang", -20295);
        bopoMap.put("ao", -20292);
        bopoMap.put("ba", -20283);
        bopoMap.put("bai", -20265);
        bopoMap.put("ban", -20257);
        bopoMap.put("bang", -20242);
        bopoMap.put("bao", -20230);
        bopoMap.put("bei", -20051);
        bopoMap.put("ben", -20036);
        bopoMap.put("beng", -20032);
        bopoMap.put("bi", -20026);
        bopoMap.put("bian", -20002);
        bopoMap.put("biao", -19990);
        bopoMap.put("bie", -19986);
        bopoMap.put("bin", -19982);
        bopoMap.put("bing", -19976);
        bopoMap.put("bo", -19805);
        bopoMap.put("bu", -19784);
        bopoMap.put("ca", -19775);
        bopoMap.put("cai", -19774);
        bopoMap.put("can", -19763);
        bopoMap.put("cang", -19756);
        bopoMap.put("cao", -19751);
        bopoMap.put("ce", -19746);
        bopoMap.put("ceng", -19741);
        bopoMap.put("cha", -19739);
        bopoMap.put("chai", -19728);
        bopoMap.put("chan", -19725);
        bopoMap.put("chang", -19715);
        bopoMap.put("chao", -19540);
        bopoMap.put("che", -19531);
        bopoMap.put("chen", -19525);
        bopoMap.put("cheng", -19515);
        bopoMap.put("chi", -19500);
        bopoMap.put("chong", -19484);
        bopoMap.put("chou", -19479);
        bopoMap.put("chu", -19467);
        bopoMap.put("chuai", -19289);
        bopoMap.put("chuan", -19288);
        bopoMap.put("chuang", -19281);
        bopoMap.put("chui", -19275);
        bopoMap.put("chun", -19270);
        bopoMap.put("chuo", -19263);
        bopoMap.put("ci", -19261);
        bopoMap.put("cong", -19249);
        bopoMap.put("cou", -19243);
        bopoMap.put("cu", -19242);
        bopoMap.put("cuan", -19238);
        bopoMap.put("cui", -19235);
        bopoMap.put("cun", -19227);
        bopoMap.put("cuo", -19224);
        bopoMap.put("da", -19218);
        bopoMap.put("dai", -19212);
        bopoMap.put("dan", -19038);
        bopoMap.put("dang", -19023);
        bopoMap.put("dao", -19018);
        bopoMap.put("de", -19006);
        bopoMap.put("deng", -19003);
        bopoMap.put("di", -18996);
        bopoMap.put("dian", -18977);
        bopoMap.put("diao", -18961);
        bopoMap.put("die", -18952);
        bopoMap.put("ding", -18783);
        bopoMap.put("diu", -18774);
        bopoMap.put("dong", -18773);
        bopoMap.put("dou", -18763);
        bopoMap.put("du", -18756);
        bopoMap.put("duan", -18741);
        bopoMap.put("dui", -18735);
        bopoMap.put("dun", -18731);
        bopoMap.put("duo", -18722);
        bopoMap.put("e", -18710);
        bopoMap.put("en", -18697);
        bopoMap.put("er", -18696);
        bopoMap.put("fa", -18526);
        bopoMap.put("fan", -18518);
        bopoMap.put("fang", -18501);
        bopoMap.put("fei", -18490);
        bopoMap.put("fen", -18478);
        bopoMap.put("feng", -18463);
        bopoMap.put("fo", -18448);
        bopoMap.put("fou", -18447);
        bopoMap.put("fu", -18446);
        bopoMap.put("ga", -18239);
        bopoMap.put("gai", -18237);
        bopoMap.put("gan", -18231);
        bopoMap.put("gang", -18220);
        bopoMap.put("gao", -18211);
        bopoMap.put("ge", -18201);
        bopoMap.put("gei", -18184);
        bopoMap.put("gen", -18183);
        bopoMap.put("geng", -18181);
        bopoMap.put("gong", -18012);
        bopoMap.put("gou", -17997);
        bopoMap.put("gu", -17988);
        bopoMap.put("gua", -17970);
        bopoMap.put("guai", -17964);
        bopoMap.put("guan", -17961);
        bopoMap.put("guang", -17950);
        bopoMap.put("gui", -17947);
        bopoMap.put("gun", -17931);
        bopoMap.put("guo", -17928);
        bopoMap.put("ha", -17922);
        bopoMap.put("hai", -17759);
        bopoMap.put("han", -17752);
        bopoMap.put("hang", -17733);
        bopoMap.put("hao", -17730);
        bopoMap.put("he", -17721);
        bopoMap.put("hei", -17703);
        bopoMap.put("hen", -17701);
        bopoMap.put("heng", -17697);
        bopoMap.put("hong", -17692);
        bopoMap.put("hou", -17683);
        bopoMap.put("hu", -17676);
        bopoMap.put("hua", -17496);
        bopoMap.put("huai", -17487);
        bopoMap.put("huan", -17482);
        bopoMap.put("huang", -17468);
        bopoMap.put("hui", -17454);
        bopoMap.put("hun", -17433);
        bopoMap.put("huo", -17427);
        bopoMap.put("ji", -17417);
        bopoMap.put("jia", -17202);
        bopoMap.put("jian", -17185);
        bopoMap.put("jiang", -16983);
        bopoMap.put("jiao", -16970);
        bopoMap.put("jie", -16942);
        bopoMap.put("jin", -16915);
        bopoMap.put("jing", -16733);
        bopoMap.put("jiong", -16708);
        bopoMap.put("jiu", -16706);
        bopoMap.put("ju", -16689);
        bopoMap.put("juan", -16664);
        bopoMap.put("jue", -16657);
        bopoMap.put("jun", -16647);
        bopoMap.put("ka", -16474);
        bopoMap.put("kai", -16470);
        bopoMap.put("kan", -16465);
        bopoMap.put("kang", -16459);
        bopoMap.put("kao", -16452);
        bopoMap.put("ke", -16448);
        bopoMap.put("ken", -16433);
        bopoMap.put("keng", -16429);
        bopoMap.put("kong", -16427);
        bopoMap.put("kou", -16423);
        bopoMap.put("ku", -16419);
        bopoMap.put("kua", -16412);
        bopoMap.put("kuai", -16407);
        bopoMap.put("kuan", -16403);
        bopoMap.put("kuang", -16401);
        bopoMap.put("kui", -16393);
        bopoMap.put("kun", -16220);
        bopoMap.put("kuo", -16216);
        bopoMap.put("la", -16212);
        bopoMap.put("lai", -16205);
        bopoMap.put("lan", -16202);
        bopoMap.put("lang", -16187);
        bopoMap.put("lao", -16180);
        bopoMap.put("le", -16171);
        bopoMap.put("lei", -16169);
        bopoMap.put("leng", -16158);
        bopoMap.put("li", -16155);
        bopoMap.put("lia", -15959);
        bopoMap.put("lian", -15958);
        bopoMap.put("liang", -15944);
        bopoMap.put("liao", -15933);
        bopoMap.put("lie", -15920);
        bopoMap.put("lin", -15915);
        bopoMap.put("ling", -15903);
        bopoMap.put("liu", -15889);
        bopoMap.put("long", -15878);
        bopoMap.put("lou", -15707);
        bopoMap.put("lu", -15701);
        bopoMap.put("lv", -15681);
        bopoMap.put("luan", -15667);
        bopoMap.put("lue", -15661);
        bopoMap.put("lun", -15659);
        bopoMap.put("luo", -15652);
        bopoMap.put("ma", -15640);
        bopoMap.put("mai", -15631);
        bopoMap.put("man", -15625);
        bopoMap.put("mang", -15454);
        bopoMap.put("mao", -15448);
        bopoMap.put("me", -15436);
        bopoMap.put("mei", -15435);
        bopoMap.put("men", -15419);
        bopoMap.put("meng", -15416);
        bopoMap.put("mi", -15408);
        bopoMap.put("mian", -15394);
        bopoMap.put("miao", -15385);
        bopoMap.put("mie", -15377);
        bopoMap.put("min", -15375);
        bopoMap.put("ming", -15369);
        bopoMap.put("miu", -15363);
        bopoMap.put("mo", -15362);
        bopoMap.put("mou", -15183);
        bopoMap.put("mu", -15180);
        bopoMap.put("na", -15165);
        bopoMap.put("nai", -15158);
        bopoMap.put("nan", -15153);
        bopoMap.put("nang", -15150);
        bopoMap.put("nao", -15149);
        bopoMap.put("ne", -15144);
        bopoMap.put("nei", -15143);
        bopoMap.put("nen", -15141);
        bopoMap.put("neng", -15140);
        bopoMap.put("ni", -15139);
        bopoMap.put("nian", -15128);
        bopoMap.put("niang", -15121);
        bopoMap.put("niao", -15119);
        bopoMap.put("nie", -15117);
        bopoMap.put("nin", -15110);
        bopoMap.put("ning", -15109);
        bopoMap.put("niu", -14941);
        bopoMap.put("nong", -14937);
        bopoMap.put("nu", -14933);
        bopoMap.put("nv", -14930);
        bopoMap.put("nuan", -14929);
        bopoMap.put("nue", -14928);
        bopoMap.put("nuo", -14926);
        bopoMap.put("o", -14922);
        bopoMap.put("ou", -14921);
        bopoMap.put("pa", -14914);
        bopoMap.put("pai", -14908);
        bopoMap.put("pan", -14902);
        bopoMap.put("pang", -14894);
        bopoMap.put("pao", -14889);
        bopoMap.put("pei", -14882);
        bopoMap.put("pen", -14873);
        bopoMap.put("peng", -14871);
        bopoMap.put("pi", -14857);
        bopoMap.put("pian", -14678);
        bopoMap.put("piao", -14674);
        bopoMap.put("pie", -14670);
        bopoMap.put("pin", -14668);
        bopoMap.put("ping", -14663);
        bopoMap.put("po", -14654);
        bopoMap.put("pu", -14645);
        bopoMap.put("qi", -14630);
        bopoMap.put("qia", -14594);
        bopoMap.put("qian", -14429);
        bopoMap.put("qiang", -14407);
        bopoMap.put("qiao", -14399);
        bopoMap.put("qie", -14384);
        bopoMap.put("qin", -14379);
        bopoMap.put("qing", -14368);
        bopoMap.put("qiong", -14355);
        bopoMap.put("qiu", -14353);
        bopoMap.put("qu", -14345);
        bopoMap.put("quan", -14170);
        bopoMap.put("que", -14159);
        bopoMap.put("qun", -14151);
        bopoMap.put("ran", -14149);
        bopoMap.put("rang", -14145);
        bopoMap.put("rao", -14140);
        bopoMap.put("re", -14137);
        bopoMap.put("ren", -14135);
        bopoMap.put("reng", -14125);
        bopoMap.put("ri", -14123);
        bopoMap.put("rong", -14122);
        bopoMap.put("rou", -14112);
        bopoMap.put("ru", -14109);
        bopoMap.put("ruan", -14099);
        bopoMap.put("rui", -14097);
        bopoMap.put("run", -14094);
        bopoMap.put("ruo", -14092);
        bopoMap.put("sa", -14090);
        bopoMap.put("sai", -14087);
        bopoMap.put("san", -14083);
        bopoMap.put("sang", -13917);
        bopoMap.put("sao", -13914);
        bopoMap.put("se", -13910);
        bopoMap.put("sen", -13907);
        bopoMap.put("seng", -13906);
        bopoMap.put("sha", -13905);
        bopoMap.put("shai", -13896);
        bopoMap.put("shan", -13894);
        bopoMap.put("shang", -13878);
        bopoMap.put("shao", -13870);
        bopoMap.put("she", -13859);
        bopoMap.put("shen", -13847);
        bopoMap.put("sheng", -13831);
        bopoMap.put("shi", -13658);
        bopoMap.put("shou", -13611);
        bopoMap.put("shu", -13601);
        bopoMap.put("shua", -13406);
        bopoMap.put("shuai", -13404);
        bopoMap.put("shuan", -13400);
        bopoMap.put("shuang", -13398);
        bopoMap.put("shui", -13395);
        bopoMap.put("shun", -13391);
        bopoMap.put("shuo", -13387);
        bopoMap.put("si", -13383);
        bopoMap.put("song", -13367);
        bopoMap.put("sou", -13359);
        bopoMap.put("su", -13356);
        bopoMap.put("suan", -13343);
        bopoMap.put("sui", -13340);
        bopoMap.put("sun", -13329);
        bopoMap.put("suo", -13326);
        bopoMap.put("ta", -13318);
        bopoMap.put("tai", -13147);
        bopoMap.put("tan", -13138);
        bopoMap.put("tang", -13120);
        bopoMap.put("tao", -13107);
        bopoMap.put("te", -13096);
        bopoMap.put("teng", -13095);
        bopoMap.put("ti", -13091);
        bopoMap.put("tian", -13076);
        bopoMap.put("tiao", -13068);
        bopoMap.put("tie", -13063);
        bopoMap.put("ting", -13060);
        bopoMap.put("tong", -12888);
        bopoMap.put("tou", -12875);
        bopoMap.put("tu", -12871);
        bopoMap.put("tuan", -12860);
        bopoMap.put("tui", -12858);
        bopoMap.put("tun", -12852);
        bopoMap.put("tuo", -12849);
        bopoMap.put("wa", -12838);
        bopoMap.put("wai", -12831);
        bopoMap.put("wan", -12829);
        bopoMap.put("wang", -12812);
        bopoMap.put("wei", -12802);
        bopoMap.put("wen", -12607);
        bopoMap.put("weng", -12597);
        bopoMap.put("wo", -12594);
        bopoMap.put("wu", -12585);
        bopoMap.put("xi", -12556);
        bopoMap.put("xia", -12359);
        bopoMap.put("xian", -12346);
        bopoMap.put("xiang", -12320);
        bopoMap.put("xiao", -12300);
        bopoMap.put("xie", -12120);
        bopoMap.put("xin", -12099);
        bopoMap.put("xing", -12089);
        bopoMap.put("xiong", -12074);
        bopoMap.put("xiu", -12067);
        bopoMap.put("xu", -12058);
        bopoMap.put("xuan", -12039);
        bopoMap.put("xue", -11867);
        bopoMap.put("xun", -11861);
        bopoMap.put("ya", -11847);
        bopoMap.put("yan", -11831);
        bopoMap.put("yang", -11798);
        bopoMap.put("yao", -11781);
        bopoMap.put("ye", -11604);
        bopoMap.put("yi", -11589);
        bopoMap.put("yin", -11536);
        bopoMap.put("ying", -11358);
        bopoMap.put("yo", -11340);
        bopoMap.put("yong", -11339);
        bopoMap.put("you", -11324);
        bopoMap.put("yu", -11303);
        bopoMap.put("yuan", -11097);
        bopoMap.put("yue", -11077);
        bopoMap.put("yun", -11067);
        bopoMap.put("za", -11055);
        bopoMap.put("zai", -11052);
        bopoMap.put("zan", -11045);
        bopoMap.put("zang", -11041);
        bopoMap.put("zao", -11038);
        bopoMap.put("ze", -11024);
        bopoMap.put("zei", -11020);
        bopoMap.put("zen", -11019);
        bopoMap.put("zeng", -11018);
        bopoMap.put("zha", -11014);
        bopoMap.put("zhai", -10838);
        bopoMap.put("zhan", -10832);
        bopoMap.put("zhang", -10815);
        bopoMap.put("zhao", -10800);
        bopoMap.put("zhe", -10790);
        bopoMap.put("zhen", -10780);
        bopoMap.put("zheng", -10764);
        bopoMap.put("zhi", -10587);
        bopoMap.put("zhong", -10544);
        bopoMap.put("zhou", -10533);
        bopoMap.put("zhu", -10519);
        bopoMap.put("zhua", -10331);
        bopoMap.put("zhuai", -10329);
        bopoMap.put("zhuan", -10328);
        bopoMap.put("zhuang", -10322);
        bopoMap.put("zhui", -10315);
        bopoMap.put("zhun", -10309);
        bopoMap.put("zhuo", -10307);
        bopoMap.put("zi", -10296);
        bopoMap.put("zong", -10281);
        bopoMap.put("zou", -10274);
        bopoMap.put("zu", -10270);
        bopoMap.put("zuan", -10262);
        bopoMap.put("zui", -10260);
        bopoMap.put("zun", -10256);
        bopoMap.put("zuo", -10254);
        bopoMap.put("", -10246);
    }


    /**
     * 得到一个十六进制字符的二进制字符串表示形式
     *
     * @param hex 十六进制字符
     * @return 二进制字符串
     */
    public static String hex2bin(String hex)
    {
        if (hex != null)
        {
            return (String) map_hex2bin.get(hex.toUpperCase());
        }
        else
            return null;
    }

    public static String hexstr2bin(String hex)
    {
        String result = null;

        if (hex != null)
        {
            if (isHex(hex) == false)
                return null;

            hex += "0";
            result = "";
            for (int i = 0; i < hex.length() - 1; i++)
            {
                result += hex2bin(hex.substring(i, i + 1));
            }

        }
        return result;
    }

    public static boolean isHex(String hex)
    {
        if (hex != null)
        {
            hex = hex.toUpperCase();
            for (int i = 0; i < hex.length(); i++)
            {
                int value = hex.charAt(i);
                if (value < 48 || (value > 57 && value < 65) || value > 70)
                    return false;
            }
        }
        else
            return false;

        return true;
    }

    /**
     * 把字符串转化成指定长度的数组
     * @param str 要转换的字符串
     * @param start
     * @param len 指定的转换后的字节类型的数据的总长度
     * @return 字节数组
     */
    public static byte[] getBytes(String str, int start, int len)
    {
        byte[] b = null;

        if (str != null)
        {
            byte[] b1;
            try
            {
                b1 = str.getBytes("GBK");
            } catch (UnsupportedEncodingException e)
            {
                e.printStackTrace();
                b1 = str.getBytes();
            }
            b = GFCommon.bytesCopy(b1, start, len);

        }

        return b;
    }

    /**
     * 返回按指定编码方式编码的字符串
     *
     * @param bArray      字节数组
     * @param charsetName 字符集
     * @return
     */
    public static String getEncodedString(byte[] bArray, String charsetName)
    {
        String ch = null;
        if (charsetName != null)
        {

            try
            {
                ch = new String(bArray, charsetName);
            } catch (UnsupportedEncodingException e)
            {
                e.printStackTrace();
            }
        }

        return ch;

    }

    /**
     * 把表示一个数的十六进制的字符串转化成十进制的数
     *
     * @param hex 十六进制字符串
     * @return 十进制的整数
     */
    public static long hexstr2long(String hex)
    {
        long value = 0;

        if (hex != null)
        {
            hex = hex.toUpperCase();
            if (hex.length() > 16)
                hex = hex.substring(0, 16);

            if (isHex(hex))
            {

                byte[] b = hexstr2bytes(hex);
                value = GFCommon.bytes2long(b);
            }
        }

        return value;
    }

    /**
     * 把字符串转化成固定长的字符串。如果不够指定的长度，在前面添加指定的字符； 如果大于指定的长度，把后面多出的去掉。
     *
     * @param str        要转换的字符串
     * @param len        转换后的长度
     * @param appendChar 添加的字符
     * @return 转换后的字符串
     */
    public static String getFixedLenStr(String str, int len, char appendChar)
    {
        if (str == null || len < 0)
            return null;
        else
        {

            int strLen = 0;
            strLen = str.length();
            if (len <= strLen)
            {
                str = str + appendChar;
                return str.substring(0, len);
            }
            else
            {
                for (int i = 0; i < len - strLen; i++)
                    str = appendChar + str;
                return str;
            }
        }
    }

    /**
     * 把一个二进制字符串的转化成一个整数
     *
     * @param bs 二进制字符串
     * @return 二进制字符串表示的值
     */
    public static long bin2long(String bs)
    {
        long value = 0;

        if (bs != null && bs.length() <= 64)
        {
            byte[] b = bin2bytes(bs);
            value = GFCommon.bytes2long(b);

        }

        return value;
    }

    public static String bin2hex(String bin)
    {
        String hex = null;

        if (bin != null && bin.length() <= 4)
        {
            if (isBinstr(bin))
            {
                for (int i = 0; i < 4 - bin.length(); i++)
                    bin = "0" + bin;

                hex = (String) map_bin2hex.get(bin);
            }
        }
        return hex;
    }

    public static String bin2hexstr(String bin)
    {
        String hex = null;

        if (bin != null)
        {
            if (isBinstr(bin))
            {
                int ys = bin.length() % 4;
                for (int i = 0; ys != 0 && i < 4 - ys; i++)
                    bin = "0" + bin;
                bin += "0";
                hex = "";
                for (int i = 0; i < bin.length() - 4; i += 4)
                {
                    String h = bin2hex(bin.substring(i, i + 4));
                    if (h != null)
                    {
                        if (h.equals("0"))
                        {
                            if (!hex.equals(""))
                                hex += h;
                        }
                        else
                            hex += h;
                    }
                }

                if (hex.equals(""))
                    hex = "0";
            }
        }
        return hex;
    }

    public static byte bin2byte(String bin)
    {
        byte b = 0;

        if (bin != null && bin.length() <= 8)
        {
            if (isBinstr(bin))
            {
                String hex = bin2hexstr(bin);
                b = hex2byte(hex);
            }
        }

        return b;
    }

    public static byte[] bin2bytes(String bin)
    {
        byte[] bs = null;

        if (bin != null)
        {
            String hex = bin2hexstr(bin);
            bs = hexstr2bytes(hex);
        }
        return bs;
    }

    public static int bin2int(String bin)
    {
        int value = 0;

        if (bin != null && bin.length() <= 32)
        {
            if (isBinstr(bin))
            {
                String hex = bin2hexstr(bin);
                value = hexstr2int(hex);
            }
        }
        return value;
    }

    public static boolean isBinstr(String bin)
    {
        boolean result = false;

        if (bin != null)
        {
            byte[] b;
            try
            {
                b = bin.getBytes("GBK");
            } catch (UnsupportedEncodingException e)
            {
                b = bin.getBytes();
                e.printStackTrace();
            }
            for (int i = 0; i < b.length; i++)
            {
                if (b[i] != 48 && b[i] != 49)
                    return false;
            }

            return true;
        }
        return result;
    }

    /**
     * 判断一个字符串是否是数字
     *
     * @param str
     * @return
     */
    public static boolean isNumeric(String str)
    {
        if (str != null)
        {

            try
            {
                str = str.trim();
                double d = Double.parseDouble(str);
                d = d + 1;
                return true;
            } catch (NumberFormatException e)
            {

            }
        }
        return false;
    }

    /**
     * 判断字符串是否全是汉字
     *
     * @param str
     * @return
     */
    public static boolean isAllChinese(String str)
    {
        if (str != null)
        {
            str = quan2ban(str);
            if (str != null)
            {
                try
                {
                    if (str.length() * 2 == str.getBytes("GBK").length)
                        return true;
                } catch (UnsupportedEncodingException e)
                {
                    e.printStackTrace();
                    if (str.length() * 2 == str.getBytes().length)
                        return true;

                }
            }
        }

        return false;
    }

    /**
     * 判断字符串是否全不是汉字
     *
     * @param str
     * @return
     */
    public static boolean isNoChinese(String str)
    {
        if (str != null)
        {
            str = quan2ban(str);
            if (str != null)
            {
                try
                {
                    if (str.length() == str.getBytes("GBK").length)
                        return true;
                } catch (UnsupportedEncodingException e)
                {
                    e.printStackTrace();
                    if (str.length() == str.getBytes().length)
                        return true;
                }
            }
        }

        return false;
    }

    /**
     * 是否是字母
     *
     * @param str
     * @return
     */
    public static boolean isLetter(String str)
    {
        if (str != null)
        {
            byte b[];

            str = str.trim();
            try
            {
                b = str.toUpperCase().getBytes("GBK");
            } catch (UnsupportedEncodingException e)
            {
                e.printStackTrace();
                b = str.toUpperCase().getBytes();
            }
            for (int i = 0; i < b.length; i++)
            {
                if (b[i] < 65 || b[i] > 90)
                    return false;

            }
            return true;
        }
        return false;
    }

    /**
     * 把一个整数转化成8位二进制字符串的表示形式
     *
     * @param value 0--256之间的整数
     * @return 长度为8的二进制字符串
     */
    public static String int2bin(int value)
    {
        if (value >= 0 && value < 256)
        {
            String bin = Integer.toBinaryString(value);
            int len = bin.length();
            for (int i = 0; i < 8 - len; i++)
                bin = "0" + bin;

            return bin;
        }

        return null;
    }

    /**
     * 把表示数字含义的字符串转你成整形
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
            } catch (NumberFormatException e)
            {

            }

        return -1;
    }

    public static long clong(String str)
    {
        if (str != null)
            try
            {
                return new Long(str).longValue();

            } catch (NumberFormatException e)
            {

            }

        return -1;
    }

    /**
     * 在一个字符串中取出指定的子字符串/
     *
     * @param str   字符串
     * @param begin 开始位置，从0数起
     * @param len   子字符串的长度
     * @return 子字符串
     */
    public static String substr(String str, int begin, int len)
    {

        if (str == null)
            return null;
        else
        {
            int strLen = 0;
            strLen = str.length();
            if (begin >= strLen)
                return null;
            else
            {
                if (len > strLen)
                    return null;
                else
                {
                    str += "0";
                    try
                    {
                        return str.substring(begin, len);
                    } catch (IndexOutOfBoundsException e)
                    {
                        return null;
                    }
                }
            }

        }

    }

    /**
     * 把字节数组转化成十六进制的字符串
     *
     * @param b
     */
    public static String bytes2hex(byte[] b)
    {
        String result = "";
        int value;

        if (b != null && b.length > 0)
            for (int i = 0; i < b.length; i++)
            {
                value = (b[i] >>> 4) & 0x0F;
                result += Integer.toHexString(value);
                value = b[i] & 0x0F;
                result += Integer.toHexString(value);
            }

        return result.toUpperCase();
    }

    /**
     * 把UNICODE编码的字符串转化成汉字编码的字符串
     *
     * @param hexString
     * @return
     */
    public static String unicode2gb(String hexString)
    {
        StringBuffer sb = new StringBuffer();

        if (hexString == null)
            return null;

        for (int i = 0; i + 4 <= hexString.length(); i = i + 4)
        {
            try
            {
                int j = Integer.parseInt(hexString.substring(i, i + 4), 16);
                sb.append((char) j);
            } catch (NumberFormatException e)
            {
                return hexString;
            }
        }

        return sb.toString();
    }

    /**
     * 把汉字转化成UNICODE编码的字符串
     *
     * @param gbString
     * @return
     */
    public static String gb2unicode(String gbString)
    {
        String result = "";
        char[] c;
        int value;

        if (gbString == null)
            return null;

        String temp = null;
        c = new char[gbString.length()];
        StringBuffer sb = new StringBuffer(gbString);
        sb.getChars(0, sb.length(), c, 0);
        for (int i = 0; i < c.length; i++)
        {
            value = (int) c[i];
            temp = Integer.toHexString(value);
            result += fill(temp, 4);
        }

        return result.toUpperCase();
    }

    /**
     * 如果字符串的长度没有达到指定的长度，则在字符串前加“0”补够指定的长度
     *
     * @param src 原先的字符串
     * @param len 指定的长度
     * @return 指定长度的字符串
     */
    public static String fill(String src, int len)
    {
        String result = null;

        if (src != null && src.length() <= len)
        {
            result = src;
            for (int i = 0; i < len - src.length(); i++)
            {
                result = "0" + result;
            }
        }
        return result;
    }

    /**
     * 在指定字符串插入到源字符串的指定位置
     *
     * @param src       源字符串
     * @param insertStr 要插入的字符串
     * @param index     要插入的位置
     * @return 插入指定的字符之后的字符串
     */
    public static String insert(String src, String insertStr, int index)
    {
        String result = src;

        if (src != null && insertStr != null)
        {
            String temp = null;

            if (index < 0)
            {
                if (index * -1 > src.length())
                    result = insertStr + src;
                else
                {
                    temp = src.substring(src.length() + index + 1);
                    result = src.substring(0, src.length() + index + 1) + insertStr + temp;
                }
            }
            else if (index >= src.length())
                result = src + insertStr;
            else
            {
                temp = src.substring(index);
                result = src.substring(0, index) + insertStr + temp;
            }
        }
        else if (src == null && insertStr != null)
            result = insertStr;
        return result;

    }

    /**
     * 把相临的两个字符对换，字符串长度为奇数时最后加“F”
     *
     * @param src
     * @return
     */
    public static String interChange(String src)
    {
        String result = null;

        if (src != null)
        {
            if (src.length() % 2 != 0)
                src += "F";
            src += "0";
            result = "";
            for (int i = 0; i < src.length() - 2; i += 2)
            {
                result += src.substring(i + 1, i + 2);
                result += src.substring(i, i + 1);
            }
        }

        return result;
    }

    /**
     * 把数组按指定的编码方式转化成字符串
     *
     * @param b        源数组
     * @param encoding 编码方式
     * @return
     */
    public static String bytes2str(byte[] b, String encoding)
    {
        String result = null;
        int actualLen = 0;
        byte[] ab;

        if (b != null && b.length > 0)
        {
            for (int i = 0; i < b.length; i++)
            {
                if (b[i] == 0)
                    break;
                actualLen++;
            }
            ab = GFCommon.bytesCopy(b, 0, actualLen);
            try
            {
                result = new String(ab, encoding);
            } catch (UnsupportedEncodingException e)
            {
                e.printStackTrace();
            }

        }
        return result;
    }

    /**
     * 把一个字符串按指定的长度分割
     *
     * @param intervalLen 间隔长度
     * @return
     */
    public static String[] split(String src, int intervalLen)
    {
        String[] result = null;
        int len = 0;

        if (src != null && intervalLen > 0)
        {
            len = src.length() / intervalLen;

            if (src.length() % intervalLen != 0)
                len++;

            result = new String[len];

            for (int i = 0, j = 0; i < len - 1; i++, j += intervalLen)
                result[i] = src.substring(j, j + intervalLen);
            result[len - 1] = src.substring((len - 1) * intervalLen);
        }

        return result;
    }

    /**
     * 把字节数组转化成十六进制的字符串
     *
     * @param b
     * @return
     */
    public static String bytes2hexstr(byte[] b)
    {
        return bytes2hexstr(b, false);
    }

    /**
     * 把字节数组转化成十六进制的字符串
     * <p/>
     *
     * @param b
     * @param highBitFirst true:高位优先，即输出的十六进制字符串是从Byte数组的最大下标开始的
     *                     false:低们优先，即输出的十六进制字符串是从Byte数组的最小下标0开始的
     * @return
     */
    public static String bytes2hexstr(byte[] b, boolean highBitFirst)
    {
        String result = null;

        if (b != null && b.length > 0)
        {
            if (highBitFirst)
            {
                for (int i = b.length - 1; i >= 0; i--)
                {
                    String hex = GFString.byte2hex(b[i]);
                    if (result == null)
                        result = hex;
                    else
                        result += hex;
                }
                result = result.toUpperCase();
            }
            else
            {
                for (int i = 0; i < b.length; i++)
                {
                    String hex = GFString.byte2hex(b[i]);
                    if (result == null)
                        result = hex;
                    else
                        result += hex;
                }
                result = result.toUpperCase();
            }
        }
        return result;
    }

    /**
     * 把字节数组转化成十六进制的字符串
     *
     * @param b
     * @return
     */
    public static String bytes2hexstr(byte[] b, int len)
    {
        String result = null;

        if (b != null && b.length > 0 && len <= b.length)
        {
            for (int i = 0; i < len; i++)
            {
                String hex = GFString.byte2hex(b[i]);
                if (result == null)
                    result = hex;
                else
                    result += hex;
            }

            result = result.toUpperCase();
        }
        return result;
    }

    /**
     * 把十六进制字符串转化成字节数组 如果长度不是偶数的话，前面加“0”
     *
     * @param hexstr
     * @return
     */
    public static byte[] hexstr2bytes(String hexstr)
    {
        byte[] b = null;
        int len = 0;

        if (hexstr != null)
        {

            if (hexstr.length() % 2 != 0)
                hexstr = "0" + hexstr;
            len = hexstr.length() / 2;
            b = new byte[len];

            String temp = hexstr + "0";
            for (int i = 0, j = 0; i < temp.length() - 2; i += 2, j++)
            {
                b[j] = GFString.hex2byte(temp.substring(i, i + 2));

            }
        }
        return b;
    }

    public static int hexstr2int(String hex)
    {
        if (hex != null && hex.length() <= 8)
        {
            hex = hex.toUpperCase();
            for (int i = 0; i < hex.length(); i++)
            {
                int value = hex.charAt(i);
                if (value < 48 || (value > 57 && value < 65) || value > 70)
                    return 0;
            }

            byte[] b = hexstr2bytes(hex);
            return GFCommon.bytes2int(b);

        }
        return 0;
    }

    public static String getChineseString(byte[] bArray, String charsetName)
    {
        String ch = null;
        if (charsetName != null)
        {

            try
            {
                ch = new String(bArray, charsetName);
            } catch (UnsupportedEncodingException e)
            {
                e.printStackTrace();
            }
        }

        return ch;

    }

    public static String bytes2str(byte[] b)
    {
        String result = null;
        int actualLen = 0;

        if (b != null && b.length > 0)
        {
            for (int i = 0; i < b.length; i++)
            {
                if (b[i] == 0)
                    break;
                actualLen++;
            }
            byte[] b2 = GFCommon.bytesCopy(b, 0, actualLen);
            if (b2 != null && b2.length > 0)
                result = new String(b2);
        }
        return result;
    }

    /**
     * 把整数转换成指定长度的字符串 如果指定长度小于整数的字符串长度，则只取前面LEN个字符。 如果LEN小0，则返回整数的字符串表示型式
     *
     * @param value
     * @param len
     * @return
     */
    public static String int2str(int value, int len)
    {
        String result = "" + value;
        int l = result.length();

        if (len >= 0)
        {
            if (l <= len)
            {
                for (int i = 0; i < len - l; i++)
                    result = "0" + result;
            }
            else
            {
                result = result.substring(0, len);
            }
        }

        return result;
    }

    /**
     * 把十六进制数转化成字节
     *
     * @param hex
     * @return
     */
    public static byte hex2byte(String hex)
    {
        byte b = 0;
        int value = 0;

        if (hex != null && hex.length() <= 2)
        {
            hex = hex.toUpperCase();
            if (hex.length() == 0)
                return 0;
            else if (hex.length() >= 1)
            {
                value = hex.charAt(0);
                if (value < 48 || (value > 57 && value < 65) || value > 70)
                    return 0;

                if (hex.length() == 2)
                {
                    value = hex.charAt(1);
                    if (value < 48 || (value > 57 && value < 65) || value > 70)
                        return 0;
                }
            }

            try
            {
                b = (byte) Integer.parseInt(hex, 16);
            } catch (NumberFormatException e)
            {
                e.printStackTrace();
            }
        }
        return b;
    }

    /**
     * 把字节转换成十六进制字符串，固定为两个字符长度
     *
     * @param b
     * @return
     */
    public static String byte2hex(byte b)
    {
        String result = null;

        result = Integer.toHexString((b >> 4) & 0x0f);
        result += Integer.toHexString(b & 0xf);
        return result.toUpperCase();
    }

    /**
     * 用UTF-16BE的编码方式把含有全角编码的字符串转成半角编码的字符串
     * <p/>
     * 比如把“汉#＃012０１２YＹ”转成“汉##012012YY”
     * <p/>
     * 对全角的空格处理还有问题
     *
     * @param str
     * @return
     */
    public static String quan2ban(String str)
    {
        String result = null;

        if (str != null)
        {
            try
            {
                byte[] uniBytes = str.getBytes("utf-16be");
                byte[] b = new byte[uniBytes.length];
                for (int i = 0; i < b.length; i++)
                {
                    if (uniBytes[i] == -1)
                    {
                        b[i] = 0;
                        if (i + 1 < uniBytes.length)
                            b[++i] = (byte) (uniBytes[i] + 0x20);

                    }
                    else
                        b[i] = uniBytes[i];
                }

                result = new String(b, "utf-16be");
            } catch (UnsupportedEncodingException e)
            {

                e.printStackTrace();
            }
        }
        return result;
    }

    /**
     * 用UTF-16BE的编码方式把含有半角的字符串转成全角字符串
     *
     * @param str
     * @return
     */
    public static String ban2quan(String str)
    {
        String result = null;

        if (str != null)
        {
            try
            {
                byte[] uniBytes = str.getBytes("utf-16be");
                byte[] b = new byte[uniBytes.length];
                for (int i = 0; i < b.length; i++)
                {
                    if (uniBytes[i] == 0)
                    {
                        b[i] = -1;
                        if (i + 1 < uniBytes.length)
                            b[++i] = (byte) (uniBytes[i] - 0x20);

                    }
                    else
                        b[i] = uniBytes[i];
                }
                result = new String(b, "utf-16be");
            } catch (UnsupportedEncodingException e)
            {

                e.printStackTrace();
            }
        }
        return result;
    }

    /**
     * 用GBK编码进行全角转半角
     *
     * @param str
     * @return
     */
    public static String quan2banGBK(String str)
    {
        String result = null;

        if (str != null)
        {
            try
            {
                int j = 0;
                byte[] uniBytes = str.getBytes("GBK");
                byte[] b = new byte[uniBytes.length];
                for (int i = 0; i < b.length; i++)
                {
                    if (uniBytes[i] == (byte) 0xA3)
                    {
                        if (i + 1 < uniBytes.length)
                            b[j] = (byte) (uniBytes[++i] - 0x80);
                    }
                    else
                    {
                        b[j] = uniBytes[i];
                        if (uniBytes[i] < 0 && i + 1 < b.length)
                            b[++j] = uniBytes[++i];

                    }
                    j++;
                }
                result = new String(b, 0, j, "GBK");
            } catch (UnsupportedEncodingException e)
            {
                e.printStackTrace();
            }
        }
        return result;
    }

    // CDA3 D6B9 20 BABA 23 A3A3 303132 A3B0A3B1A3B259A3D939D3AC39A3D9

    /**
     * 用GBK编码进行半角转全角
     * <p/>
     * 从每个字节判起，如果一个字节的值不大于0X7F，则它是Ascii码的字符。进入下一个判断。
     * <p/>
     * 如果一个字节的值大于0X81，且紧跟着它的下一个字节的值在0x40--0xFE之间，则是汉字或全角字符
     *
     * @param str
     * @return
     */
    public static String ban2quanGBK(String str)
    {
        String result = null;

        if (str != null)
        {
            try
            {
                int j = 0;
                byte[] uniBytes = str.getBytes("GBK");
                byte[] b = new byte[uniBytes.length * 2];
                for (int i = 0; i < uniBytes.length; i++)
                {
                    if (uniBytes[i] >= 0)
                    {
                        b[j] = (byte) 0xA3;
                        if (j + 1 < b.length)
                            b[++j] = (byte) (uniBytes[i] + 0x80);

                    }
                    else
                    {
                        b[j] = uniBytes[i];
                        if (i + 1 < uniBytes.length && j + 1 < b.length)
                            b[++j] = uniBytes[++i];
                    }

                    j++;
                }
                result = new String(b, 0, j, "GBK");
            } catch (UnsupportedEncodingException e)
            {

                e.printStackTrace();
            }
        }
        return result;
    }

    /**
     * 去掉字符串中的空白符
     *
     * @param s
     * @return
     */
    public static String removeSpace(String s)
    {
        String rs = null;
        String s1 = null;

        if (s != null)
        {
            s += " ";
            StringBuffer sb = new StringBuffer();
            for (int i = 0; i < s.length(); i++)
            {
                s1 = s.substring(i, i + 1);
                if (!s1.equals(" "))
                    sb.append(s1);
            }

            rs = sb.toString();
        }
        return rs;
    }

    /**
     * 对字符串中的空格进行格式化,去掉开头和最后的空格,把字符之间的空格缩减为1个.
     * <p/>
     * 比如:<空格><空格>我是一个人<空格><空格><空格>中国人<空格>大学生<空格><空格>
     * <p/>
     * 结果应该为:我是一个人<空格>中国人<空格>大学生
     *
     * @param src
     * @return
     */
    public static String formatSpace(String src)
    {
        String result = null;

        if (src != null)
        {
            result = "";
            String[] ss = src.split(" ");
            for (int i = 0; i < ss.length; i++)
            {
                if (ss[i] != null && ss[i].length() > 0)
                {
                    result += ss[i] + " ";
                }
            }

            if (result.length() > 0 && result.substring(result.length() - 1).equals(" "))
                result = result.substring(0, result.length() - 1);
        }

        return result;
    }

    /**
     * 7-BIT编码 把ASCII码值最高位为0的字符串进行压缩转换成8位二进制表示的字符串
     *
     * @param src
     * @return
     */
    public static String encode7bit(String src)
    {
        String result = null;
        String hex = null;
        byte value;
        int length = 0;
        try
        {
            length = src.getBytes("GBK").length;
        } catch (Exception e)
        {
            length = src.getBytes().length;
            e.printStackTrace();
        }
        if (src != null && src.length() == length)
        {
            result = "";
            byte left = 0;
            byte[] b;
            try
            {
                b = src.getBytes("GBK");
            } catch (UnsupportedEncodingException e)
            {
                e.printStackTrace();
                b = src.getBytes();
            }
            for (int i = 0, j = 0; i < b.length; i++)
            {
                j = i & 7;
                if (j == 0)
                    left = b[i];
                else
                {
                    value = (byte) ((b[i] << (8 - j)) | left);
                    left = (byte) (b[i] >> j);
                    hex = GFString.byte2hex((byte) value);
                    result += hex;
                    if (i == b.length - 1)
                        result += GFString.byte2hex(left);
                }
            }

            result = result.toUpperCase();
        }
        return result;
    }

    /**
     * 对7-BIT编码进行解码
     *
     * @param src 十六进制的字符串，且为偶数个
     * @return 源字符串
     */
    public static String decode7bit(String src)
    {
        String result = null;
        int[] b;
        String temp = null;
        byte srcAscii;
        byte left = 0;

        if (src != null && src.length() % 2 == 0)
        {
            result = "";
            b = new int[src.length() / 2];
            temp = src + "0";
            for (int i = 0, j = 0, k = 0; i < temp.length() - 2; i += 2, j++)
            {
                b[j] = Integer.parseInt(temp.substring(i, i + 2), 16);

                k = j % 7;
                srcAscii = (byte) (((b[j] << k) & 0x7F) | left);
                result += (char) srcAscii;
                left = (byte) (b[j] >>> (7 - k));
                if (k == 6)
                {
                    result += (char) left;
                    left = 0;
                }
                if (j == src.length() / 2)
                    result += (char) left;
            }
        }
        return result;
    }

    /**
     * <pre>
     *                                     是否是手机号码
     *                                     1.11位
     *                                     2.是数字
     *                                     3.以&quot;13&quot;开头
     * </pre>
     *
     * @param msg
     * @return
     */
    public static boolean isMobileNo(String msg)
    {
        // msg = quan2ban(msg);
        // msg = removeSpace(msg);
        if (msg != null && msg.length() == 11)
        {
            if (isNumeric(msg) && (msg.substring(0, 2).equals("13") || msg.substring(0, 2).equals("15")))
            {
                return true;
            }
        }

        return false;
    }

    /**
     * <pre>
     *                                    是否是一个电话号码.
     *                                    首先做预处理,把转全角字符转成半角,并把非数字字符去掉,用空格替代
     *
     *                                    1.长度要至等于7,但不能超过12
     *                                    2.手机号是一个电话号码
     *                                    3.按空格分隔,长度大于等于3且小于等于12的数字字段至少有一个,且最大不超过2个
     * </pre>
     *
     * @param msg
     * @return
     */
    public static boolean isTelNo(String msg)
    {
        // msg = quan2banGBK(msg);
        // msg = removeSpace(msg);
        if (msg != null && msg.length() >= 7)
        {
            String temp = msg + " ";
            String t = null;
            for (int i = 0; i < temp.length() - 1; i++)
            {
                t = temp.substring(i, i + 1);
                if (!isNumeric(t))
                {
                    temp = temp.substring(0, i) + " " + temp.substring(i + 1);
                }
            }

            msg = removeSpace(temp);
            if (isNumeric(msg) && msg.length() >= 7 && msg.length() <= 12)
                if (msg.substring(0, 1).equals("0"))
                {
                    if (msg.length() >= 10)
                        return true;

                }
                else
                {
                    if (isMobileNo(msg))
                        return true;
                    else if (msg.length() <= 8)
                        return true;
                }

        }

        return false;
    }


    /**
     * <pre>
     *                               得到指定位置前的非空格字符
     *                               比如：源字符串为:2室一厅，“室”前一个有效字符为2
     *                               源字符串为：2 室 一厅，“室”前一个有效字符为2
     * </pre>
     *
     * @param msg
     * @param index
     * @return
     */
    public static String getAnteriorNotSpaceChar(String msg, int index)
    {
        String ch = null;

        if (msg != null && index > 0)
        {
            for (int i = index - 1; i >= 0; i--)
            {
                String s = msg.substring(i, i + 1);
                if (!s.equals(" "))
                    return s;
            }
        }

        return ch;
    }

    /**
     * 按字符串长度的长短进行排序
     * <p/>
     * 选用快速排序算法
     *
     * @param list
     * @param long2short True:从长到短.False:从短到长
     * @return
     */
    public static ArrayList<String> sortByLen(ArrayList<String> list, boolean long2short)
    {
        ArrayList<String> rs = null;

        if (list != null)
        {
            rs = new ArrayList<String>(list.size());
            for (String name : list)
            {
                name = GFString.removeSpace(name);
                if (name != null && name.length() > 1)
                {
                    if (rs.size() > 0)
                    {
                        for (int i = 0; i < rs.size(); i++)
                        {
                            if (name.length() >= rs.get(i).length())
                            {
                                rs.add(i, name);
                                break;
                            }
                            else if (i == rs.size() - 1)
                            {
                                rs.add(name);
                                break;
                            }
                            else
                                continue;
                        }
                    }
                    else
                        rs.add(name);
                }
            }

            if (!long2short)
            {
                ArrayList<String> rs2 = new ArrayList<String>();
                for (String s : rs)
                    rs2.add(0, s);
                rs = rs2;
            }
        }

        return rs;

    }

    /**
     * 把指定位置指定长度的字符用新字符串替换掉
     *
     * @param src 源字符串
     * @param index  替换字符串的开始下标
     * @param len    替换的长度
     * @param newstr 新字符串
     * @return
     */
    public static String replace(String src, int index, int len, String newstr)
    {
        String result = src;
        if (src != null && index >= 0 && index < src.length())
        {
            if (newstr == null)
                newstr = "";

            String p1 = src.substring(0, index);

            if (index + len >= src.length())
                result = p1 + newstr;
            else
            {
                String p2 = src.substring(index + len);
                result = p1 + newstr + p2;
            }
        }
        return result;
    }

    public static boolean hasZero(String msg)
    {
        if (msg != null)
        {
            byte[] bb;
            try
            {
                bb = msg.getBytes("GBK");
            } catch (UnsupportedEncodingException e)
            {
                e.printStackTrace();
                bb = msg.getBytes();
            }
            for (byte b : bb)
                if (b == 0)
                    return true;
        }

        return false;
    }

    /**
     * 判断字符串是否是字母数字的
     *
     * @param str
     * @return
     */
    public static boolean isAlphanumeric(String str)
    {
        if (str != null)
        {
            byte[] bs;
            try
            {
                bs = str.getBytes("GBK");
            } catch (UnsupportedEncodingException e)
            {
                e.printStackTrace();
                bs = str.getBytes();
            }
            for (byte b : bs)
            {
                if (b < 48 || b > 57 && b < 65 || b > 90 && b < 97 || b > 122)
                    return false;
            }
            return true;
        }
        return false;
    }

    /**
     * 去掉地名(市/区/县/乡/村)的后缀"市/区/县/乡/镇/村"
     *
     * @param placename
     * @return
     */
    public static String removePlacenameSuffix(String placename)
    {
        int index = -1;
        String[] suffix = {"省", "市", "区", "县", "乡", "镇", "村"};
        if (placename != null && placename.length() > 1)
        {
            for (String s : suffix)
            {
                index = placename.indexOf(s);
                if (placename.length() > 2 && index == placename.length() - 1)
                {
                    placename = placename.substring(0, index);
                    break;
                }
            }
        }

        return placename;
    }

    /**
     * 添加地名后缀(市/区/县/乡/村)的后缀"市/区/县/乡/镇/村"
     *
     * @param placename
     * @param suffix      地名类型 0:省 1:市 2:区 3:县
     * @return
     */
    public static String addPlacenameSuffix(String placename, String suffix)
    {
        int index = -1;
        if (placename != null && placename.length() > 1)
        {
            if (suffix != null && suffix.length() == 1)
            {
                index = placename.indexOf(suffix);
                if (index != placename.length() - 1)
                {
                    placename += suffix;
                }
            }

        }

        return placename;
    }

    /**
     * 比较两个字符串,看str1是否在str2前,按字母排序. 比如:abc是在adc之前
     *
     * @param str1
     * @param str2
     * @return
     */
    public static boolean isBefore(String str1, String str2)
    {
        boolean rs = false;
        if (str1 != null && str2 != null)
        {
            int len = str1.length() < str2.length() ? str1.length() : str2.length();
            byte[] b1;
            byte[] b2;
            try
            {
                b1 = str1.getBytes("GBK");
                b2 = str2.getBytes("GBK");
            } catch (UnsupportedEncodingException e)
            {
                e.printStackTrace();
                b1 = str1.getBytes();
                b2 = str2.getBytes();
            }


            for (int i = 0; i < len; i++)
            {
                if (b2[i] > b1[i])
                    return true;
                else if (b2[i] < b1[i])
                    return false;

            }
        }
        return rs;
    }


    /**
     * 是否是联通手机号码
     *
     * @param sim
     * @return
     */
    public static boolean isUnicommMobile(String sim)
    {
        boolean result = false;
        if (sim != null && sim.length() == 11)
        {
            String part = sim.substring(0, 3);
            if (part.equals("130") || part.equals("131") || part.equals("132") || part.equals("133") || part.equals("153") || part.equals("156"))
                result = true;
        }
        return result;
    }

    /**
     * 是否是联通手机号码
     *
     * @param sim
     * @return
     */
    public static boolean isChinaMobile(String sim)
    {
        boolean result = false;
        if (sim != null && sim.length() == 11)
        {
            String part = sim.substring(0, 3);
            if (part.equals("134") || part.equals("135") || part.equals("136") || part.equals("137") || part.equals("138") || part.equals("139")
                    || part.equals("159") || part.equals("158"))
                result = true;
        }
        return result;

    }

    /**
     * 取得指定位置后面的紧邻的字符
     *
     * @param str
     * @param index
     * @return
     */
    public static String getNextString(String str, int index)
    {
        String rs = null;

        if (str != null && str.length() > 0)
        {
            if (index < 0)
                rs = str.length() > 1 ? str.substring(0, 1) : str;
            else if (index == str.length() - 1)
                rs = null;
            else if (index == str.length() - 2)
                rs = str.substring(index + 1);
            else
                rs = str.substring(index + 1, index + 2);
        }

        return rs;
    }

    /**
     * 对字符串进行原子分隔,比如:解放军第101医院----解 放 军 第 1 0 1 医 院
     *
     * @param str
     * @return
     */
    public static String[] atomSplit(String str)
    {
        if (str == null)
        {
            return null;
        }

        String[] result = null;
        int nLen = str.length();
        result = new String[nLen];
        for (int i = 0; i < nLen; i++)
        {
            result[i] = str.substring(i, i + 1);
        }
        return result;
    }

    public static boolean hasTelNo(String str)
    {
        if (str != null && str.length() >= 7)
        {
            String[] ss = atomSplit(quan2banGBK(str));
            String rs = "";
            for (String s : ss)
            {
                if ("-".equals(s) || "/".equals(s) || "(".equals(s) || ")".equals(s) || isNumeric(s))
                {
                    rs += s;
                }
                else if (rs.length() > 0)
                    break;

            }

            if (rs.length() >= 7)
            {
                if (isMobileNo(rs))
                    return true;
                else if (isTelNo(rs))
                    return true;
            }
        }

        return false;
    }

    /**
     * 找到POS词性标记的位置
     *
     * @param str 分词的字符串
     * @param pos 字词标记
     * @return
     */
    public static int findPos(String str, String pos)
    {
        int result = -1;

        if (str != null && pos != null)
        {
            for (int i = 0; i < str.length(); i++)
            {
                int index = str.indexOf(pos, i);
                if (index + pos.length() == str.length() || (index != -1 && str.substring(index + pos.length()).indexOf(" ") == 0))
                {
                    result = index;
                    break;
                }

            }
        }
        return result;
    }

    /**
     * 去掉词性标注，获取关键词
     *
     * @param str 带词性标注的关键词,比如：团校/bs /sh
     * @return
     */
    public static String getPOSKey(String str)
    {
        if (str != null)
        {
            int index = str.indexOf("/");
            if (index > 0)
            {
                return str.substring(0, index);
            }
        }

        return null;
    }

    /**
     * <pre>
     *  根据词性标注进行分隔，一个关键词可能有多个词性标注，在分隔是视为一个整体。
     *  比如：团校/bs /sh 到 雅仕苑/bs /cm
     *  分隔后:
     *  团校/bs /sh
     *  到
     *  雅仕苑/bs /cm
     * </pre>
     *
     * @param str
     * @return
     */
    public static String[] splitByPOS(String str)
    {
        String[] result = null;
        ArrayList<String> list = new ArrayList<String>();
        if (str != null)
        {
            String[] ss = str.split(" ");
            int i = 0;
            for (String s : ss)
            {
                if (s.indexOf("/") == 0 && i - 1 >= 0 && i - 1 < list.size())
                {
                    String key = list.get(i - 1);
                    list.set(i - 1, key + " " + s);
                }
                else
                {
                    list.add(s);
                    i++;
                }
            }

            result = new String[list.size()];
            list.toArray(result);
        }
        return result;
    }

    /**
     * 得到一个汉字串对应的拼音.只把串的汉字进行转换,其它字符保持不变
     *
     * @param cstr
     * @return
     */
    public static String getBopomofo(String cstr)
    {
        String bopomofo = null;

        if (cstr != null)
        {
            bopomofo = "";
            String[] atoms = atomSplit(cstr);
            for (String atom : atoms)
            {
                if (isAllChinese(atom))
                {
                    byte[] b;
                    try
                    {
                        b = atom.getBytes("GBK");
                    } catch (UnsupportedEncodingException e)
                    {
                        e.printStackTrace();
                        b = atom.getBytes();
                    }
                    int id = (256 + b[0]) * 256 + (256 + b[1]) - 256 * 256;

                    int id1 = -20319;
                    int id2 = 0;
                    String last = null;
                    Iterator itr = bopoMap.keySet().iterator();
                    while (itr.hasNext())
                    {
                        String py = (String) itr.next();
                        id2 = bopoMap.get(py);
                        if (id >= id1 && id < id2)
                        {
                            bopomofo += last == null ? py : last;
                            break;
                        }
                        else
                        {
                            last = py;
                            id1 = id2;
                        }
                    }

                }
                else
                    bopomofo += atom;
            }

            bopomofo = bopomofo.toUpperCase();
        }

        return bopomofo;
    }

    /**
     * 按字典顺序对两个字符串进行比较
     *
     * @param s1
     * @param s2
     * @return
     */
    public static int compareTo(String s1, String s2)
    {
        if (s1 == null && s2 == null)
            return 0;
        else if (s1 != null && s2 == null)
            return 1;
        else if (s1 == null && s2 != null)
            return -1;
        else
        {
            int len = Math.min(s1.length(), s2.length());
            s1 += " ";
            s2 += " ";
            for (int i = 0; i < len; i++)
            {
                String id1 = s1.substring(i, i + 1);
                String id2 = s2.substring(i, i + 1);
                int rs = getID(id1) - getID(id2);

                if (rs != 0)
                    return rs;
            }

            if (s1.length() > s2.length())
                return 1;
            else if (s1.length() < s2.length())
                return -1;
            else
                return 0;
        }

    }

    /**
     * 根据ID号得到对应的GB汉字
     *
     * @param id 0--6767
     * @return
     */
    public static String getGB(int id)
    {
        String result = null;

        if (id >= 0 && id < 6768)
        {
            byte[] b = new byte[2];
            b[0] = (byte) ((id) / 94 + 176);
            b[1] = (byte) ((id) % 94 + 161);
            try
            {
                result = new String(b, "GBK");
            } catch (UnsupportedEncodingException e)
            {
                e.printStackTrace();
            }
        }
        return result;
    }

    public static int getGBID(String s)
    {
        int result = -1;

        if (s != null && s.length() == 1 && isAllChinese(s))
        {
            byte[] b;
            try
            {
                b = s.getBytes("GBK");
            } catch (UnsupportedEncodingException e)
            {
                e.printStackTrace();
                b = s.getBytes();
            }
            int high = b[0] + 256;
            int low = b[1] + 256;

            return (high - 176) * 94 + (low - 161);
        }
        return result;
    }

    public static int getID(String s)
    {
        int result = -1;

        if (s != null && s.length() == 1)
        {
            byte[] b;
            try
            {
                b = s.getBytes("GBK");
            } catch (UnsupportedEncodingException e)
            {
                e.printStackTrace();
                b = s.getBytes();
            }
            if (b.length == 2)
            {
                int high = b[0] + 256;
                int low = b[1] + 256;

                return high * 256 + low;
            }
            else
                return b[0];
        }
        return result;
    }


    public static String getTelcode(String telno)
    {
        String head = null;
        if (isTelNo(telno) && telno.length() > 7)
        {
            int len = telno.length();
            switch (len)
            {
                case 10:
                    head = telno.substring(0, 3);
                    break;
                case 11:
                    if (telno.indexOf("01") == 0 || telno.indexOf("02") == 0)
                        head = telno.substring(0, 3);
                    else
                        head = telno.substring(0, 4);
                    break;
                case 12:
                    if (telno.indexOf("098") == 0 || telno.indexOf("094") == 0 && telno.indexOf("0943") == -1 || telno.indexOf("092") == 0
                            || telno.indexOf("086") == 0 || telno.indexOf("084") == 0 || telno.indexOf("0827") == 0 || telno.indexOf("0829") == 0
                            || telno.indexOf("0822") == 0 || telno.indexOf("0824") == 0 || telno.indexOf("080") == 0 || telno.indexOf("07437") == 0
                            || telno.indexOf("0483") == 0 || telno.indexOf("0788") == 0)
                        head = telno.substring(0, 5);
                    else
                        head = telno.substring(0, 4);
                    break;
            }
        }
        return head;
    }

    /**
     * 得到该词性对应的词
     *
     * @param src      源字符串
     * @param indexPos 词性标记的位置
     */
    public static String getPosWord(String src, int indexPos)
    {
        String result = null;

        if (src != null && indexPos > 0 && indexPos < src.length() - 1)
        {
            String temp = src.substring(0, indexPos + 1);
            String[] ss = temp.split(" ");
            for (int i = ss.length - 1; i >= 0; i--)
            {
                int index = ss[i].indexOf("/");
                if (index == -1)
                    break;
                else if (index > 0)
                {
                    result = ss[i].substring(0, index);
                    break;
                }
            }
        }

        return result;
    }


    /**
     * 取得字符串中第一次出现的整数
     *
     * @param str
     * @return
     */
    public static String getFirstInt(String str)
    {
        String result = null;

        if (str != null)
        {
            String temp = "";
            String[] atoms = atomSplit(str);
            for (int i = 0; i < atoms.length; i++)
            {
                if (isNumeric(atoms[i]))
                    temp += atoms[i];
                if (i + 1 < atoms.length && !isNumeric(atoms[i + 1]))
                    break;
            }

            if (temp.length() > 0)
                result = temp;

        }

        return result;
    }

    /**
     * 字符串当中是否含有无法显示的乱码
     * <p/>
     * GBK 亦采用双字节表示，总体编码范围为 8140-FEFE，首字节在 81-FE 之间，尾字节在 40-FE 之间，剔除 xx7F 一条线。总计
     * 23940 个码位，共收入 21886 个汉字和图形符号，其中汉字（包括部首和构件）21003 个，图形符号 883 个。
     *
     * @param msg
     * @return
     */
    public static boolean hasDisorderChar(String msg)
    {
        if (msg != null)
        {
            String[] atoms = atomSplit(msg);
            for (int i = 0; i < atoms.length; i++)
            {
                byte[] bs;
                try
                {
                    bs = atoms[i].getBytes("GBK");
                } catch (UnsupportedEncodingException e)
                {
                    e.printStackTrace();
                    bs = atoms[i].getBytes();
                }
                if (bs.length == 1)
                {
                    if (bs[0] < 32 || bs[0] > 126)
                        return true;
                }
                else if (bs.length == 2)
                {
                    if (GFCommon.getUnsigned(bs[0]) < 0x81 || GFCommon.getUnsigned(bs[0]) > 0xFE || GFCommon.getUnsigned(bs[1]) < 40
                            || GFCommon.getUnsigned(bs[1]) > 0xFE)
                        return true;
                }

            }
        }

        return false;
    }

    /**
     * 格式化时间成时分秒的形式
     *
     * @param millisTime 毫秒数
     * @return
     */
    public static String formatTime(long millisTime)
    {
        StringBuffer sb = new StringBuffer();
        millisTime = millisTime / 1000;
        sb.append(millisTime / 3600);
        sb.append("小时");
        sb.append((millisTime % 3600) / 60);
        sb.append("分钟");
        sb.append((millisTime % 3600) % 60);
        sb.append("秒");
        return sb.toString();
    }

    public static ArrayList<String> readTxtFile2(String fileName) throws IOException
    {
        ArrayList<String> result = null;
        FileInputStream fin = null;
        InputStreamReader in = null;
        BufferedReader br = null;
        File file = null;
        String value = null;

        if (fileName != null)
        {
            file = new File(fileName);
            if (file.exists())
            {
                result = new ArrayList<String>();
                try
                {
                    fin = new FileInputStream(file);
                    in = new InputStreamReader(fin);
                    br = new BufferedReader(in);
                    while ((value = br.readLine()) != null)
                    {
                        result.add(value);
                    }
                } catch (IOException e)
                {
                    throw new IOException();
                }
            }
        }
        return result;
    }
}
