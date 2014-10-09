/*
 * Created on 2005-1-10
 * 
 * TODO To change the template for this generated file go to Window -
 * Preferences - Java - Code Style - Code Templates
 */
package com.hankcs.hanlp.utility;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;


public class GFCommon
{
    /**
     * 把一个数组的数据COPY到另外一个数组中指定的位置。 如果maxlen大于source数组的长度，则在source数组后补0补足够maxlen位；
     * 如果maxlen小于source数组的长度，则把source数组后面多出的几位去掉，剩够maxlen位
     *
     * @param d      destination 数组
     * @param s      source数组
     * @param from   destination的开始位
     * @param maxlen source数组中的数据COPY到destination中之后在其中占有的最大长度
     * @return 目的数组此时被占用的最后一位
     */
    public static int bytesCopy(byte d[], byte s[], int from, int maxlen)
    {
        int end = from;

        if (s != null && d != null)
        {
            if (from >= 0 && maxlen > 0)
            {
                if (s.length < maxlen)
                {
                    for (int i = 0; i < s.length && i + from < d.length; i++)
                        d[i + from] = s[i];
                    end = from + maxlen - 1;

                }
                else
                {
                    for (int i = 0; i < maxlen && i + from < d.length; i++)
                    {
                        d[i + from] = s[i];
                        end = i + from;
                    }
                }
            }
            else if (from < 0 && maxlen > 0)
            {
                for (int i = d.length + from, j = 0; i > 0
                        && j < (s.length > maxlen ? maxlen : s.length); i--, j++)
                {
                    d[i] = s[j];
                    end = i;
                }
            }
        }
        return end;
    }

    /**
     * 从一个数组中的指定位置开始，向后取出len个值； 如果from为负值，则从指定位置向前取出len个值
     *
     * @param src  源数组
     * @param from 起始位置
     * @param len  COPY的长度
     * @return 源数组的一部分
     */
    public static byte[] bytesCopy(byte[] src, int from, int len)
    {
        byte[] result = null;
        int totalLen = 0;

        if (src != null && src.length > 0 && len > 0)
        {
            if (from >= 0)
            {
                totalLen = src.length > from + len ? len : src.length - from;
                result = new byte[Math.abs(len)];

                for (int i = from, j = 0; i < from + totalLen; i++, j++)
                    result[j] = src[i];

            }
            else
            {
                int i0 = src.length + from;// 正向实际位置
                if (i0 - len < 0)
                    totalLen = i0 + 1;
                else
                    totalLen = len;

                result = new byte[totalLen];
                for (int i = i0, j = 0; i >= 0 && j < totalLen; i--, j++)
                    result[j] = src[i];
            }

        }

        return result;
    }

    public static byte[] int2bytes(int a, boolean isHighFirst)
    {
        byte[] result = new byte[4];

        if (isHighFirst)
        {
            result[0] = (byte) (a >> 24 & 0xff);
            result[1] = (byte) (a >> 16 & 0xff);
            result[2] = (byte) (a >> 8 & 0xff);
            result[3] = (byte) (a & 0xff);
        }
        else
        {
            result[3] = (byte) (a >> 24 & 0xff);
            result[2] = (byte) (a >> 16 & 0xff);
            result[1] = (byte) (a >> 8 & 0xff);
            result[0] = (byte) (a & 0xff);
        }
        return result;
    }

    public static byte[] int2bytes(int a)
    {
        return int2bytes(a, true);
    }

    /**
     * 得到一个对象的类名
     *
     * @param obj 对象
     * @return 对象的类名
     */
    public static String getClassName(Object obj)
    {
        String name = null;
        if (obj != null)
        {
            int index = 0;
            String temp = obj.getClass().toString();

            index = temp.lastIndexOf(".");
            if (index > 0 && index < temp.length())
                name = temp.substring(index + 1);

        }
        return name;
    }

    public static int bytes2int(byte[] b)
    {

        return (int) bytes2long(b);
    }

    public static int bytes2int(byte[] b, boolean isHighFirst)
    {

        return (int) bytes2long(b, isHighFirst);
    }

    /**
     * 字节数组转成长整形。按高位在前进行转换。
     *
     * @param b
     * @return
     */
    public static long bytes2long(byte[] b)
    {

        return bytes2long(b, true);
    }

    /**
     * 字节数组转成长整形
     *
     * @param b
     * @param isHighFirst 是否高位在前
     * @return
     */
    public static long bytes2long(byte[] b, boolean isHighFirst)
    {
        long result = 0;

        if (b != null && b.length <= 8)
        {
            long value;

            if (isHighFirst)
            {
                for (int i = b.length - 1, j = 0; i >= 0; i--, j++)
                {
                    value = (long) (b[i] & 0xFF);
                    result += value << (j << 3);
                }
            }
            else
            {
                for (int i = 0, j = 0; i < b.length - 1; i++, j++)
                {
                    value = (long) (b[i] & 0xFF);
                    result += value << (j << 3);
                }
            }
        }

        return result;
    }

    public static String byte2bin(byte b)
    {
        String result = "";

        for (int i = 0; i < 8; i++)
            if (((b >>> (7 - i)) & 1) == 0)
                result += "0";
            else
                result += "1";
        return result;
    }

    public static String int2bin(int value)
    {
        String result = "";

        for (int i = 0; i < 32; i++)
            if (((value >>> (31 - i)) & 1) == 0)
            {
                if (result.length() != 0)
                    result += "0";
            }
            else
                result += "1";
        if (result.length() == 0)
            result = "0";
        return result;
    }

    public static String long2bin(long value)
    {
        String result = "";

        for (int i = 0; i < 64; i++)
            if (((value >>> (63 - i)) & 1) == 0)
                result += "0";
            else
                result += "1";
        return result;
    }

    public static byte[] long2bytes(long value)
    {
        return long2bytes(value, true);
    }

    public static byte[] long2bytes(long value, boolean isHighFirst)
    {
        byte[] b = new byte[8];

        if (isHighFirst)
        {
            for (int i = 0; i < 8; i++)
            {
                b[i] = (byte) (value >> (8 * (7 - i)) & 0xFF);
            }
        }
        else
        {
            for (int i = 0, j = 7; i < 8; i++, j--)
                b[j] = (byte) (value >> (8 * (7 - i)) & 0xFF);

        }

        return b;
    }

    /**
     * 格式化IP地址,把219.11.33.44转化成如下形式:219011033044
     *
     * @param ip
     * @return
     */
    public static String formatIP(String ip)
    {
        String result = null;

        if (ip != null)
        {
            String[] p = new String[4];
            int index = ip.indexOf(".");
            if (index > 0 && index < ip.length() - 1)
                p[0] = ip.substring(0, index);
            else
                return null;
            ip = ip.substring(index + 1);

            index = ip.indexOf(".");
            if (index > 0 && index < ip.length() - 1)
                p[1] = ip.substring(0, index);
            else
                return null;
            ip = ip.substring(index + 1);

            index = ip.indexOf(".");
            if (index > 0 && index < ip.length() - 1)
                p[2] = ip.substring(0, index);
            else
                return null;
            p[3] = ip.substring(index + 1);

            if (p != null && p.length == 4)
            {
                result = GFString.getFixedLenStr(p[0], 3, '0');
                result += GFString.getFixedLenStr(p[1], 3, '0');
                result += GFString.getFixedLenStr(p[2], 3, '0');
                result += GFString.getFixedLenStr(p[3], 3, '0');
            }
        }
        return result;
    }

    public static boolean isActiveThread(ThreadGroup group, String threadName)
    {

        if (group != null && threadName != null)
        {
            Thread[] thd = new Thread[group.activeCount()];
            group.enumerate(thd);

            String name = null;
            for (int i = 0; i < thd.length && thd[i] != null; i++)
            {
                name = thd[i].getName();
                if (name != null && name.equals(threadName))
                    return true;
            }
        }

        return false;
    }

    /**
     * 得到系统信息。
     * <p/>
     * 比如：操作系统，JAVA虚拟机等
     *
     * @return
     */
    public static String getSystemInfo()
    {
        String result = "os.name:" + System.getProperty("os.name") + "\n"
                + "os.arch:" + System.getProperty("os.arch") + "\n\n"
                + "java.vendor:" + System.getProperty("java.vendor") + "\n"
                + "java.home:" + System.getProperty("java.home") + "\n"
                + "java.version:" + System.getProperty("java.version") + "\n"
                + "java.vm.version:" + System.getProperty("java.vm.version")
                + "\n\n" + "user.name:" + System.getProperty("user.name")
                + "\n" + "user.dir:" + System.getProperty("user.dir");

        return result;
    }

    /**
     * 获取数据库的连接。
     *
     * @param driver   数据库驱动参数
     * @param url      数据库URL地址
     * @param userName 数据库登陆用户名
     * @param pwd      登陆密码
     * @param conn     数据库连接
     * @return 数据库连接
     */
    public static Connection getConn(String driver, String url,
                                     String userName, String pwd)
    {
        Connection conn = null;

        if (driver != null && url != null && userName != null && pwd != null)
        {
            try
            {
                Class.forName(driver);
                conn = DriverManager.getConnection(url, userName, pwd);
                if (conn != null)
                {
                    String str = "建立和远程数据库的连接!";
                    System.out.println(str);
                }
            } catch (ClassNotFoundException e)
            {
            } catch (SQLException e)
            {
                e.printStackTrace();
            }
        }

        return conn;
    }

    /**
     * 取得Class文件所在的路径。
     *
     * @param className 类名
     * @return Class文件所在的路径，不包括Class文件本身的包路径。比如：com.gftech.web.Test,返回的路径格式如下/E:/gftech/project/web/bin/
     */
    public static String getClassPath(String className)
    {

        try
        {
            return Class.forName(className).getClassLoader().getResource("")
                    .getPath();
        } catch (ClassNotFoundException e)
        {
            e.printStackTrace();
        }

        return null;
    }

    /**
     * 取得Class文件所在的路径。
     *
     * @param objName 对象名称
     * @return Class文件所在的路径，不包括Class文件本身的包路径。比如：com.gftech.web.Test,返回的路径格式如下/E:/gftech/project/web/bin/
     */
    public static String getClassPath(Object objName)
    {

        return objName.getClass().getClassLoader().getResource("").getPath();

    }

    /**
     * 返回Jsp应用程序中WEB-INF的路径。
     *
     * @param classPath WEB-INF/classes的路径，格式为/E:/web/myjsp/WEB-INF/classes/
     * @return WEB-INF的路径，格式为E:\web\myjsp\WEB-INF\
     */
    public static String getWebinfPath(String classPath)
    {
        String path = null;
        if (classPath != null)
        {
            String[] strs = classPath.split("/");
            path = "";
            for (int i = 1; i < strs.length - 1; i++)
            {
                if (strs[i] != null)
                    path += strs[i] + System.getProperty("file.separator");
            }

        }

        return path;
    }

    /**
     * 产生一个不大于seed的随机数
     *
     * @param seed 产生随机数的种子
     * @return 随机数
     */
    public static int random(int seed)
    {
        long result = 0;
        if (seed != 0)
        {
            double d = Math.random();
            String temp = d + "";
            int len = temp.length() - 2;// 去掉开头两位
            d = d * Math.pow(10, len);
            result = (long) d % seed;
        }
        return (int) result;
    }

    /**
     * 生成一个在min到max之间的随机数
     *
     * @param min 最小值
     * @param max 最大值
     * @return
     */
    public static int random(int min, int max)
    {
        int rd = random(max);
        if (rd >= min)
            return rd;
        else
            return random(min, max);

    }

    /**
     * 得到数组中0第一次数组b中出现的位置
     *
     * @param b
     * @return
     */
    public static int getZeroIndex(byte[] b)
    {
        if (b != null)
        {
            for (int i = 0; i < b.length; i++)
            {
                if (b[i] == 0)
                    return i;
            }
        }
        return -1;
    }

    /**
     * 数组中是否含有0值
     *
     * @param b
     * @return
     */
    public static boolean isHasZero(byte[] b)
    {
        if (b == null)
            return true;
        for (byte b1 : b)
            if (b1 == 0)
                return true;

        return false;
    }


    public static int getUnsigned(byte b)
    {
        if (b > 0)
            return (int) b;
        else
            return (b & 0x7F + 128);
    }
}