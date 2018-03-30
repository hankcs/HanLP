package com.hankcs.hanlp.model.crf.crfpp;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * @author zhifac
 */
public class DecoderFeatureIndex extends FeatureIndex
{
    private DoubleArrayTrieInteger dat;

    public DecoderFeatureIndex()
    {
        dat = new DoubleArrayTrieInteger();
    }

    public int getID(String key)
    {
        return dat.exactMatchSearch(key);
    }

    public boolean open(InputStream stream)
    {
        try
        {
            ObjectInputStream ois = new ObjectInputStream(stream);
            int version = (Integer) ois.readObject();
            costFactor_ = (Double) ois.readObject();
            maxid_ = (Integer) ois.readObject();
            xsize_ = (Integer) ois.readObject();
            y_ = (List<String>) ois.readObject();
            unigramTempls_ = (List<String>) ois.readObject();
            bigramTempls_ = (List<String>) ois.readObject();
            int[] datBase = (int[]) ois.readObject();
            int[] datCheck = (int[]) ois.readObject();
            dat = new DoubleArrayTrieInteger();
            dat.setBase(datBase);
            dat.setCheck(datCheck);
            dat.setSize(datBase.length);
            alpha_ = (double[]) ois.readObject();
            ois.close();
            return true;
        }
        catch (Exception e)
        {
            e.printStackTrace();
            return false;
        }
    }

    public boolean convert(String binarymodel, String textmodel)
    {
        File binFile = new File(binarymodel);
        if (binFile.exists())
        {
            try
            {
                if (!open(new FileInputStream(binFile)))
                {
                    System.err.println("Fail to read binary model " + binarymodel);
                    return false;
                }
                OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(textmodel), "UTF-8");
                osw.write("version: " + Encoder.MODEL_VERSION + "\n");
                osw.write("cost-factor: " + costFactor_ + "\n");
                osw.write("maxid: " + maxid_ + "\n");
                osw.write("xsize: " + xsize_ + "\n");
                osw.write("\n");
                for (String y : y_)
                {
                    osw.write(y + "\n");
                }
                osw.write("\n");
                for (String utempl : unigramTempls_)
                {
                    osw.write(utempl + "\n");
                }
                for (String bitempl : bigramTempls_)
                {
                    osw.write(bitempl + "\n");
                }
                osw.write("\n");

                dat.recoverKeyValue();
                if (dat.getKey().size() != dat.getValue().length || dat.getKey().isEmpty())
                {
                    System.err.println("fail to recover key values for DoubleArrayTrie");
                    return false;
                }

                for (int i = 0; i < dat.getKey().size(); i++)
                {
                    osw.write(dat.getValue()[i] + " " + dat.getKey().get(i) + "\n");
                }
                osw.write("\n");

                for (int k = 0; k < maxid_; k++)
                {
                    String val = new DecimalFormat("0.0000000000000000").format(alpha_[k]);
                    osw.write(val + "\n");
                }
                osw.close();
                return true;
            }
            catch (Exception e)
            {
                e.printStackTrace();
                return false;
            }
        }
        else
        {
            System.err.println(binarymodel + " does not exist");
            return false;
        }
    }

    public boolean openTextModel(String filename1, boolean cacheBinModel)
    {
        InputStreamReader isr = null;
        try
        {
            String binFileName = filename1 + ".bin";
            File binFile = new File(binFileName);
            if (binFile.exists())
            {
                System.out.println("Found binary model " + binFileName);
                return open(new FileInputStream(binFile));
            }
            isr = new InputStreamReader(new FileInputStream(filename1), "UTF-8");
            BufferedReader br = new BufferedReader(isr);
            String line;

            int version = Integer.valueOf(br.readLine().substring("version: ".length()));
            costFactor_ = Double.valueOf(br.readLine().substring("cost-factor: ".length()));
            maxid_ = Integer.valueOf(br.readLine().substring("maxid: ".length()));
            xsize_ = Integer.valueOf(br.readLine().substring("xsize: ".length()));
            System.out.println("Done reading meta-info");
            br.readLine();

            while ((line = br.readLine()) != null && line.length() > 0)
            {
                y_.add(line);
            }
            System.out.println("Done reading labels");
            while ((line = br.readLine()) != null && line.length() > 0)
            {
                if (line.startsWith("U"))
                {
                    unigramTempls_.add(line);
                }
                else if (line.startsWith("B"))
                {
                    bigramTempls_.add(line);
                }
            }
            System.out.println("Done reading templates");
            List<String> keys = new ArrayList<String>();
            List<Integer> values = new ArrayList<Integer>();
            while ((line = br.readLine()) != null && line.length() > 0)
            {
                String[] content = line.trim().split(" ");
                keys.add(content[1]);
                values.add(Integer.valueOf(content[0]));
            }
            System.out.println("Done reading feature indices");
            int[] valueIntArr = new int[values.size()];
            for (int i = 0; i < values.size(); i++)
            {
                valueIntArr[i] = values.get(i);
            }
            dat.build(keys, null, valueIntArr, keys.size());
            System.out.println("Done building trie");
            List<Double> alpha = new ArrayList<Double>();
            while ((line = br.readLine()) != null && line.length() > 0)
            {
                alpha.add(Double.valueOf(line));
            }
            System.out.println("Done reading weights");
            alpha_ = new double[alpha.size()];
            for (int i = 0; i < alpha.size(); i++)
            {
                alpha_[i] = alpha.get(i);
            }
            br.close();

            if (cacheBinModel)
            {
                ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(binFileName));
                oos.writeObject(version);
                oos.writeObject(costFactor_);
                oos.writeObject(maxid_);
                oos.writeObject(xsize_);
                oos.writeObject(y_);
                oos.writeObject(unigramTempls_);
                oos.writeObject(bigramTempls_);
                oos.writeObject(dat.getBase());
                oos.writeObject(dat.getCheck());
                oos.writeObject(alpha_);
                oos.close();
                System.out.println("Writing binary model to " + binFileName);
            }
        }
        catch (Exception e)
        {
            if (isr != null)
            {
                try
                {
                    isr.close();
                }
                catch (Exception e2)
                {
                }
            }
            e.printStackTrace();
            System.err.println("Error reading " + filename1);
            return false;
        }
        return true;
    }

    public static void main(String[] args)
    {
        if (args.length < 2)
        {
            return;
        }
        else
        {
            DecoderFeatureIndex featureIndex = new DecoderFeatureIndex();
            if (!featureIndex.convert(args[0], args[1]))
            {
                System.err.println("fail to convert binary model to text model");
            }
        }
    }
}
