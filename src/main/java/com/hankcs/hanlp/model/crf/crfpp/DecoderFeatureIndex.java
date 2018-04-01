package com.hankcs.hanlp.model.crf.crfpp;

import com.hankcs.hanlp.collection.dartsclone.DoubleArray;
import com.hankcs.hanlp.collection.trie.datrie.MutableDoubleArrayTrieInteger;
import com.hankcs.hanlp.corpus.io.IOUtil;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * @author zhifac
 */
public class DecoderFeatureIndex extends FeatureIndex
{
    private MutableDoubleArrayTrieInteger dat;

    public DecoderFeatureIndex()
    {
        dat = new MutableDoubleArrayTrieInteger();
    }

    public int getID(String key)
    {
        return dat.get(key);
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
            dat = (MutableDoubleArrayTrieInteger) ois.readObject();
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
        try
        {
            if (!open(IOUtil.newInputStream(binarymodel)))
            {
                System.err.println("Fail to read binary model " + binarymodel);
                return false;
            }
            OutputStreamWriter osw = new OutputStreamWriter(IOUtil.newOutputStream(textmodel), "UTF-8");
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

            for (MutableDoubleArrayTrieInteger.KeyValuePair pair : dat)
            {
                osw.write(pair.getValue() + " " + pair.getKey() + "\n");
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
            try
            {
                if (open(IOUtil.newInputStream(binFileName)))
                {
                    System.out.println("Found binary model " + binFileName);
                    return true;
                }
            }
            catch (IOException e)
            {
                // load text model
            }

            isr = new InputStreamReader(IOUtil.newInputStream(filename1), "UTF-8");
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
            while ((line = br.readLine()) != null && line.length() > 0)
            {
                String[] content = line.trim().split(" ");
                dat.put(content[1], Integer.valueOf(content[0]));
            }
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
                System.out.println("Writing binary model to " + binFileName);
                ObjectOutputStream oos = new ObjectOutputStream(IOUtil.newOutputStream(binFileName));
                oos.writeObject(version);
                oos.writeObject(costFactor_);
                oos.writeObject(maxid_);
                oos.writeObject(xsize_);
                oos.writeObject(y_);
                oos.writeObject(unigramTempls_);
                oos.writeObject(bigramTempls_);
                oos.writeObject(dat);
                oos.writeObject(alpha_);
                oos.close();
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
