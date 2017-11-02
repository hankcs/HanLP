package com.hankcs.hanlp.mining.word2vec;

import java.io.*;
import java.nio.charset.Charset;
import static com.hankcs.hanlp.utility.Predefine.logger;

public final class VectorsReader
{

    public final Charset ENCODING = Charset.forName("UTF-8");
    int words, size;
    String[] vocab;
    float[][] matrix;
    final String file;

    public VectorsReader(String file)
    {
        this.file = file;
    }

    public void readVectorFile() throws IOException
    {
        logger.info(String.format("reading %s file. please wait...\n", file));

        InputStream is = null;
        Reader r = null;
        BufferedReader br = null;
        try
        {
            is = new FileInputStream(file);
            r = new InputStreamReader(is, ENCODING);
            br = new BufferedReader(r);

            String line = br.readLine();
            words = Integer.parseInt(line.split("\\s+")[0].trim());
            size = Integer.parseInt(line.split("\\s+")[1].trim());

            vocab = new String[words];
            matrix = new float[words][];

            for (int i = 0; i < words; i++)
            {
                line = br.readLine().trim();
                String[] params = line.split("\\s+");
                vocab[i] = params[0];
                matrix[i] = new float[size];
                double len = 0;
                for (int j = 0; j < size; j++)
                {
                    matrix[i][j] = Float.parseFloat(params[j + 1]);
                    len += matrix[i][j] * matrix[i][j];
                }
                len = Math.sqrt(len);
                for (int j = 0; j < size; j++)
                {
                    matrix[i][j] /= len;
                }
            }
        }
        catch (IOException e)
        {
            Utils.closeQuietly(br);
            Utils.closeQuietly(r);
            Utils.closeQuietly(is);

        }
    }

    public int getSize()
    {
        return size;
    }

    public int getNumWords()
    {
        return words;
    }

    public String getWord(int idx)
    {
        return vocab[idx];
    }

    public float getMatrixElement(int row, int column)
    {
        return matrix[row][column];
    }
}
