package com.hankcs.hanlp.mining.word2vec;


import java.io.IOException;

public class Distance extends AbstractClosestVectors
{

    public Distance(String file)
    {
        super(file);
    }

    static void usage()
    {
        System.err.printf("Usage: java %s <FILE>\nwhere FILE contains word projections in the text format\n",
                          Distance.class.getName());
        System.exit(0);
    }

    public static void main(String[] args) throws IOException
    {
        if (args.length < 1) usage();
        new Distance(args[0]).execute();
    }

    protected Result getTargetVector()
    {
        final int words = vectorsReader.getNumWords();
        final int size = vectorsReader.getSize();

        String[] input = null;
        while ((input = nextWords(1, "Enter a word")) != null)
        {
            // linear search the input word in vocabulary
            float[] vec = null;
            int bi = -1;
            double len = 0;
            for (int i = 0; i < words; i++)
            {
                if (input[0].equals(vectorsReader.getWord(i)))
                {
                    bi = i;
                    System.out.printf("\nWord: %s  Position in vocabulary: %d\n", input[0], bi);
                    vec = new float[size];
                    for (int j = 0; j < size; j++)
                    {
                        vec[j] = vectorsReader.getMatrixElement(bi, j);
                        len += vec[j] * vec[j];
                    }
                }
            }
            if (vec == null)
            {
                System.out.printf("%s : Out of dictionary word!\n", input[0]);
                continue;
            }

            len = Math.sqrt(len);
            for (int i = 0; i < size; i++)
            {
                vec[i] /= len;
            }

            return new Result(vec, new int[]{bi});
        }

        return null;
    }
}
