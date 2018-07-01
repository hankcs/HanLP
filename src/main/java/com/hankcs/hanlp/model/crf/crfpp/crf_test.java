package com.hankcs.hanlp.model.crf.crfpp;


import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.model.perceptron.cli.Args;
import com.hankcs.hanlp.model.perceptron.cli.Argument;

import java.io.*;
import java.util.List;

/**
 * 对应crf_test
 *
 * @author zhifac
 */
public class crf_test
{
    private static class Option
    {
        @Argument(description = "set FILE for model file", alias = "m", required = true)
        String model;
        @Argument(description = "output n-best results", alias = "n")
        Integer nbest = 0;
        @Argument(description = "set INT for verbose level", alias = "v")
        Integer verbose = 0;
        @Argument(description = "set cost factor", alias = "c")
        Double cost_factor = 1.0;
        @Argument(description = "output file path", alias = "o")
        String output;
        @Argument(description = "show this help and exit", alias = "h")
        Boolean help = false;
    }

    public static boolean run(String[] args)
    {
        Option cmd = new Option();
        List<String> unkownArgs = null;
        try
        {
            unkownArgs = Args.parse(cmd, args, false);
        }
        catch (IllegalArgumentException e)
        {
            Args.usage(cmd);
            return false;
        }
        if (cmd.help)
        {
            Args.usage(cmd);
            return true;
        }
        int nbest = cmd.nbest;
        int vlevel = cmd.verbose;
        double costFactor = cmd.cost_factor;
        String model = cmd.model;
        String outputFile = cmd.output;

        TaggerImpl tagger = new TaggerImpl(TaggerImpl.Mode.TEST);
        try
        {
            InputStream stream = IOUtil.newInputStream(model);
            if (!tagger.open(stream, nbest, vlevel, costFactor))
            {
                System.err.println("open error");
                return false;
            }
            String[] restArgs = unkownArgs.toArray(new String[0]);
            if (restArgs.length == 0)
            {
                return false;
            }

            OutputStreamWriter osw = null;
            if (outputFile != null)
            {
                osw = new OutputStreamWriter(IOUtil.newOutputStream(outputFile));
            }
            for (String inputFile : restArgs)
            {
                InputStream fis = IOUtil.newInputStream(inputFile);
                InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
                BufferedReader br = new BufferedReader(isr);

                while (true)
                {
                    TaggerImpl.ReadStatus status = tagger.read(br);
                    if (TaggerImpl.ReadStatus.ERROR == status)
                    {
                        System.err.println("read error");
                        return false;
                    }
                    else if (TaggerImpl.ReadStatus.EOF == status && tagger.empty())
                    {
                        break;
                    }
                    if (!tagger.parse())
                    {
                        System.err.println("parse error");
                        return false;
                    }
                    if (osw == null)
                    {
                        System.out.print(tagger.toString());
                    }
                    else
                    {
                        osw.write(tagger.toString());
                    }
                }
                if (osw != null)
                {
                    osw.flush();
                }
                br.close();
            }
            if (osw != null)
            {
                osw.close();
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public static void main(String[] args)
    {
        crf_test.run(args);
    }
}
