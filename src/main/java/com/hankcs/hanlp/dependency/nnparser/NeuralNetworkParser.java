/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/10/30 20:00</create-date>
 *
 * <copyright file="NeuralNetworkParser.java" company="码农场">
 * Copyright (c) 2008-2015, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser;

import com.hankcs.hanlp.corpus.io.*;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.TextUtility;
import com.hankcs.hanlp.dependency.nnparser.action.Action;
import com.hankcs.hanlp.dependency.nnparser.action.ActionFactory;
import com.hankcs.hanlp.dependency.nnparser.option.SpecialOption;
import com.hankcs.hanlp.dependency.nnparser.util.math;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.*;
import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * @author hankcs
 */
public class NeuralNetworkParser implements ICacheAble
{
    Matrix W1;
    Matrix W2;
    Matrix E;
    Matrix b1;
    Matrix saved;

    Alphabet forms_alphabet;
    Alphabet postags_alphabet;
    Alphabet deprels_alphabet;
    /**
     * 少量类目数的聚类
     */
    Alphabet cluster4_types_alphabet;
    /**
     * 中等类目数的聚类
     */
    Alphabet cluster6_types_alphabet;
    /**
     * 大量类目数的聚类
     */
    Alphabet cluster_types_alphabet;

    Map<Integer, Integer> precomputation_id_encoder;
    /**
     * 将词映射到词聚类中的某个类
     */
    Map<Integer, Integer> form_to_cluster4;
    Map<Integer, Integer> form_to_cluster6;
    Map<Integer, Integer> form_to_cluster;

    /**
     * 神经网络分类器
     */
    NeuralNetworkClassifier classifier;
    /**
     * 动作转移系统
     */
    TransitionSystem system;
    /**
     * 根节点词语
     */
    String root;

    /**
     * 语料库之外的词语的id
     */
    int kNilForm;
    /**
     * 语料库之外的词语的词性的id
     */
    int kNilPostag;
    /**
     * 语料库之外的依存关系名称的id
     */
    int kNilDeprel;
    int kNilDistance;
    int kNilValency;
    int kNilCluster4;
    int kNilCluster6;
    int kNilCluster;

    /**
     * 词语特征在特征空间中的起始位置
     */
    int kFormInFeaturespace;
    int kPostagInFeaturespace;
    int kDeprelInFeaturespace;
    int kDistanceInFeaturespace;
    int kValencyInFeaturespace;
    int kCluster4InFeaturespace;
    int kCluster6InFeaturespace;
    int kClusterInFeaturespace;
    int kFeatureSpaceEnd;

    int nr_feature_types;

    /**
     * 指定使用距离特征，具体参考Zhang and Nivre (2011)
     */
    boolean use_distance;
    /**
     * 指定使用valency特征，具体参考Zhang and Nivre (2011)
     */
    boolean use_valency;
    /**
     * 指定使用词聚类特征，具体参考Guo et. al, (2015)
     */
    boolean use_cluster;

    static String model_header;

    /**
     * 加载parser模型
     * @param path
     * @return
     */
    public boolean load(String path)
    {
        String binPath = path + Predefine.BIN_EXT;
        if (load(ByteArrayStream.createByteArrayStream(binPath))) return true;
        if (!loadTxt(path)) return false;
        try
        {
            logger.info("正在缓存" + binPath);
            DataOutputStream out = new DataOutputStream(IOUtil.newOutputStream(binPath));
            save(out);
            out.close();
        }
        catch (Exception e)
        {
            logger.warning("缓存" + binPath + "失败：\n" + TextUtility.exceptionToString(e));
        }

        return true;
    }

    /**
     * 从txt加载
     * @param path
     * @return
     */
    public boolean loadTxt(String path)
    {
        IOUtil.LineIterator lineIterator = new IOUtil.LineIterator(path);
        model_header = lineIterator.next();
        if (model_header == null) return false;
        root = lineIterator.next();
        use_distance = "1".equals(lineIterator.next());
        use_valency = "1".equals(lineIterator.next());
        use_cluster = "1".equals(lineIterator.next());

        W1 = read_matrix(lineIterator);
        W2 = read_matrix(lineIterator);
        E = read_matrix(lineIterator);
        b1 = read_vector(lineIterator);
        saved = read_matrix(lineIterator);

        forms_alphabet = read_alphabet(lineIterator);
        postags_alphabet = read_alphabet(lineIterator);
        deprels_alphabet = read_alphabet(lineIterator);

        precomputation_id_encoder = read_map(lineIterator);

        if (use_cluster)
        {
            cluster4_types_alphabet = read_alphabet(lineIterator);
            cluster6_types_alphabet = read_alphabet(lineIterator);
            cluster_types_alphabet = read_alphabet(lineIterator);

            form_to_cluster4 = read_map(lineIterator);
            form_to_cluster6 = read_map(lineIterator);
            form_to_cluster = read_map(lineIterator);
        }

        assert !lineIterator.hasNext() : "文件有残留，可能是读取逻辑不对";

        classifier = new NeuralNetworkClassifier(W1, W2, E, b1, saved, precomputation_id_encoder);
        classifier.canonical();

        return true;
    }

    /**
     * 保存到磁盘
     * @param out
     * @throws Exception
     */
    public void save(DataOutputStream out) throws Exception
    {
        TextUtility.writeString(model_header, out);
        TextUtility.writeString(root, out);

        out.writeInt(use_distance ? 1 : 0);
        out.writeInt(use_valency ? 1 : 0);
        out.writeInt(use_cluster ? 1 : 0);

        W1.save(out);
        W2.save(out);
        E.save(out);
        b1.save(out);
        saved.save(out);

        forms_alphabet.save(out);
        postags_alphabet.save(out);
        deprels_alphabet.save(out);

        save_map(precomputation_id_encoder, out);

        if (use_cluster)
        {
            cluster4_types_alphabet.save(out);
            cluster6_types_alphabet.save(out);
            cluster_types_alphabet.save(out);

            save_map(form_to_cluster4, out);
            save_map(form_to_cluster6 , out);
            save_map(form_to_cluster , out);
        }
    }

    /**
     * 从bin加载
     * @param byteArray
     * @return
     */
    public boolean load(ByteArray byteArray)
    {
        if (byteArray == null) return false;
        model_header = byteArray.nextString();
        root = byteArray.nextString();

        use_distance = byteArray.nextInt() == 1;
        use_valency = byteArray.nextInt() == 1;
        use_cluster = byteArray.nextInt() == 1;

        W1 = new Matrix();
        W1.load(byteArray);
        W2 = new Matrix();
        W2.load(byteArray);
        E = new Matrix();
        E .load(byteArray);
        b1 = new Matrix();
        b1 .load(byteArray);
        saved = new Matrix();
        saved .load(byteArray);

        forms_alphabet = new Alphabet();
        forms_alphabet .load(byteArray);
        postags_alphabet = new Alphabet();
        postags_alphabet .load(byteArray);
        deprels_alphabet = new Alphabet();
        deprels_alphabet .load(byteArray);

        precomputation_id_encoder = read_map(byteArray);

        if (use_cluster)
        {
            cluster4_types_alphabet = new Alphabet();
            cluster4_types_alphabet.load(byteArray);
            cluster6_types_alphabet = new Alphabet();
            cluster6_types_alphabet .load(byteArray);
            cluster_types_alphabet = new Alphabet();
            cluster_types_alphabet .load(byteArray);

            form_to_cluster4 = read_map(byteArray);
            form_to_cluster6 = read_map(byteArray);
            form_to_cluster = read_map(byteArray);
        }

        assert !byteArray.hasMore() : "文件有残留，可能是读取逻辑不对";

        classifier = new NeuralNetworkClassifier(W1, W2, E, b1, saved, precomputation_id_encoder);
        classifier.canonical();

        return true;
    }

    private static Matrix read_matrix(IOUtil.LineIterator lineIterator)
    {
        String[] rc = lineIterator.next().split("\t");
        int rows = Integer.valueOf(rc[0]);
        int cols = Integer.valueOf(rc[1]);
        double[][] valueArray = new double[rows][cols];
        for (double[] valueRow : valueArray)
        {
            String[] args = lineIterator.next().split("\t");
            for (int i = 0; i < valueRow.length; i++)
            {
                valueRow[i] = Double.valueOf(args[i]);
            }
        }

        return new Matrix(valueArray);
    }

    private static Matrix read_vector(IOUtil.LineIterator lineIterator)
    {
        int rows = Integer.valueOf(lineIterator.next());
        double[][] valueArray = new double[rows][1];
        String[] args = lineIterator.next().split("\t");
        for (int i = 0; i < rows; i++)
        {
            valueArray[i][0] = Double.valueOf(args[i]);
        }

        return new Matrix(valueArray);
    }

    private static Alphabet read_alphabet(IOUtil.LineIterator lineIterator)
    {
        int size = Integer.valueOf(lineIterator.next());
        TreeMap<String, Integer> map = new TreeMap<String, Integer>();
        for (int i = 0; i < size; i++)
        {
            String[] args = lineIterator.next().split("\t");
            map.put(args[0], Integer.valueOf(args[1]));
        }

        Alphabet trie = new Alphabet();
        trie.build(map);

        return trie;
    }

    private static Map<Integer, Integer> read_map(IOUtil.LineIterator lineIterator)
    {
        int size = Integer.valueOf(lineIterator.next());
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i = 0; i < size; i++)
        {
            String[] args = lineIterator.next().split("\t");
            map.put(Integer.valueOf(args[0]), Integer.valueOf(args[1]));
        }

        return map;
    }

    private static Map<Integer, Integer> read_map(ByteArray byteArray)
    {
        int size = byteArray.nextInt();
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i = 0; i < size; i++)
        {
            map.put(byteArray.nextInt(), byteArray.nextInt());
        }

        return map;
    }

    private static void save_map(Map<Integer, Integer> map, DataOutputStream out) throws IOException
    {
        out.writeInt(map.size());
        for (Map.Entry<Integer, Integer> entry : map.entrySet())
        {
            out.writeInt(entry.getKey());
            out.writeInt(entry.getValue());
        }
    }

    /**
     * 初始化
     */
    void setup_system()
    {
        system = new TransitionSystem();
        system.set_root_relation(deprels_alphabet.idOf(root));
        system.set_number_of_relations(deprels_alphabet.size() - 2);
    }

    /**
     * 初始化特征空间的长度等信息
     */
    void build_feature_space()
    {
        kFormInFeaturespace = 0;
        kNilForm = forms_alphabet.idOf(SpecialOption.NIL);
        kFeatureSpaceEnd = forms_alphabet.size();

        kPostagInFeaturespace = kFeatureSpaceEnd;
        kNilPostag = kFeatureSpaceEnd + postags_alphabet.idOf(SpecialOption.NIL);
        kFeatureSpaceEnd += postags_alphabet.size();

        kDeprelInFeaturespace = kFeatureSpaceEnd;
        kNilDeprel = kFeatureSpaceEnd + deprels_alphabet.idOf(SpecialOption.NIL);
        kFeatureSpaceEnd += deprels_alphabet.size();

        kDistanceInFeaturespace = kFeatureSpaceEnd;
        kNilDistance = kFeatureSpaceEnd + (use_distance ? 8 : 0);
        kFeatureSpaceEnd += (use_distance ? 9 : 0);

        kValencyInFeaturespace = kFeatureSpaceEnd;
        kNilValency = kFeatureSpaceEnd + (use_valency ? 8 : 0);
        kFeatureSpaceEnd += (use_valency ? 9 : 0);

        kCluster4InFeaturespace = kFeatureSpaceEnd;
        if (use_cluster)
        {
            kNilCluster4 = kFeatureSpaceEnd + cluster4_types_alphabet.idOf(SpecialOption.NIL);
            kFeatureSpaceEnd += cluster4_types_alphabet.size();
        }
        else
        {
            kNilCluster4 = kFeatureSpaceEnd;
        }

        kCluster6InFeaturespace = kFeatureSpaceEnd;
        if (use_cluster)
        {
            kNilCluster6 = kFeatureSpaceEnd + cluster6_types_alphabet.idOf(SpecialOption.NIL);
            kFeatureSpaceEnd += cluster6_types_alphabet.size();
        }
        else
        {
            kNilCluster6 = kFeatureSpaceEnd;
        }

        kClusterInFeaturespace = kFeatureSpaceEnd;
        if (use_cluster)
        {
            kNilCluster = kFeatureSpaceEnd + cluster_types_alphabet.idOf(SpecialOption.NIL);
            kFeatureSpaceEnd += cluster_types_alphabet.size();
        }
        else
        {
            kNilCluster = kFeatureSpaceEnd;
        }

    }

    /**
     * 将实例转为依存树
     * @param data 实例
     * @param dependency 输出的依存树
     * @param with_dependencies 是否输出依存关系（仅在解析后才有意义）
     */
    void transduce_instance_to_dependency(final Instance data,
                                          Dependency dependency, boolean with_dependencies)
    {
        int L = data.forms.size();
        for (int i = 0; i < L; ++i)
        {
            Integer form = forms_alphabet.idOf(data.forms.get(i));
            if (form == null)
            {
                form = forms_alphabet.idOf(SpecialOption.UNKNOWN);
            }
            Integer postag = postags_alphabet.idOf(data.postags.get(i));
            if (postag == null) postag = postags_alphabet.idOf(SpecialOption.UNKNOWN);
            int deprel = (with_dependencies ? deprels_alphabet.idOf(data.deprels.get(i)) : -1);

            dependency.forms.add(form);
            dependency.postags.add(postag);
            dependency.heads.add(with_dependencies ? data.heads.get(i) : -1);
            dependency.deprels.add(with_dependencies ? deprel : -1);
        }
    }

    /**
     * 获取词聚类特征
     * @param data 输入数据
     * @param cluster4
     * @param cluster6
     * @param cluster
     */
    void get_cluster_from_dependency(final Dependency data,
                                     List<Integer> cluster4,
                                     List<Integer> cluster6,
                                     List<Integer> cluster)
    {
        if (use_cluster)
        {
            int L = data.forms.size();
            for (int i = 0; i < L; ++i)
            {
                int form = data.forms.get(i);
                cluster4.add(i == 0 ?
                                     cluster4_types_alphabet.idOf(SpecialOption.ROOT) : form_to_cluster4.get(form));
                cluster6.add(i == 0 ?
                                     cluster6_types_alphabet.idOf(SpecialOption.ROOT) : form_to_cluster6.get(form));
                cluster.add(i == 0 ?
                                    cluster_types_alphabet.idOf(SpecialOption.ROOT) : form_to_cluster.get(form));
            }
        }
    }

    /**
     * 依存分析
     * @param data 实例
     * @param heads 依存指向的储存位置
     * @param deprels 依存关系的储存位置
     */
    void predict(final Instance data, List<Integer> heads,
                 List<String> deprels)
    {
        Dependency dependency = new Dependency();
        List<Integer> cluster = new ArrayList<Integer>(), cluster4 = new ArrayList<Integer>(), cluster6 = new ArrayList<Integer>();
        transduce_instance_to_dependency(data, dependency, false);
        get_cluster_from_dependency(dependency, cluster4, cluster6, cluster);

        int L = data.forms.size();
        State[] states = new State[L * 2];
        for (int i = 0; i < states.length; i++)
        {
            states[i] = new State();
        }
        states[0].copy(new State(dependency));
        system.transit(states[0], ActionFactory.make_shift(), states[1]);
        for (int step = 1; step < L * 2 - 1; ++step)
        {
            List<Integer> attributes = new ArrayList<Integer>();
            if (use_cluster)
            {
                get_features(states[step], cluster4, cluster6, cluster, attributes);
            }
            else
            {
                get_features(states[step], attributes);
            }

            List<Double> scores = new ArrayList<Double>(system.number_of_transitions());
            classifier.score(attributes, scores);

            List<Action> possible_actions = new ArrayList<Action>();
            system.get_possible_actions(states[step], possible_actions);

            int best = -1;
            for (int j = 0; j < possible_actions.size(); ++j)
            {
                int l = system.transform(possible_actions.get(j));
                if (best == -1 || scores.get(best) < scores.get(l))
                {
                    best = l;
                }
            }

            Action act = system.transform(best);
            system.transit(states[step], act, states[step + 1]);
        }

//        heads.resize(L);
//        deprels.resize(L);
        for (int i = 0; i < L; ++i)
        {
            heads.add(states[L * 2 - 1].heads.get(i));
            deprels.add(deprels_alphabet.labelOf(states[L * 2 - 1].deprels.get(i)));
        }
    }

    /**
     * 获取某个状态的上下文
     * @param s 状态
     * @param ctx 上下文
     */
    void get_context(final State s, Context ctx)
    {
        ctx.S0 = (s.stack.size() > 0 ? s.stack.get(s.stack.size() - 1) : -1);
        ctx.S1 = (s.stack.size() > 1 ? s.stack.get(s.stack.size() - 2) : -1);
        ctx.S2 = (s.stack.size() > 2 ? s.stack.get(s.stack.size() - 3) : -1);
        ctx.N0 = (s.buffer < s.ref.size() ? s.buffer : -1);
        ctx.N1 = (s.buffer + 1 < s.ref.size() ? s.buffer + 1 : -1);
        ctx.N2 = (s.buffer + 2 < s.ref.size() ? s.buffer + 2 : -1);

        ctx.S0L = (ctx.S0 >= 0 ? s.left_most_child.get(ctx.S0) : -1);
        ctx.S0R = (ctx.S0 >= 0 ? s.right_most_child.get(ctx.S0) : -1);
        ctx.S0L2 = (ctx.S0 >= 0 ? s.left_2nd_most_child.get(ctx.S0) : -1);
        ctx.S0R2 = (ctx.S0 >= 0 ? s.right_2nd_most_child.get(ctx.S0) : -1);
        ctx.S0LL = (ctx.S0L >= 0 ? s.left_most_child.get(ctx.S0L) : -1);
        ctx.S0RR = (ctx.S0R >= 0 ? s.right_most_child.get(ctx.S0R) : -1);

        ctx.S1L = (ctx.S1 >= 0 ? s.left_most_child.get(ctx.S1) : -1);
        ctx.S1R = (ctx.S1 >= 0 ? s.right_most_child.get(ctx.S1) : -1);
        ctx.S1L2 = (ctx.S1 >= 0 ? s.left_2nd_most_child.get(ctx.S1) : -1);
        ctx.S1R2 = (ctx.S1 >= 0 ? s.right_2nd_most_child.get(ctx.S1) : -1);
        ctx.S1LL = (ctx.S1L >= 0 ? s.left_most_child.get(ctx.S1L) : -1);
        ctx.S1RR = (ctx.S1R >= 0 ? s.right_most_child.get(ctx.S1R) : -1);
    }

    void get_features(final State s,
                      List<Integer> features)
    {
        Context ctx = new Context();
        get_context(s, ctx);
        get_basic_features(ctx, s.ref.forms, s.ref.postags, s.deprels, features);
        get_distance_features(ctx, features);
        get_valency_features(ctx, s.nr_left_children, s.nr_right_children, features);
    }

    /**
     * 生成特征
     * @param s 当前状态
     * @param cluster4
     * @param cluster6
     * @param cluster
     * @param features 输出特征
     */
    void get_features(final State s,
                      final List<Integer> cluster4,
                      final List<Integer> cluster6,
                      final List<Integer> cluster,
                      List<Integer> features)
    {
        Context ctx = new Context();
        get_context(s, ctx);
        get_basic_features(ctx, s.ref.forms, s.ref.postags, s.deprels, features);
        get_distance_features(ctx, features);
        get_valency_features(ctx, s.nr_left_children, s.nr_right_children, features);
        get_cluster_features(ctx, cluster4, cluster6, cluster, features);
    }

    /**
     * 获取单词
     * @param forms 单词列表
     * @param id 单词下标
     * @return 单词
     */
    int FORM(final List<Integer> forms, int id)
    {
        return ((id != -1) ? (forms.get(id)) : kNilForm);
    }

    /**
     * 获取词性
     * @param postags 词性列表
     * @param id 词性下标
     * @return 词性
     */
    int POSTAG(final List<Integer> postags, int id)
    {
        return ((id != -1) ? (postags.get(id) + kPostagInFeaturespace) : kNilPostag);
    }

    /**
     * 获取依存
     * @param deprels 依存列表
     * @param id 依存下标
     * @return 依存
     */
    int DEPREL(final List<Integer> deprels, int id)
    {
        return ((id != -1) ? (deprels.get(id) + kDeprelInFeaturespace) : kNilDeprel);
    }

    /**
     * 添加特征
     * @param features 输出特征的储存位置
     * @param feat 特征
     */
    void PUSH(List<Integer> features, int feat)
    {
        features.add(feat);
    }

    /**
     * 获取基本特征
     * @param ctx 上下文
     * @param forms 单词
     * @param postags 词性
     * @param deprels 依存
     * @param features 输出特征的储存位置
     */
    void get_basic_features(final Context ctx,
                            final List<Integer> forms,
                            final List<Integer> postags,
                            final List<Integer> deprels,
                            List<Integer> features)
    {
        PUSH(features, FORM(forms, ctx.S0));
        PUSH(features, POSTAG(postags, ctx.S0));
        PUSH(features, FORM(forms, ctx.S1));
        PUSH(features, POSTAG(postags, ctx.S1));
        PUSH(features, FORM(forms, ctx.S2));
        PUSH(features, POSTAG(postags, ctx.S2));
        PUSH(features, FORM(forms, ctx.N0));
        PUSH(features, POSTAG(postags, ctx.N0));
        PUSH(features, FORM(forms, ctx.N1));
        PUSH(features, POSTAG(postags, ctx.N1));
        PUSH(features, FORM(forms, ctx.N2));
        PUSH(features, POSTAG(postags, ctx.N2));
        PUSH(features, FORM(forms, ctx.S0L));
        PUSH(features, POSTAG(postags, ctx.S0L));
        PUSH(features, DEPREL(deprels, ctx.S0L));
        PUSH(features, FORM(forms, ctx.S0R));
        PUSH(features, POSTAG(postags, ctx.S0R));
        PUSH(features, DEPREL(deprels, ctx.S0R));
        PUSH(features, FORM(forms, ctx.S0L2));
        PUSH(features, POSTAG(postags, ctx.S0L2));
        PUSH(features, DEPREL(deprels, ctx.S0L2));
        PUSH(features, FORM(forms, ctx.S0R2));
        PUSH(features, POSTAG(postags, ctx.S0R2));
        PUSH(features, DEPREL(deprels, ctx.S0R2));
        PUSH(features, FORM(forms, ctx.S0LL));
        PUSH(features, POSTAG(postags, ctx.S0LL));
        PUSH(features, DEPREL(deprels, ctx.S0LL));
        PUSH(features, FORM(forms, ctx.S0RR));
        PUSH(features, POSTAG(postags, ctx.S0RR));
        PUSH(features, DEPREL(deprels, ctx.S0RR));
        PUSH(features, FORM(forms, ctx.S1L));
        PUSH(features, POSTAG(postags, ctx.S1L));
        PUSH(features, DEPREL(deprels, ctx.S1L));
        PUSH(features, FORM(forms, ctx.S1R));
        PUSH(features, POSTAG(postags, ctx.S1R));
        PUSH(features, DEPREL(deprels, ctx.S1R));
        PUSH(features, FORM(forms, ctx.S1L2));
        PUSH(features, POSTAG(postags, ctx.S1L2));
        PUSH(features, DEPREL(deprels, ctx.S1L2));
        PUSH(features, FORM(forms, ctx.S1R2));
        PUSH(features, POSTAG(postags, ctx.S1R2));
        PUSH(features, DEPREL(deprels, ctx.S1R2));
        PUSH(features, FORM(forms, ctx.S1LL));
        PUSH(features, POSTAG(postags, ctx.S1LL));
        PUSH(features, DEPREL(deprels, ctx.S1LL));
        PUSH(features, FORM(forms, ctx.S1RR));
        PUSH(features, POSTAG(postags, ctx.S1RR));
        PUSH(features, DEPREL(deprels, ctx.S1RR));
    }

    /**
     * 获取距离特征
     * @param ctx 当前特征
     * @param features 输出特征
     */
    void get_distance_features(final Context ctx,
                               List<Integer> features)
    {
        if (!use_distance)
        {
            return;
        }

        int dist = 8;
        if (ctx.S0 >= 0 && ctx.S1 >= 0)
        {
            dist = math.binned_1_2_3_4_5_6_10[ctx.S0 - ctx.S1];
            if (dist == 10)
            {
                dist = 7;
            }
        }
        features.add(dist + kDistanceInFeaturespace);
    }

    /**
     * 获取(S0和S1的)配价特征
     * @param ctx 上下文
     * @param nr_left_children 左孩子数量列表
     * @param nr_right_children 右孩子数量列表
     * @param features 输出特征
     */
    void get_valency_features(final Context ctx,
                              final List<Integer> nr_left_children,
                              final List<Integer> nr_right_children,
                              List<Integer> features)
    {
        if (!use_valency)
        {
            return;
        }

        int lvc = 8;
        int rvc = 8;
        if (ctx.S0 >= 0)
        {
            lvc = math.binned_1_2_3_4_5_6_10[nr_left_children.get(ctx.S0)];
            rvc = math.binned_1_2_3_4_5_6_10[nr_right_children.get(ctx.S0)];
            if (lvc == 10)
            {
                lvc = 7;
            }
            if (rvc == 10)
            {
                rvc = 7;
            }
        }
        features.add(lvc + kValencyInFeaturespace);
        features.add(rvc + kValencyInFeaturespace);

        lvc = 8;
        rvc = 8;
        if (ctx.S1 >= 0)
        {
            lvc = math.binned_1_2_3_4_5_6_10[nr_left_children.get(ctx.S1)];
            rvc = math.binned_1_2_3_4_5_6_10[nr_right_children.get(ctx.S1)];
            if (lvc == 10)
            {
                lvc = 7;
            }
            if (rvc == 10)
            {
                rvc = 7;
            }
        }
        features.add(lvc + kValencyInFeaturespace);
        features.add(rvc + kValencyInFeaturespace);
    }

    int CLUSTER(final List<Integer> cluster, int id)
    {
        return (id >= 0 ? (cluster.get(id) + kClusterInFeaturespace) : kNilCluster);
    }

    int CLUSTER4(final List<Integer> cluster4, int id)
    {
        return (id >= 0 ? (cluster4.get(id) + kCluster4InFeaturespace) : kNilCluster4);
    }

    int CLUSTER6(final List<Integer> cluster6, int id)
    {
        return (id >= 0 ? (cluster6.get(id) + kCluster6InFeaturespace) : kNilCluster6);
    }

    /**
     * 获取词聚类特征
     * @param ctx 上下文
     * @param cluster4
     * @param cluster6
     * @param cluster
     * @param features 输出特征
     */
    void get_cluster_features(final Context ctx,
                              final List<Integer> cluster4,
                              final List<Integer> cluster6,
                              final List<Integer> cluster,
                              List<Integer> features)
    {
        if (!use_cluster)
        {
            return;
        }

        PUSH(features, CLUSTER(cluster, ctx.S0));
        PUSH(features, CLUSTER4(cluster4, ctx.S0));
        PUSH(features, CLUSTER6(cluster6, ctx.S0));
        PUSH(features, CLUSTER(cluster, ctx.S1));
        PUSH(features, CLUSTER(cluster, ctx.S2));
        PUSH(features, CLUSTER(cluster, ctx.N0));
        PUSH(features, CLUSTER4(cluster4, ctx.N0));
        PUSH(features, CLUSTER6(cluster6, ctx.N0));
        PUSH(features, CLUSTER(cluster, ctx.N1));
        PUSH(features, CLUSTER(cluster, ctx.N2));
        PUSH(features, CLUSTER(cluster, ctx.S0L));
        PUSH(features, CLUSTER(cluster, ctx.S0R));
        PUSH(features, CLUSTER(cluster, ctx.S0L2));
        PUSH(features, CLUSTER(cluster, ctx.S0R2));
        PUSH(features, CLUSTER(cluster, ctx.S0LL));
        PUSH(features, CLUSTER(cluster, ctx.S0RR));
        PUSH(features, CLUSTER(cluster, ctx.S1L));
        PUSH(features, CLUSTER(cluster, ctx.S1R));
        PUSH(features, CLUSTER(cluster, ctx.S1L2));
        PUSH(features, CLUSTER(cluster, ctx.S1R2));
        PUSH(features, CLUSTER(cluster, ctx.S1LL));
        PUSH(features, CLUSTER(cluster, ctx.S1RR));
    }

}
