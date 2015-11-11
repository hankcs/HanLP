/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/11/1 20:04</create-date>
 *
 * <copyright file="NeuralNetworkClassifier.java" company="��ũ��">
 * Copyright (c) 2008-2015, ��ũ��. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dependency.nnparser;

import com.hankcs.hanlp.dependency.nnparser.option.LearnOption;

import java.util.List;
import java.util.Map;

import static com.hankcs.hanlp.dependency.nnparser.util.Log.ERROR_LOG;
import static com.hankcs.hanlp.dependency.nnparser.util.Log.INFO_LOG;

/**
 * 基于神经网络模型的分类器
 * @author hankcs
 */
public class NeuralNetworkClassifier
{
    /**
     * 输入层到隐藏层的权值矩阵
     */
    Matrix W1;
    /**
     * 隐藏层到输出层（softmax层）的权值矩阵
     */
    Matrix W2;
    /**
     * 向量化（词向量、词性向量、依存关系向量）
     */
    Matrix E;
    /**
     * 输入层的偏置
     */
    Matrix b1;
    /**
     * 预先计算的矩阵乘法结果
     */
    Matrix saved;

    // 以下是训练相关参数，暂未实现
    Matrix grad_W1;
    Matrix grad_W2;
    Matrix grad_E;
    Matrix grad_b1;
    Matrix grad_saved;

    Matrix eg2W1;
    Matrix eg2W2;
    Matrix eg2E;
    Matrix eg2b1;

    double loss;
    double accuracy;

    // Precomputed matrix
    // The configuration
    int embedding_size;      //! The size of the embedding.
    /**
     * 隐藏层节点数量
     */
    int hidden_layer_size;   //! The size of the hidden layer
    /**
     * 特征个数
     */
    int nr_objects;          //! The sum of forms, postags and deprels
    /**
     * 特征种数
     */
    int nr_feature_types;    //! The number of feature types
    /**
     * 分类个数
     */
    int nr_classes;          //! The number of classes

    int batch_size;
    int nr_threads;
    boolean fix_embeddings;

    double dropout_probability;
    double lambda;
    double ada_eps;
    double ada_alpha;

    /**
     * 将某个特征映射到预计算的矩阵的某一个列号
     */
    Map<Integer, Integer> precomputation_id_encoder;

    boolean initialized;

    void initialize(
            int _nr_objects,
            int _nr_classes,
            int _nr_feature_types,
            final LearnOption opt,
            final List<List<Double>> embeddings,
            final List<Integer> precomputed_features
    )
    {
        if (initialized)
        {
            ERROR_LOG("classifier: weight should not be initialized twice!");
            return;
        }

        batch_size = opt.batch_size;
        fix_embeddings = opt.fix_embeddings;
        dropout_probability = opt.dropout_probability;
        lambda = opt.lambda;
        ada_eps = opt.ada_eps;
        ada_alpha = opt.ada_alpha;

        // Initialize the parameter.
        nr_feature_types = _nr_feature_types;
        nr_objects = _nr_objects;
        nr_classes = _nr_classes; // nr_deprels*2+1-NIL

        embedding_size = opt.embedding_size;
        hidden_layer_size = opt.hidden_layer_size;

        // Initialize the network
        int nrows = hidden_layer_size;
        int ncols = embedding_size * nr_feature_types;
        W1 = Matrix.random(nrows, ncols).times(Math.sqrt(6. / (nrows + ncols)));
        b1 = Matrix.random(nrows, 1).times(Math.sqrt(6. / (nrows + ncols)));

        nrows = _nr_classes;  //
        ncols = hidden_layer_size;
        W2 = Matrix.random(nrows, ncols).times(Math.sqrt(6. / (nrows + ncols)));

        // Initialized the embedding
        nrows = embedding_size;
        ncols = _nr_objects;

        E = Matrix.random(nrows, ncols).times(opt.init_range);

        for (int i = 0; i < embeddings.size(); ++i)
        {
            final List<Double> embedding = embeddings.get(i);
            int id = embedding.get(0).intValue();
            for (int j = 1; j < embedding.size(); ++j)
            {
                E.set(j - 1, id, embedding.get(j));
            }
        }

        grad_W1 = Matrix.zero(W1.getRowDimension(), W1.getColumnDimension());
        grad_b1 = Matrix.zero(b1.rows(), 1);
        grad_W2 = Matrix.zero(W2.rows(), W2.cols());
        grad_E = Matrix.zero(E.rows(), E.cols());

        // Initialized the precomputed features
        Map<Integer, Integer> encoder = precomputation_id_encoder;
        int rank = 0;

        for (int i = 0; i < precomputed_features.size(); ++i)
        {
            int fid = precomputed_features.get(i);
            encoder.put(fid, rank++);
        }

        saved = Matrix.zero(hidden_layer_size, encoder.size());
        grad_saved = Matrix.zero(hidden_layer_size, encoder.size());

        //
        initialize_gradient_histories();

        initialized = true;

        info();
        INFO_LOG("classifier: size of batch = %d", batch_size);
        INFO_LOG("classifier: alpha = %e", ada_alpha);
        INFO_LOG("classifier: eps = %e", ada_eps);
        INFO_LOG("classifier: lambda = %e", lambda);
        INFO_LOG("classifier: fix embedding = %s", (fix_embeddings ? "true" : "false"));
    }

    void initialize_gradient_histories()
    {
        eg2W1 = Matrix.zero(W1.rows(), W1.cols());
        eg2b1 = Matrix.zero(b1.rows(), 1);
        eg2W2 = Matrix.zero(W2.rows(), W2.cols());
        eg2E = Matrix.zero(E.rows(), E.cols());
    }

    NeuralNetworkClassifier(
            Matrix _W1,
            Matrix _W2,
            Matrix _E,
            Matrix _b1,
            Matrix _saved,
            Map<Integer, Integer> encoder)

    {
        initialized = false;
        W1 = _W1;
        W2 = _W2;
        E = _E;
        b1 = _b1;
        saved = _saved;
        precomputation_id_encoder = encoder;
        embedding_size = 0;
        hidden_layer_size = 0;
        nr_objects = 0;
        nr_feature_types = 0;
        nr_classes = 0;
    }

    /**
     * 给每个类别打分
     * @param attributes 属性
     * @param retval 返回各个类别的得分
     */
    void score(final List<Integer> attributes,
               List<Double> retval)
    {
        Map <Integer,Integer >   encoder = precomputation_id_encoder;
        // arma.vec hidden_layer = arma.zeros<arma.vec>(hidden_layer_size);
        Matrix hidden_layer = Matrix.zero(hidden_layer_size, 1);

        for (int i = 0, off = 0; i < attributes.size(); ++i, off += embedding_size)
        {
            int aid = attributes.get(i);
            int fid = aid * nr_feature_types + i;
            Integer rep = encoder.get(fid);
            if (rep != null)
            {
                hidden_layer.plusEquals(saved.col(rep));
            }
            else
            {
                // 使用向量而不是特征本身
                // W1[0:hidden_layer, off:off+embedding_size] * E[fid:]'
                hidden_layer.plusEquals(W1.block(0, off, hidden_layer_size, embedding_size).times(E.col(aid)));
            }
        }

        hidden_layer.plusEquals(b1);    // 加上偏置

        Matrix output = W2.times(new Matrix (hidden_layer.cube())); // 立方激活函数
//        retval.resize(nr_classes, 0.);
        retval.clear();
        for (int i = 0; i < nr_classes; ++i)
        {
            retval.add(output.get(i, 0));
        }
    }

//    void compute_ada_gradient_step(
//            int begin,
//            int end)
//    {
//        if (!initialized)
//        {
//            ERROR_LOG("classifier: should not run the learning algorithm with un-initialized classifier.");
//            return;
//        }
//
//        // precomputing
//        Set<Integer> precomputed_features = new TreeSet<Integer>();
//        get_precomputed_features(begin, end, precomputed_features);
//        precomputing(precomputed_features);
//
//        // calculate gradient
//        /*grad_saved.zeros();*/
//        grad_saved.setZero();
//        compute_gradient(begin, end, end - begin);
//        compute_saved_gradient(precomputed_features);
//
//        // add regularizer.
//        add_l2_regularization();
//    }

//    void take_ada_gradient_step()
//    {
//        eg2W1 += Eigen.MatrixXd(grad_W1.array().square());
//        W1 -= ada_alpha * Eigen.MatrixXd(grad_W1.array() / (eg2W1.array() + ada_eps).sqrt());
//
//        eg2b1 += Matrix (grad_b1.array().square());
//        b1 -= ada_alpha * Matrix (grad_b1.array() / (eg2b1.array() + ada_eps).sqrt());
//
//        eg2W2 += Eigen.MatrixXd (grad_W2.array().square());
//        W2 -= ada_alpha * Eigen.MatrixXd (grad_W2.array() / (eg2W2.array() + ada_eps).sqrt());
//
//        if (!fix_embeddings)
//        {
//            eg2E += Eigen.MatrixXd (grad_E.array().square());
//            E -= ada_alpha * Eigen.MatrixXd (grad_E.array() / (eg2E.array() + ada_eps).sqrt());
//        }
//    }

    double get_cost()
    {
        return loss;
    }

    double get_accuracy()
    {
        return accuracy;
    }

//    void get_precomputed_features(
//            int begin,
//            int end,
//            Set<Integer> retval)
//    {
//        final std.unordered_map <int,int >   encoder = precomputation_id_encoder;
//        for (List < Sample >.final_iterator sample = begin;
//        sample != end;
//        ++sample){
//        for (int i = 0; i < sample -> attributes.size(); ++i)
//        {
//            int fid = sample -> attributes[i] * nr_feature_types + i;
//            if (encoder.find(fid) != encoder.end())
//            {
//                retval.insert(fid);
//            }
//        }
//    }
//        // INFO_LOG("classifier: percentage of necessary precomputation: %lf%%",
//        // (double)retval.size() / encoder.size() * 100);
//    }

//    void precomputing(Set<Integer> precomputed_features)
//    {
//        final std.unordered_map <int,int >   encoder = precomputation_id_encoder;
//        std.unordered_set <int>features;
//        for (std.unordered_map <int,int >.final_iterator rep = encoder.begin();
//        rep != encoder.end();
//        ++rep){
//        features.insert(rep -> first);
//    }
//        precomputing(features);
//    }

//    void precomputing(
//            final std.unordered_set<int> features)
//    {
//        saved.setZero();
//        for (std.unordered_set <int>.final_iterator rep = features.begin();
//        rep != features.end();
//        ++rep){
//        int fid = ( * rep);
//        int rank = precomputation_id_encoder[fid];
//        int aid = fid / nr_feature_types;
//        int off = (fid % nr_feature_types) * embedding_size;
//        saved.col(rank) = W1.block(0, off, hidden_layer_size, embedding_size) * E.col(aid);
//    }
//        // INFO_LOG("classifier: precomputed %d", features.size());
//    }

//    void compute_gradient(
//            int begin,
//            int end,
//            int batch_size)
//    {
//        final std.unordered_map <int,int >   encoder = precomputation_id_encoder;
//
//        grad_W1.setZero();
//        grad_b1.setZero();
//        grad_W2.setZero();
//        grad_E.setZero();
//
//        loss = 0;
//        accuracy = 0;
//
//        // special for Eigen.XXX.Random
//        double mask_prob = dropout_probability * 2 - 1;
//        for (List < Sample >.final_iterator sample = begin;
//        sample != end;
//        ++sample){
//        final List <int> attributes = sample -> attributes;
//        final List <double> classes = sample -> classes;
//
//        Matrix Y = Matrix.Map(   classes[0], classes.size());
//        Matrix _ = (Eigen.ArrayXd.Random (hidden_layer_size) > mask_prob).select(
//                Matrix.Ones (hidden_layer_size),
//                Matrix.zero(hidden_layer_size));
//        Matrix hidden_layer = Matrix.zero(hidden_layer_size);
//
//        for (int i = 0, off = 0; i < attributes.size(); ++i, off += embedding_size)
//        {
//            int aid = attributes[i];
//            int fid = aid * nr_feature_types + i;
//            std.unordered_map <int,int >.final_iterator rep = encoder.find(fid);
//            if (rep != encoder.end())
//            {
//                hidden_layer += _.asDiagonal() * saved.col(rep -> second);
//            }
//            else
//            {
//                hidden_layer +=
//                        _.asDiagonal() * W1.block(0, off, hidden_layer_size, embedding_size) * E.col(aid);
//            }
//        }
//
//        hidden_layer += _.asDiagonal() * b1;
//
//        Matrix cubic_hidden_layer = hidden_layer.array().cube().min(50).max(-50);
//        Matrix output = W2 * cubic_hidden_layer;
//
//        int opt_class = -1, correct_class = -1;
//        for (int i = 0; i < nr_classes; ++i)
//        {
//            if (classes[i] >= 0    (opt_class < 0 || output(i) > output(opt_class)))
//            {
//                opt_class = i;
//            }
//            if (classes[i] == 1)
//            {
//                correct_class = i;
//            }
//        }
//
//    /*arma.uvec classes_mask = arma.find(Y >= 0);*/
//        Matrix __ = (Y.array() >= 0).select(
//                Matrix.Ones (hidden_layer_size),
//                Matrix.zero(hidden_layer_size));
//        double best = output(opt_class);
//        output = __.asDiagonal() * Matrix ((output.array() - best).exp());
//        double sum1 = output(correct_class);
//        double sum2 = output.sum();
//
//        loss += (log(sum2) - log(sum1));
//        if (classes[opt_class] == 1)
//        {
//            accuracy += 1;
//        }
//
//        Matrix delta =
//                -(__.asDiagonal() * Y - Matrix (output.array() / sum2))/batch_size;
//
//        grad_W2 += delta * cubic_hidden_layer.transpose();
//        Matrix grad_cubic_hidden_layer = _.asDiagonal() * W2.transpose() * delta;
//
//        Matrix grad_hidden_layer =
//                3 * grad_cubic_hidden_layer.array() * hidden_layer.array().square();
//
//        grad_b1 += grad_hidden_layer;
//
//        for (int i = 0, off = 0; i < attributes.size(); ++i, off += embedding_size)
//        {
//            int aid = attributes[i];
//            int fid = aid * nr_feature_types + i;
//            std.unordered_map <int,int >.final_iterator rep = encoder.find(fid);
//            if (rep != encoder.end())
//            {
//                grad_saved.col(rep -> second) += grad_hidden_layer;
//            }
//            else
//            {
//                grad_W1.block(0, off, hidden_layer_size, embedding_size) +=
//                        grad_hidden_layer * E.col(aid).transpose();
//                if (!fix_embeddings)
//                {
//                    grad_E.col(aid) +=
//                            W1.block(0, off, hidden_layer_size, embedding_size).transpose() * grad_hidden_layer;
//                }
//            }
//        }
//    }
//
//        loss /= batch_size;
//        accuracy /= batch_size;
//    }

//    void compute_saved_gradient(
//            final Set<Integer> features)
//    {
//        std.unordered_map <int,int >   encoder = precomputation_id_encoder;
//        for (std.unordered_set <int>.final_iterator rep = features.begin();
//        rep != features.end();
//        ++rep){
//        int fid = ( * rep);
//        int rank = encoder[fid];
//        int aid = fid / nr_feature_types;
//        int off = (fid % nr_feature_types) * embedding_size;
//
//        grad_W1.block(0, off, hidden_layer_size, embedding_size) +=
//                grad_saved.col(rank) * E.col(aid).transpose();
//
//        if (!fix_embeddings)
//        {
//            grad_E.col(aid) +=
//                    W1.block(0, off, hidden_layer_size, embedding_size).transpose() * grad_saved.col(rank);
//        }
//    }
//    }

//    void add_l2_regularization()
//    {
//        loss += lambda * .5 * (W1.squaredNorm() + b1.squaredNorm() + W2.squaredNorm());
//        if (!fix_embeddings)
//        {
//            loss += lambda * .5 * E.squaredNorm();
//        }
//
//        grad_W1 += lambda * W1;
//        grad_b1 += lambda * b1;
//        grad_W2 += lambda * W2;
//        if (!fix_embeddings)
//        {
//            grad_E += lambda * E;
//        }
//    }

    /**
     * 初始化参数
     */
    void canonical()
    {
        hidden_layer_size = b1.rows();
        nr_feature_types = W1.cols() / E.rows();
        nr_classes = W2.rows();
        embedding_size = E.rows();
    }

    void info()
    {
        INFO_LOG("classifier: E(%d,%d)", E.rows(), E.cols());
        INFO_LOG("classifier: W1(%d,%d)", W1.rows(), W1.cols());
        INFO_LOG("classifier: b1(%d)", b1.rows());
        INFO_LOG("classifier: W2(%d,%d)", W2.rows(), W2.cols());
        INFO_LOG("classifier: saved(%d,%d)", saved.rows(), saved.cols());
        INFO_LOG("classifier: precomputed size=%d", precomputation_id_encoder.size());
        INFO_LOG("classifier: hidden layer size=%d", hidden_layer_size);
        INFO_LOG("classifier: embedding size=%d", embedding_size);
        INFO_LOG("classifier: number of classes=%d", nr_classes);
        INFO_LOG("classifier: number of feature types=%d", nr_feature_types);
    }
}
