/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-08-12 7:11 PM</create-date>
 *
 * <copyright file="Cluster.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.hanlp.mining.cluster;

import java.util.*;

/**
 * @author hankcs
 */
public class Cluster<K> implements Comparable<Cluster<K>>
{
    List<Document<K>> documents_;          ///< documents
    SparseVector composite_;                           ///< a composite SparseVector
    SparseVector centroid_;                            ///< a centroid SparseVector
    List<Cluster<K>> sectioned_clusters_;  ///< sectioned clusters
    double sectioned_gain_;                      ///< a sectioned gain
    Random random;

    public Cluster()
    {
        this(new ArrayList<Document<K>>());
    }

    public Cluster(List<Document<K>> documents)
    {
        this.documents_ = documents;
        composite_ = new SparseVector();
        random = new Random();
    }

    /**
     * Add the vectors of all documents to a composite vector.
     */
    void set_composite_vector()
    {
        composite_.clear();
        for (Document<K> document : documents_)
        {
            composite_.add_vector(document.feature());
        }
    }

    /**
     * Clear status.
     */
    void clear()
    {
        documents_.clear();
        composite_.clear();
        if (centroid_ != null)
            centroid_.clear();
        if (sectioned_clusters_ != null)
            sectioned_clusters_.clear();
        sectioned_gain_ = 0.0;
    }


    /**
     * Get the size.
     *
     * @return the size of this cluster
     */
    int size()
    {
        return documents_.size();
    }

    /**
     * Get the pointer of a centroid vector.
     *
     * @return the pointer of a centroid vector
     */
    SparseVector centroid_vector()
    {
        if (documents_.size() > 0 && composite_.size() == 0)
            set_composite_vector();
        centroid_ = (SparseVector) composite_vector().clone();
        centroid_.normalize();
        return centroid_;
    }

    /**
     * Get the pointer of a composite vector.
     *
     * @return the pointer of a composite vector
     */
    SparseVector composite_vector()
    {
        return composite_;
    }

    /**
     * Get documents in this cluster.
     *
     * @return documents in this cluster
     */
    List<Document<K>> documents()
    {
        return documents_;
    }

    /**
     * Add a document.
     *
     * @param doc the pointer of a document object
     */
    void add_document(Document doc)
    {
        doc.feature().normalize();
        documents_.add(doc);
        composite_.add_vector(doc.feature());
    }

    /**
     * Remove a document from this cluster.
     *
     * @param index the index of vector container of documents
     */
    void remove_document(int index)
    {
        ListIterator<Document<K>> listIterator = documents_.listIterator(index);
        Document<K> document = listIterator.next();
        listIterator.set(null);
        composite_.sub_vector(document.feature());
    }

    /**
     * Delete removed documents from the internal container.
     */
    void refresh()
    {
        ListIterator<Document<K>> listIterator = documents_.listIterator();
        while (listIterator.hasNext())
        {
            if (listIterator.next() == null)
                listIterator.remove();
        }
    }

    /**
     * Get a gain when this cluster sectioned.
     *
     * @return a gain
     */
    double sectioned_gain()
    {
        return sectioned_gain_;
    }

    /**
     * Set a gain when the cluster sectioned.
     */
    void set_sectioned_gain()
    {
        double gain = 0.0f;
        if (sectioned_gain_ == 0 && sectioned_clusters_.size() > 1)
        {
            for (Cluster<K> cluster : sectioned_clusters_)
            {
                gain += cluster.composite_vector().norm();
            }
            gain -= composite_.norm();
        }
        sectioned_gain_ = gain;
    }

    /**
     * Get sectioned clusters.
     *
     * @return sectioned clusters
     */
    List<Cluster<K>> sectioned_clusters()
    {
        return sectioned_clusters_;
    }

//    /**
//     * Choose documents randomly.
//     */
//    void choose_randomly(int ndocs, List<Document > docs)
//{
//    HashMap<int, bool>.type choosed;
//    int siz = size();
//    init_hash_map(siz, choosed, ndocs);
//    if (siz < ndocs)
//        ndocs = siz;
//    int count = 0;
//    while (count < ndocs)
//    {
//        int index = myrand(seed_) % siz;
//        if (choosed.find(index) == choosed.end())
//        {
//            choosed.insert(std.pair<int, bool>(index, true));
//            docs.push_back(documents_[index]);
//            ++count;
//        }
//    }
//}

    /**
     * 选取初始质心
     *
     * @param ndocs 质心数量
     * @param docs  输出到该列表中
     */
    void choose_smartly(int ndocs, List<Document> docs)
    {
        int siz = size();
        double[] closest = new double[siz];
        if (siz < ndocs)
            ndocs = siz;
        int index, count = 0;

        index = random.nextInt(siz);  // initial center
        docs.add(documents_.get(index));
        ++count;
        double potential = 0.0;
        for (int i = 0; i < documents_.size(); i++)
        {
            double dist = 1.0 - SparseVector.inner_product(documents_.get(i).feature(), documents_.get(index).feature());
            potential += dist;
            closest[i] = dist;
        }

        // choose each center
        while (count < ndocs)
        {
            double randval = random.nextDouble() * potential;

            for (index = 0; index < documents_.size(); index++)
            {
                double dist = closest[index];
                if (randval <= dist)
                    break;
                randval -= dist;
            }
            if (index == documents_.size())
                index--;
            docs.add(documents_.get(index));
            ++count;

            double new_potential = 0.0;
            for (int i = 0; i < documents_.size(); i++)
            {
                double dist = 1.0 - SparseVector.inner_product(documents_.get(i).feature(), documents_.get(index).feature());
                double min = closest[i];
                if (dist < min)
                {
                    closest[i] = dist;
                    min = dist;
                }
                new_potential += min;
            }
            potential = new_potential;
        }
    }

    /**
     * 将本簇划分为nclusters个簇
     *
     * @param nclusters
     */
    void section(int nclusters)
    {
        if (size() < nclusters)
            throw new IllegalArgumentException("簇数目小于文档数目");

        sectioned_clusters_ = new ArrayList<Cluster<K>>(nclusters);
        List<Document> centroids = new ArrayList<Document>(nclusters);
        // choose_randomly(nclusters, centroids);
        choose_smartly(nclusters, centroids);
        for (int i = 0; i < centroids.size(); i++)
        {
            Cluster<K> cluster = new Cluster<K>();
            sectioned_clusters_.add(cluster);
        }

        for (Document<K> d : documents_)
        {
            double max_similarity = -1.0;
            int max_index = 0;
            for (int j = 0; j < centroids.size(); j++)
            {
                double similarity = SparseVector.inner_product(d.feature(), centroids.get(j).feature());
                if (max_similarity < similarity)
                {
                    max_similarity = similarity;
                    max_index = j;
                }
            }
            sectioned_clusters_.get(max_index).add_document(d);
        }
    }

    @Override
    public int compareTo(Cluster<K> o)
    {
        return Double.compare(o.sectioned_gain(), sectioned_gain());
    }
}
