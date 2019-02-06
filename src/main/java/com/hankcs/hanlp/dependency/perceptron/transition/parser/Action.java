package com.hankcs.hanlp.dependency.perceptron.transition.parser;

import com.hankcs.hanlp.dependency.perceptron.transition.configuration.Configuration;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 12/23/14
 * Time: 11:08 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public enum Action implements IAction
{
    Shift
            {
                @Override
                public void commit(int relation, float score, int relationSize, Configuration config)
                {
                    ArcEager.shift(config.state);
                    config.addAction(ordinal());
                    config.setScore(score);
                }
            },
    Reduce
            {
                @Override
                public void commit(int relation, float score, int relationSize, Configuration config)
                {
                    ArcEager.reduce(config.state);
                    config.addAction(ordinal());
                    config.setScore(score);
                }
            },
    Unshift
            {
                @Override
                public void commit(int relation, float score, int relationSize, Configuration config)
                {
                    ArcEager.unShift(config.state);
                    config.addAction(ordinal());
                    config.setScore(score);
                }
            },
    RightArc
            {
                @Override
                public void commit(int relation, float score, int relationSize, Configuration config)
                {
                    ArcEager.rightArc(config.state, relation);
                    config.addAction(ordinal() + relation);
                    config.setScore(score);
                }
            },
    LeftArc
            {
                @Override
                public void commit(int relation, float score, int relationSize, Configuration config)
                {
                    ArcEager.leftArc(config.state, relation);
                    config.addAction(ordinal() + relationSize + relation);
                    config.setScore(score);
                }
            }
}