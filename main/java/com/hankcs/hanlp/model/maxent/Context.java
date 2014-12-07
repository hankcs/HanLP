/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.hankcs.hanlp.model.maxent;

/**
 * 将参数与特征关联起来的类，用来储存最大熵的参数，也用来储存模型和经验分布
 */
public class Context
{

    /**
     * 参数
     */
    protected double[] parameters;
    /**
     * 输出（标签）
     */
    protected int[] outcomes;

    /**
     * 构建一个新的上下文
     *
     * @param outcomePattern 输出
     * @param parameters     参数
     */
    public Context(int[] outcomePattern, double[] parameters)
    {
        this.outcomes = outcomePattern;
        this.parameters = parameters;
    }

    /**
     * 获取输出
     *
     * @return 输出数组
     */
    public int[] getOutcomes()
    {
        return outcomes;
    }

    /**
     * 获取参数
     *
     * @return 参数数组
     */
    public double[] getParameters()
    {
        return parameters;
    }
}
