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
 * 封装了模型用来计算概率的一些参数
 *
 * @author open-nlp
 */
public class EvalParameters
{

    /**
     * 将输出与参数映射起来，下标可以用 <code>pmap</code> 查询到
     */
    private Context[] params;
    /**
     * 一共有几种输出
     */
    private final int numOutcomes;
    /**
     * 一个事件中最多包含的特征数
     */
    private double correctionConstant;

    /**
     * correctionConstant的倒数
     */
    private final double constantInverse;
    /**
     * 修正参数
     */
    private double correctionParam;

    /**
     * 创建一个参数，可被用于预测
     *
     * @param params             环境
     * @param correctionParam    修正参数
     * @param correctionConstant 一个事件中最多包含的特征数
     * @param numOutcomes        事件的可能label数
     */
    public EvalParameters(Context[] params, double correctionParam, double correctionConstant, int numOutcomes)
    {
        this.params = params;
        this.correctionParam = correctionParam;
        this.numOutcomes = numOutcomes;
        this.correctionConstant = correctionConstant;
        this.constantInverse = 1.0 / correctionConstant;
    }

    public EvalParameters(Context[] params, int numOutcomes)
    {
        this(params, 0, 0, numOutcomes);
    }

    public Context[] getParams()
    {
        return params;
    }

    public int getNumOutcomes()
    {
        return numOutcomes;
    }

    public double getCorrectionConstant()
    {
        return correctionConstant;
    }

    public double getConstantInverse()
    {
        return constantInverse;
    }

    public double getCorrectionParam()
    {
        return correctionParam;
    }

    public void setCorrectionParam(double correctionParam)
    {
        this.correctionParam = correctionParam;
    }
}