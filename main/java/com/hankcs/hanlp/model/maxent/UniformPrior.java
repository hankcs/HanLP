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
 * 先验概率计算工具
 */
public class UniformPrior
{
    private int numOutcomes;
    private double r;

    /**
     * 获取先验概率
     * @param dist 储存位置
     */
    public void logPrior(double[] dist)
    {
        for (int oi = 0; oi < numOutcomes; oi++)
        {
            dist[oi] = r;
        }
    }

    /**
     * 初始化
     * @param outcomeLabels
     */
    public void setLabels(String[] outcomeLabels)
    {
        this.numOutcomes = outcomeLabels.length;
        r = Math.log(1.0 / numOutcomes);
    }
}
