/**
 * 本package是对Yara Parser的包装与优化，主要做了如下几点优化
 * - 代码重构，提高复用率（由于dynamic oracle需要在训练的过程中逐渐动态地创建特征，
 *   所以无法复用HanLP的感知机框架，这也是为什么选择直接包装该模块而不是重新实现的原因之一。）
 * - 接口调整，与词法分析器整合
 * - debug
 * - 文档注释
 * Yara Parser的版权与授权信息如下：
 * © Copyright 2014-2015, Yahoo! Inc.
 * © Licensed under the terms of the Apache License 2.0.
 */
package com.hankcs.hanlp.dependency.perceptron;