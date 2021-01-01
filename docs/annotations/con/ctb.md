<!--
# ========================================================================
# Copyright 2020 hankcs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ========================================================================
-->

# Chinese Tree Bank

See also [The Bracketing Guidelines for the Penn Chinese Treebank (3.0)](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1040&context=ircs_reports).

| Tag | Definition                             | 定义                                     | 例子            |
|------|----------------------------------------------|----------------------------------------------------|-------------------|
| ADJP | adjective phrase                             | 形容词短语，以形容词为中心词                                     | 不完全、大型            |
| ADVP | adverbial phrase headed by AD (adverb)       | 副词短语，以副词为中心词                                       | 非常、很              |
| CLP  | classifier phrase                            | 由量词构成的短语                                           | 系列、大批             |
| CP   | clause headed by C (complementizer)          | 从句，通过带补语（如“的”、“吗”等）                                | 张三喜欢李四吗？          |
| DNP  | phrase formed by ‘‘XP + DEG’’                | 结构为XP + DEG(的)的短语，其中XP可以是ADJP、DP、QP、PP等等，用于修饰名词短语。 | 大型的、前几年的、五年的、在上海的 |
| DP   | determiner phrase                            | 限定词短语，通常由限定词和数量词构成                                 | 这三个、任何            |
| DVP  | phrase formed by ‘‘XP + DEV’’                | 结构为XP+地的短评，用于修饰动词短语VP                              | 心情失落地、大批地         |
| FRAG | fragment                                     | 片段                                                 | (完）               |
| INTJ | interjection                                 | 插话，感叹语                                             | 哈哈、切              |
| IP   | simple clause headed by I (INFL)             | 简单子句或句子，通常不带补语（如“的”、“吗”等）                          | 张三喜欢李四。           |
| LCP  | phrase formed by ‘‘XP + LC’’                 | 用于表本地点+方位词（LC)的短语                                  | 生活中、田野上           |
| LST  | list marker                                  | 列表短语，包括标点符号                                        | 一.                |
| MSP  | some particles                               | 其他小品词                                              | 所、而、来、去           |
| NN   | common noun                                  | 名词                                                 | HanLP、技术          |
| NP   | noun phrase                                  | 名词短语，中心词通常为名词                                      | 美好生活、经济水平         |
| PP   | preposition phrase                           | 介词短语，中心词通常为介词                                      | 在北京、据报道           |
| PRN  | parenthetical                                | 插入语                                                | ，（张三说)，           |
| QP   | quantifier phrase                            | 量词短语                                               | 三个、五百辆            |
| ROOT | root node                                    | 根节点                                                | 根节点               |
| UCP  | unidentical coordination phrase              | 不对称的并列短语，指并列词两侧的短语类型不致                             | (养老、医疗）保险         |
| VCD  | coordinated verb compound                    | 复合动词                                               | 出版发行              |
| VCP  | verb compounds formed by VV + VC             | VV + VC形式的动词短语                                     | 看作是               |
| VNV  | verb compounds formed by A-not-A or A-one-A  | V不V形式的动词短语                                         | 能不能、信不信           |
| VP   | verb phrase                                  | 动词短语，中心词通常为动词                                      | 完成任务、努力工作         |
| VPT  | potential form V-de-R or V-bu-R              | V不R、V得R形式的动词短语                                     | 打不赢、打得过           |
| VRD  | verb resultative compound                    | 动补结构短语                                             | 研制成功、降下来          |
| VSB  | verb compounds formed by a modifier + a head | 修饰语+中心词构成的动词短语                                     | 拿来支付、仰头望去         |