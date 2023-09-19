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

# ctb

 See also [The Part-Of-Speech Tagging Guidelines for the Penn Chinese Treebank (3.0)](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1039&context=ircs_reports).

| Tag  | Description                                          | Chinese | Chinese Description                                                      | Examples              |
|-----|------------------------------------------------------|---------|---------------------------------------------------------|------------------------|
| AD  | adverb                                               | 副词      | 副词                                                      | 仍然、很、大大、约              |
| AS  | aspect marker                                        | 动态助词    | 助词                                                      | 了、着、过                  |
| BA  | `bǎ` in ba-construction                              | 把字句     | 当“把”、“将”出现在结构“NP0 + BA + NP1+VP”时的词性                    | 把、将                    |
| CC  | coordinating conjunction                             | 并列连接词   | 并列连词                                                    | 与、和、或者、还是              |
| CD  | cardinal number                                      | 概数词     | 数词或表达数量的词                                               | 一百、好些、若干               |
| CS  | subordinating conjunction                            | 从属连词    | 从属连词                                                    | 如果、那么、就                |
| DEC | `de` as a complementizer or a nominalizer            | 补语成分“的” | 当“的”或“之”作补语标记或名词化标记时的词性，其结构为：S/VP DEC {NP}，如，喜欢旅游的大学生   | 的、之                    |
| DEG | `de` as a genitive marker and an associative marker  | 属格“的”   | 当“的”或“之”作所有格时的词性，其结构为:NP/PP/JJ/DT DEG {NP}， 如，他的车、经济的发展 | 的、之                    |
| DER | resultative `de`, `de` in V-de const and V-de-R      | 表结果的“得” | 当“得”出现在结构“V-得-R”时的词性，如，他跑得很快                            | 得                      |
| DEV | manner `de`, `de` before VP                          | 表方式的“地” | 当“地”出现在结构“X-地-VP”时的词性，如，高兴地说                            | 地                      |
| DT  | determiner                                           | 限定词     | 代冠词，通常用来修饰名词                                            | 这、那、该、每、各              |
| ETC | for words like "etc."                                | 表示省略    | “等”、“等等”的词性                                             | 等、等等              |
| EM  | emoji                                                | 表情符     | 表情符、或称颜文字                                      | ：）             |
| FW  | foreign words                                        | 外来语     | 外来词                                                     | 卡拉、A型                  |
| IC  | incomplete component                                 | 不完整成分   | 不完整成分，尤指ASR导致的错误                         | 好*xin*、那个*ba*  |
| IJ  | interjection                                         | 句首感叹词   | 感叹词，通常出现在句子首部                                           | 啊                      |
| JJ  | other noun-modifier                                  | 其他名词修饰语 | 形容词                                                     | 共同、新                   |
| LB  | `bèi` in long bei-const                              | 长句式表被动  | 当“被”、“叫”、“给”出现在结构“NP0 + LB + NP1+ VP”结构时 的词性，如，他被我训了一顿  | 被、叫、给                  |
| LC  | localizer                                            | 方位词     | 方位词以及表示范围的限定词                                                     | 前、旁、到、在内、以来、为止               |
| M   | measure word                                         | 量词      | 量词                                                      | 个、群、公里                 |
| MSP | other particle                                       | 其他小品词   | 其他虚词，包括“所”、“以”、“来”和“而”等出现在VP前的词                         | 所、以、来、而                |
| NN  | common noun                                          | 其他名词    | 除专有名词和时间名词外的所有名词                                        | 桌子、生活、经济               |
| NOI | noise that characters are written in the wrong order | 噪声      | 汉字顺序颠倒产生的噪声                    | 事/NOI 类/NOI 各/NOI 故/NOI |
| NR  | proper noun                                          | 专有名词    | 专有名词，通常表示地名、人名、机构名等                                     | 北京、乔丹、微软               |
| NT  | temporal noun                                        | 时间名词    | 表示时间概念的名词                                               | 一月、汉朝、当今               |
| OD  | ordinal number                                       | 序数词     | 序列词                                                     | 第一百                    |
| ON  | onomatopoeia                                         | 象声词     | 象声词                                                     | 哗哗、呼、咯吱              |
| P   | preposition e.g., "from" and "to"                    | 介词      | 介词                                                      | 从、对、根据                 |
| PN  | pronoun                                              | 代词      | 代词，通常用来指代名词                                             | 我、这些、其、自己              |
| PU  | punctuation                                          | 标点符号    | 标点符号                                                    | ?、。、；                  |
| SB  | `bèi` in short bei-const                             | 短句式表被动  | 当“被”、“给”出现在NP0 +SB+ VP结果时的词性，如，他被训了 一顿                  | 被、叫                    |
| SP  | sentence final particle                              | 句末助词    | 经常出现在句尾的词                                               | 吧、呢、啊、啊                |
| URL | web address                                          | 网址      | 网址                                                      | www.hankcs.com         |
| VA  | predicative adjective                                | 表语形容词   | 可以接在“很”后面的形容词谓语                                         | 雪白、厉害                  |
| VC  | copula, be words                                     | 系动词     | 系动词，表示“是”或“非”概念的动词                                       | 是、为、非                  |
| VE  | `yǒu` as the main verb                               | 动词有无    | 表示“有”或“无”概念的动词                                          | 有、没有、无                 |
| VV  | other verb                                           | 其他动词    | 其他普通动词，包括情态词、控制动词、动作动词、心理动词等等                           | 可能、要、走、喜欢              |
