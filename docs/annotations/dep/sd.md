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

# Stanford Dependencies

## Chinese

```{eval-rst}

See also :cite:`chang-etal-2009-discriminative`.
    
```

|Tag|Description|中文简称|例句|依存弧|
| ---- | ---- | ---- | ---- | ---- |
|nn|noun compound modifier|复合名词修饰|服务中心|nn(中心，服务）|
|punct|punctuation|标点符号|海关统计表明，|punct(表明，，)|
|nsubj|nominal subject|名词性主语|梅花盛开|nsubj (盛开，梅花）|
|conj|conjunct (links two conjuncts)|连接性状语|设备和原材料|conj(原材料，设备）|
|dobj|direct object|直接宾语|浦东颁布了七十一件文件|dobj(颁布，文件）|
|advmod|adverbial modifier|副词性状语|部门先送上文件|advmod(送上，先）|
|prep|prepositional modifier|介词性修饰语|在实践中逐步完善|prep(完善，在）|
|nummod|number modifier|数词修饰语|七十一件文件|nummod(件，七十一）|
|amod|adjectival modifier|形容词修饰语|跨世纪工程|amod(工程，跨世纪）|
|pobj|prepositional object|介词性宾语|根据有关规定|pobj (根据，规定）|
|rcmod|relative clause modifier|关系从句修饰语|不曾遇到过的情况|rcmod(情况，遇到）|
|cpm|complementizer|补语|开发浦东的经济活动|cpm(开发，的）|
|assm|associative marker|关联标记|企业的商品|assm(企业，的）|
|assmod|associative modifier|关联修饰|企业的商品|assmod(商品，企业）|
|cc|coordinating conjunction|并列关系|设备和原材料|cc(原材料，和）|
|clf|classifier modifier|类别修饰|七十一件文件|clf(文件，件）|
|ccomp|clausal complement|从句补充|银行决定先取得信用评级|ccomp(决定，取得）|
|det|determiner|限定语|这些经济活动|det(活动，这些）|
|lobj|localizer object|范围宾语|近年来|lobj(来，近年）|
|range|dative object that is a quantifier phrase|数量词间接宾语|成交药品一亿多元|range(成交，元）|
|asp|aspect marker|时态标记|发挥了作用|asp(发挥，了）|
|tmod|temporal modifier|时间修饰语|以前不曾遇到过|tmod(遇到，以前）|
|plmod|localizer modifier of a preposition|介词性地点修饰|在这片热土上|plmod(在，上）|
|attr|attributive|属性|贸易额为二百亿美元|attr(为，美元）|
|mmod|modal verb modifier|情态动词|利益能得到保障|mmod(得到，能）|
|loc|localizer|位置补语|占九成以上|loc(占，以上）|
|top|topic|主题|建筑是主要活动|top(是，建筑）|
|pccomp|clausal complement of a preposition|介词补语|据有关部门介绍|pccomp(据，介绍）|
|etc|etc modifier|省略关系|科技、文教等领域|etc(文教，等）|
|lccomp|clausal complement of a localizer|位置补语|中国对外开放中升起的明星|lccomp(中，开放）|
|ordmod|ordinal number modifier|量词修饰|第七个机构|ordmod(个，第七）|
|xsubj|controlling subject|控制主语|银行决定先取得信用评级|xsubj (取得，银行）|
|neg|negative modifier|否定修饰|以前不曾遇到过|neg(遇到，不）|
|rcomp|resultative complement|结果补语|研究成功|rcomp(研究，成功）|
|comod|coordinated verb compound modifier|并列联合动词|颁布实行|comod(颁布，实行）|
|vmod|verb modifier|动词修饰|其在支持外商企业方面的作用|vmod(方面，支持）|
|prtmod|particles such as 所，以，来，而|小品词|在产业化所取得的成就|prtmod(取得，所）|
|ba|“ba” construction|把字关系|把注意力转向市场|ba(转向，把）|
|dvpm|manner DE(地）modifier|地字修饰|有效地防止流失|dvpm(有效，地）|
|dvpmod|a "XP+DEV", phrase that modifies VP|地字动词短语|有效地防止流失|dvpmod(防止，有效）|
|prnmod|parenthetical modifier|插入词修饰|八五期间（1990-1995 )|pmmod(期间，1995)|
|cop|copular|系动词|原是自给自足的经济|cop(自给自足，是）|
|pass|passive marker|被动标记|被认定为高技术产业|pass(认定，被）|
|nsubjpass|nominal passive subject|被动名词主语|镍被称作现代工业的维生素|nsubjpass(称作，镍）|
|dep|dependent|其他依赖关系|新华社北京二月十二日电|dep(电，新华社）|

## English

See also [Stanford typed dependencies manual](https://nlp.stanford.edu/software/dependencies_manual.pdf).

| Tag        | Description                       |
|------------|-----------------------------------|
| abbrev     | abbreviation modifier             |
| acomp      | adjectival complement             |
| advcl      | adverbial clause modifier         |
| advmod     | adverbial modifier                |
| agent      | agent                             |
| amod       | adjectival modifier               |
| appos      | appositional modifier             |
| arg        | argument                          |
| attr       | attributive                       |
| aux        | auxiliary                         |
| auxpass    | passive auxiliary                 |
| cc         | coordination                      |
| ccomp      | clausal complement                |
| comp       | complement                        |
| complm     | complementizer                    |
| conj       | conjunct                          |
| cop        | copula                            |
| csubj      | clausal subject                   |
| csubjpass  | clausal passive subject           |
| dep        | dependent                         |
| det        | determiner                        |
| discourse  | discourse element                 |
| dobj       | direct object                     |
| expl       | expletive                         |
| goeswith   | goes with                         |
| iobj       | indirect object                   |
| mark       | marker                            |
| mod        | modifier                          |
| mwe        | multi-word expression             |
| neg        | negation modifier                 |
| nn         | noun compound modifier            |
| npadvmod   | noun phrase as adverbial modifier |
| nsubj      | nominal subject                   |
| nsubjpass  | passive nominal subject           |
| num        | numeric modifier                  |
| number     | element of compound number        |
| obj        | object                            |
| parataxis  | parataxis                         |
| pcomp      | prepositional complement          |
| pobj       | object of a preposition           |
| poss       | possession modifier               |
| possessive | possessive modifier               |
| preconj    | preconjunct                       |
| pred       | predicate                         |
| predet     | predeterminer                     |
| prep       | prepositional modifier            |
| prepc      | prepositional clausal modifier    |
| prt        | phrasal verb particle             |
| punct      | punctuation                       |
| purpcl     | purpose clause modifier           |
| quantmod   | quantifier phrase modifier        |
| rcmod      | relative clause modifier          |
| ref        | referent                          |
| rel        | relative                          |
| root       | root                              |
| sdep       | semantic dependent                |
| subj       | subject                           |
| tmod       | temporal modifier                 |
| vmod       | verb modifier                     |
| xcomp      | open clausal complement           |
| xsubj      | controlling subject               |