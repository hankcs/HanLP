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

# SemEval2016

See also [SemEval-2016 Task 9](https://www.hankcs.com/nlp/sdp-corpus.html) and [CSDP](https://csdp-doc.readthedocs.io/zh_CN/latest/%E9%99%84%E5%BD%95/).

| 关系类型   | Tag           | Description        | Example                     |
|--------|---------------|--------------------|-----------------------------|
| 施事关系   | Agt           | Agent              | 我送她一束花 (我 <– 送)             |
| 当事关系   | Exp           | Experiencer        | 我跑得快 (跑 –> 我)               |
| 感事关系   | Aft           | Affection          | 我思念家乡 (思念 –> 我)             |
| 领事关系   | Poss          | Possessor          | 他有一本好读 (他 <– 有)             |
| 受事关系   | Pat           | Patient            | 他打了小明 (打 –> 小明)             |
| 客事关系   | Cont          | Content            | 他听到鞭炮声 (听 –> 鞭炮声)           |
| 成事关系   | Prod          | Product            | 他写了本小说 (写 –> 小说)            |
| 源事关系   | Orig          | Origin             | 我军缴获敌人四辆坦克 (缴获 –> 坦克)       |
| 涉事关系   | Datv          | Dative             | 他告诉我个秘密 ( 告诉 –> 我 )         |
| 比较角色   | Comp          | Comitative         | 他成绩比我好 (他 –> 我)             |
| 属事角色   | Belg          | Belongings         | 老赵有俩女儿 (老赵 <– 有)            |
| 类事角色   | Clas          | Classification     | 他是中学生 (是 –> 中学生)            |
| 依据角色   | Accd          | According          | 本庭依法宣判 (依法 <– 宣判)           |
| 缘故角色   | Reas          | Reason             | 他在愁女儿婚事 (愁 –> 婚事)           |
| 意图角色   | Int           | Intention          | 为了金牌他拼命努力 (金牌 <– 努力)        |
| 结局角色   | Cons          | Consequence        | 他跑了满头大汗 (跑 –> 满头大汗)         |
| 方式角色   | Mann          | Manner             | 球慢慢滚进空门 (慢慢 <– 滚)           |
| 工具角色   | Tool          | Tool               | 她用砂锅熬粥 (砂锅 <– 熬粥)           |
| 材料角色   | Malt          | Material           | 她用小米熬粥 (小米 <– 熬粥)           |
| 时间角色   | Time          | Time               | 唐朝有个李白 (唐朝 <– 有)            |
| 空间角色   | Loc           | Location           | 这房子朝南 (朝 –> 南)              |
| 历程角色   | Proc          | Process            | 火车正在过长江大桥 (过 –> 大桥)         |
| 趋向角色   | Dir           | Direction          | 部队奔向南方 (奔 –> 南)             |
| 范围角色   | Sco           | Scope              | 产品应该比质量 (比 –> 质量)           |
| 数量角色   | Quan          | Quantity           | 一年有365天 (有 –> 天)            |
| 数量数组   | Qp            | Quantity-phrase    | 三本书 (三 –> 本)                |
| 频率角色   | Freq          | Frequency          | 他每天看书 (每天 <– 看)             |
| 顺序角色   | Seq           | Sequence           | 他跑第一 (跑 –> 第一)              |
| 描写角色   | Desc(Feat)    | Description        | 他长得胖 (长 –> 胖)               |
| 宿主角色   | Host          | Host               | 住房面积 (住房 <– 面积)             |
| 名字修饰角色 | Nmod          | Name-modifier      | 果戈里大街 (果戈里 <– 大街)           |
| 时间修饰角色 | Tmod          | Time-modifier      | 星期一上午 (星期一 <– 上午)           |
| 反角色    | r + main role |                    | 打篮球的小姑娘 (打篮球 <– 姑娘)         |
| 嵌套角色   | d + main role |                    | 爷爷看见孙子在跑 (看见 –> 跑)          |
| 并列关系   | eCoo          | event Coordination | 我喜欢唱歌和跳舞 (唱歌 –> 跳舞)         |
| 选择关系   | eSelt         | event Selection    | 您是喝茶还是喝咖啡 (茶 –> 咖啡)         |
| 等同关系   | eEqu          | event Equivalent   | 他们三个人一起走 (他们 –> 三个人)        |
| 先行关系   | ePrec         | event Precedent    | 首先，先                        |
| 顺承关系   | eSucc         | event Successor    | 随后，然后                       |
| 递进关系   | eProg         | event Progression  | 况且，并且                       |
| 转折关系   | eAdvt         | event adversative  | 却，然而                        |
| 原因关系   | eCau          | event Cause        | 因为，既然                       |
| 结果关系   | eResu         | event Result       | 因此，以致                       |
| 推论关系   | eInf          | event Inference    | 才，则                         |
| 条件关系   | eCond         | event Condition    | 只要，除非                       |
| 假设关系   | eSupp         | event Supposition  | 如果，要是                       |
| 让步关系   | eConc         | event Concession   | 纵使，哪怕                       |
| 手段关系   | eMetd         | event Method       |                             |
| 目的关系   | ePurp         | event Purpose      | 为了，以便                       |
| 割舍关系   | eAban         | event Abandonment  | 与其，也不                       |
| 选取关系   | ePref         | event Preference   | 不如，宁愿                       |
| 总括关系   | eSum          | event Summary      | 总而言之                        |
| 分叙关系   | eRect         | event Recount      | 例如，比方说                      |
| 连词标记   | mConj         | Recount Marker     | 和，或                         |
| 的字标记   | mAux          | Auxiliary          | 的，地，得                       |
| 介词标记   | mPrep         | Preposition        | 把，被                         |
| 语气标记   | mTone         | Tone               | 吗，呢                         |
| 时间标记   | mTime         | Time               | 才，曾经                        |
| 范围标记   | mRang         | Range              | 都，到处                        |
| 程度标记   | mDegr         | Degree             | 很，稍微                        |
| 频率标记   | mFreq         | Frequency Marker   | 再，常常                        |
| 趋向标记   | mDir          | Direction Marker   | 上去，下来                       |
| 插入语标记  | mPars         | Parenthesis Marker | 总的来说，众所周知                   |
| 否定标记   | mNeg          | Negation Marker    | 不，没，未                       |
| 情态标记   | mMod          | Modal Marker       | 幸亏，会，能                      |
| 标点标记   | mPunc         | Punctuation Marker | ，。！                         |
| 重复标记   | mPept         | Repetition Marker  | 走啊走 (走 –> 走)                |
| 多数标记   | mMaj          | Majority Marker    | 们，等                         |
| 实词虚化标记 | mVain         | Vain Marker        |                             |
| 离合标记   | mSepa         | Seperation Marker  | 吃了个饭 (吃 –> 饭) 洗了个澡 (洗 –> 澡) |
| 根节点    | Root          | Root               | 全句核心节点                      |