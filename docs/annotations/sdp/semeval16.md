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

## CSDP

SemEval2016 adopts the CSDP guideline listed as follows.

### 语义关系标注标签集

| 分类         |              |                 |                                                              |
| ------------ | ------------ | --------------- | ------------------------------------------------------------ |
| 语义周边角色 | 主体角色     | 施事AGT；       | 施事Agt；感事Aft                                             |
|              |              | 当事EXP；       | 当事Exp；领事Poss                                            |
|              | 客体角色     | 受事PAT；       | 受事Pat                                                      |
|              |              | 客事CONT；      | 客事Cont；成事Prod；结局Cons                                 |
|              |              | 涉事DATV；      | 涉事Datv；比较Comp；源事Orig                                 |
|              |              | 系事LINK；      | 类事Clas；属事Belg                                           |
|              | 情境角色     | 工具TOOL；      | 工具Tool                                                     |
|              |              | 材料MATL；      | 材料Matl                                                     |
|              |              | 方式MANN；      | 方式Mann；依据Accd                                           |
|              |              | 范围SCO；       | 范围Sco                                                      |
|              |              | 缘由REAS；      | 缘故Reas；意图Int                                            |
|              |              | 时间TIME；      | 时间Time；时间起点Tini；时间终点Tfin；时段Tdur；时距Trang    |
|              |              | 空间LOC；       | 空间Loc；原处所Lini；终处所Lfin；通过处所Lthru；趋向Dir      |
|              |              | 度量MEAS；      | 数量Quan；起始量Nini；终止量Nfin；数量短语Qp；频率Freq；顺序Seq；变化量Nvar |
|              |              | 状态STAT；      | 状态Stat；起始状态Sini；终止状态Sfin；历经状态Sproc          |
|              |              | 修饰FEAT；      | 描写Desc；宿主Host；名词修饰语Nmod；时间修饰语Tmod           |
| 语义结构关系 | 反关系       | 反施事rAGT；    | 反施事rAgt；反感事rAft                                       |
|              |              | 反当事rEXP。    | 反当事rExp；反领事rPoss                                      |
|              |              | 反受事rPAT；    | 反受事rPat                                                   |
|              |              | 反客事rCONT；   | 反客事rCont；反成事rProd；反结局rCons                        |
|              |              | 反涉事rDATV；   | 反涉事rDatv；反比较rComp；反源事rOrig                        |
|              |              | 反系事rLINK。   | 反类事rClas；反属事rBelg                                     |
|              |              | 反工具rTOOL；   | 反工具rTool                                                  |
|              |              | 反材料rMATL；   | 反材料rMatl                                                  |
|              |              | 反方式RMANN；   | 反方式rMann；反依据rAccd                                     |
|              |              | 反范围rSCO；    | 反范围rSco                                                   |
|              |              | 反缘由rREAS；   | 反缘故rReas；反意图rInt                                      |
|              |              | 反时间rTIME；   | 反时间rTime；反时间起点rTini；反时间终点rTfin；反时段rTdur；反时距rTrang |
|              |              | 反空间rLOC；    | 反空间rLoc；反原处所rLini；反终处所rLfin；反通过处所rLthru；反趋向rDir |
|              |              | 反度量rMEAS；   | 反数量rQuan；反起始量rNini；反终止量rNfin；反数量短语rQp；反频率rFreq；反顺序rSeq；反变化量rNvar |
|              |              | 反状态rSTAT；   | 反状态rStat；反起始状态rSini；反终止状态rSfin；反历经状态rSproc |
|              |              | 反修饰rFEAT；   | 反描写rDesc；反宿主rHost; 反名词修饰语rNmod; 反时间修饰语rTmod |
|              | 嵌套事件关系 | 嵌套施事dAGT；  | 嵌套施事dAgt；嵌套感事dAft                                   |
|              |              | 嵌套当事dEXP。  | 嵌套当事dExp；嵌套领事dPoss                                  |
|              |              | 嵌套受事dPAT；  | 嵌套受事dPat                                                 |
|              |              | 嵌套客事dCONT； | 嵌套客事dCont；嵌套成事dProd；嵌套结局dCons                  |
|              |              | 嵌套涉事dDATV； | 嵌套涉事dDatv；嵌套比较dComp；嵌套源事dOrig                  |
|              |              | 嵌套系事dLINK。 | 嵌套类事dClas；嵌套属事dBelg                                 |
|              |              | 嵌套工具dTOOL； | 嵌套工具dTool                                                |
|              |              | 嵌套材料dMATL； | 嵌套材料dMatl                                                |
|              |              | 嵌套方式dMANN； | 嵌套方式dMann；嵌套依据dAccd                                 |
|              |              | 嵌套范围dSCO；  | 嵌套范围dSco                                                 |
|              |              | 嵌套缘由dREAS； | 嵌套缘故dReas；嵌套意图dInt                                  |
|              |              | 嵌套时间dTIME； | 嵌套时间dTime；嵌套时间起点dTini；嵌套时间终点dTfin；嵌套时段dTdur；嵌套时距dTrang |
|              |              | 嵌套空间dLOC；  | 嵌套空间dLoc；嵌套原处所dLini；嵌套终处所dLfin；嵌套通过处所dLthru；嵌套趋向dDir |
|              |              | 嵌套度量dMEAS； | 嵌套数量dQuan；嵌套起始量dNini；嵌套终止量dNfin；嵌套数量短语dQp；嵌套频率dFreq；嵌套顺序dSeq；嵌套变化量dNvar |
|              |              | 嵌套状态dSTAT； | 嵌套状态dStat；嵌套起始状态dSini；嵌套终止状态dSfin；嵌套历经状态dSproc |
|              |              | 嵌套修饰dFEAT； | 嵌套描写dDesc；嵌套宿主dHost; 嵌套名词修饰语dNmod; 嵌套时间修饰语dTmod |
|              | 事件关系     | 并列关系eCOO；  | 并列eCoo；等同eEqu；分叙eRect；选择eSelt;割舍eAban；选取ePref；总括eSum |
|              |              | 先行关系ePREC； | 先行ePrec；原因eCau；条件eCond；假设eSupp；手段eMetd；让步eConc |
|              |              | 后继关系eSUCC； | 后继eSucc；递进eProg；转折 eAdvt；目的ePurp；结果eResu；推论eInf |
| 语义依附标记 | 标点标记     | 标点标记mPUNC； | 标点标记mPunc                                                |
|              | 依附标记     | 否定标记mNEG；  | 否定标记mNeg                                                 |
|              |              | 关系标记mRELA； | 连词标记mConj；介词标记mPrep                                 |
|              |              | 依附标记mDEPD； | 语气标记mTone；时间标记mTime;范围标记mRang；情态标记mMod； 频率标记mFreq；程度标记mDegr；趋向标记mDir；的字标记mAux； 多数标记mMaj；插入语标记mPars；离合标记mSepa；实词虚化标记mVain 重复标记mRept |

## SemEval2016

The following table is a subset of CSDP but offers some examples to illustrate the idea.

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

See also [SemEval-2016 Task 9](https://www.hankcs.com/nlp/sdp-corpus.html) and [CSDP](https://csdp-doc.readthedocs.io/zh_CN/latest/%E9%99%84%E5%BD%95/).
