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

# Chinese Proposition Bank

|      | 标签       | 角色    | 例子                      |
|------|----------|-------|-------------------------|
| 中心角色 | ARG0     | 施事者   | (ARG0中国政府)提供援助         |
|      | ARG1     | 受事者   | 中国政府提供(ARG1援助)          |
|      | ARG2     | 依谓词而定 | 失业率控制(ARG2在百分之十内)       |
|      | ARG3     | 依谓词而定 | (ARG3从城市)扩大到农村          |
|      | ARG4     | 依谓词而定 | 提高(ARG4百分之二十)          |
| 附属角色 | ARGM-ADV | 状语    | (ARGM-ADV共同)承担          |
|      | ARGM-BNF | 受益者   | (ARGM-BNF为其他国家)进行融资     |
|      | ARGM-CND | 条件    | (ARGM-CND如果成功)，他就留下     |
|      | ARGM-DIR | 方向    | (ARGM-DIR向和平)迈出一大步      |
|      | ARGM-EXT | 范围    | 在北京逗留(ARGM-EXT两天)      |
|      | ARGM-FRQ | 频率    | 每半年执行(ARGM-FRQ一次)      |
|      | ARGM-LOC | 地点、位置 | (ARGM-LOC在机场)被捕获        |
|      | ARGM-MNR | 方式    | (ARGM-MNR以中英文)发行        |
|      | ARGM-PRP | 目的或原因 | (ARGM-PRP由于危机)而破产       |
|      | ARGM-TMP | 时间    | 公司(ARGM-TMP去年)成立       |
|      | ARGM-TPC | 主题    | (ARGM-TPC稳定政策)，核心是...   |
|      | ARGM-DIS | 话语标记  | (ARGM-DIS因此)，他感到不公      |
|      | ARGM-CRD | 并列论元  | (ARGM-CRD与台湾)非正式接触      |
|      | ARGM-PRD | 次谓词   | 指控廉政公署五人(ARGM-PRD接受贿赂) |


```{note}
Although ARG0 and ARG1 share general definitions across all predicates, word sense disambiguation is required to find 
the coresponding definition of semantic roles. Given the word sense of `变化`, say `变化-2`, 
[its second frameset](http://verbs.colorado.edu/chinese/cpb/html_frames/0183-bian-hua.html) can 
be found which defines the following 2 arguments:

1.    ARG0: agent/cause
2.    ARG1: entity arg0 changes

These definitions are different from that of frameset `变化-1`:

1.    ARG0: entity undergoing change
   
Sometimes, the number of arguments and definitions can vary a lot across framesets. 
In summary, word sense disambiguation is essential if SRL is to be used to best effect in practical applications  
```