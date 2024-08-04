# TransformerEN2CN

Translate English to Chinese using Transformer.

使用Transformer模型完成英文翻译为中文任务。

- [TransformerEN2CN](#transformeren2cn)
  - [Dataset](#dataset)

## Dataset

cmn数据集：中英文预料，https://www.manythings.org/anki/

格式：英文 + TAB + 中文 + TAB + Attribution描述

数据量：24360条中英文对

```shell
Hi.	嗨。	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #891077 (Martha)
Hi.	你好。	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #4857568 (musclegirlxyp)
Run.	你用跑的。	CC-BY 2.0 (France) Attribution: tatoeba.org #4008918 (JSakuragi) & #3748344 (egg0073)
```
