说一说elmo缺点。

ELMO之后有了Bert，这样回看，ELMO 有什么值得改进的缺点呢？
首先，一个非常明显的缺点在特征抽取器选择方面，ELMO 使用了 LSTM 而不是新贵 Transformer，
Transformer 是谷歌在 17 年做机器翻译任务的“Attention is all you need”的论文中提出的，引起了相当大的反响，
很多研究已经证明了 Transformer 提取特征的能力是要远强于 LSTM 的。如果 ELMO 采取 Transformer 作为特征提取器，
那么估计 Bert 的反响远不如现在的这种火爆场面。
另外一点，ELMO 采取双向拼接这种融合特征的能力可能比 Bert 一体化的融合特征方式弱，
但是，这只是一种从道理推断产生的怀疑，目前并没有具体实验说明这一点。
