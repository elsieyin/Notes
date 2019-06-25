[allennlp](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md#using-elmo-as-a-pytorch-module-to-train-a-new-model)

Pytorch版本
Allennlp的ELMo的API为allennlp.modules.elmo.Elmo,摘要如下:
 
options_file : ELMo JSON options file
weight_file : ELMo hdf5 weight file
num_output_representations: The number of ELMo representation layers to output.
requires_grad: optional, If True, compute gradient of ELMo parameters for fine tuning.
其中一个关键的参数为num_output_reprensenations  这个num_output_representation 
一开始理解这个参数是输出三层中前几层，但是其实并不是这样，因为这个参数的取值可以是任意正整数。 
我们知道ELMo文章中最后词向量表示即公式1计算的是三层表示的线性加权，针对不同的任务，可能会有不同的加权比例。 
因此num_output_representation 表示输出多种线性加权的词向量，即多个公式1产生的词向量。 
而Allennlp的ELMo 并不直接提供中间的三层输出(char-cnn, lstm-1, lstm-2)，不过可以通过稍微修改源代码的方法获得。


```from allennlp.modules.elmo import Elmo, batch_to_ids
options_file = "options.json"  # 配置文件地址 
weight_file = "weights.hdf5" # 权重文件地址
# 这里的1表示产生一组线性加权的词向量。
# 如果改成2 即产生两组不同的线性加权的词向量。
elmo = Elmo(options_file, weight_file, 1, dropout=0)
# use batch_to_ids to convert sentences to character ids
sentence_lists = ["I have a dog", "How are you , today is Monday","I am fine thanks"]
character_ids = batch_to_ids(sentences_lists)
embeddings = elmo(character_ids)['elmo_representations']
```

TF的版本比较简单
[TFhub](https://tfhub.dev/google/elmo/2)
详见训练作业

```
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
tokens_input = [["the", "cat", "is", "on", "the", "mat"],
                ["dogs", "are", "in", "the", "fog", ""]]
tokens_length = [6, 5]
embeddings = elmo(
    inputs={
        "tokens": tokens_input,
        "sequence_len": tokens_length
    },
    signature="tokens",
    as_dict=True)["elmo"]
```

word_emb: ELMo的最开始一层的基于character的word embedding, shape为[batch_size, max_length, 512]
lstm_outpus1/2: ELMo中的第一层和第二层LSTM的隐状态输出，shape同样为 [batch_size, max_length, 1024]
elmo: 对应文章中的公式1，每个词的输入层(word_emb)，第一层LSTM输出，第二层LSTM输出的线性加权之后的最终的词向量，shape为[batch_size, max_length, 1024]，此外这个线性权重是可训练的。
default: 前面得到的均为word级别的向量，这个选项给出了简单使用mean-pooling求的句子级别的向量，即将上述elmo的所有词取平均，方便后续下游任务。
sequence_len 输入中每个句子的长度
一般情况使用output ['elmo'] 也就是Allennlp版本的elmo_representations， 即可得到每个词的ELMo 词向量，即可用于后续的任务，比如分类等。
