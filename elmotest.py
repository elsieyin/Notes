from allennlp.modules.elmo import Elmo, batch_to_ids
options_file = "options.json"  # 配置文件地址 
weight_file = "weights.hdf5" # 权重文件地址
# 这里的1表示产生一组线性加权的词向量。
# 如果改成2 即产生两组不同的线性加权的词向量。
elmo = Elmo(options_file, weight_file, 1, dropout=0)
# use batch_to_ids to convert sentences to character ids
sentence_lists = ["I have a dog", "How are you , today is Monday","I am fine thanks"]
character_ids = batch_to_ids(sentences_lists)
embeddings = elmo(character_ids)['elmo_representations']
