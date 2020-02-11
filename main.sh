# /bin/bash
pushd data
  wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz -O ldcc-20140209.tar.gz
  tar -zxf ldcc-20140209.tar.gz
popd

python data_split.py

pushd embs
  wget https://github.com/singletongue/WikiEntVec/releases/download/20190520/jawiki.all_vectors.200d.txt.bz2 -O jawiki.all_vectors.200d.txt.bz2
  bzip2 -d jawiki.all_vectors.200d.txt.bz2
  wget https://github.com/singletongue/WikiEntVec/releases/download/20190520/jawiki.entity_vectors.200d.txt.bz2 -O jawiki.entity_vectors.200d.txt.bz2
  bzip2 -d jawiki.entity_vectors.200d.txt.bz2
  wget https://github.com/singletongue/WikiEntVec/releases/download/20190520/jawiki.word_vectors.200d.txt.bz2 -O jawiki.word_vectors.200d.txt.bz2
  bzip2 -d jawiki.entity_vectors.200d.txt.bz2 jawiki.word_vectors.200d.txt.bz2
popd

python main.py w2v.model_name=all,entity,word model.hidden_size=32,64,128,256 training.learning_rate=0.01,0.005 -m
