
from transformers import BertTokenizer, BertModel
import torch
import tensorflow
from  sentence_transformers  import  SentenceTransformer

#bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

class text_feature(object):
    model  =  SentenceTransformer ( 'all-MiniLM-L6-v2' )
    #利用Bert获得指定文本的向量表示
    # def get_text_feature(self,text): 
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    #     model = BertModel.from_pretrained('bert-base-chinese')
    #     inputs = tokenizer(text, return_tensors='pt')
    #     outputs = model(**inputs)
    #     return outputs
    
    def get_sentence_feature(self,sentence):
        
        if(sentence is None):
            return 0
        sentence_embeddings = text_feature.model.encode(sentence)
        #print(type(sentence_embeddings))
        if(sentence_embeddings is None):
            print("None!!!!!!")
        return sentence_embeddings
      

if __name__ == "__main__":
    tf = text_feature()
    print(tf.get_sentence_feature("试一下"))