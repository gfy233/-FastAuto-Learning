from cv2 import idct
from towhee import pipeline
import os
import numpy as np
embedding_pipeline = pipeline('image-embedding')


class picture_feature(object):

    def get_picture_feature(self,id):
        #output = embedding_pipeline('/data/gfy2021/gfy/stacking_DRANN/process_data/image/1.jpg')
        #print("output",output)
  
        filelist = os.listdir("/data/gfy2021/gfy/KDD/process_data/image")
        #if("")print(id)
        for file in filelist:
            #if(id==672):
                #print(id,file.split(".")[0])
            if(file.split(".")[0]==str(id)):
                
                #print(embedding_pipeline('/data/gfy2021/gfy/stacking_DRANN/process_data/image/'+file))
                return embedding_pipeline('/data/gfy2021/gfy/KDD/process_data/image/'+file)
        #print("0")
        return  np.zeros(10)

    
if __name__ == "__main__":

    #print( embedding_pipeline('/data/gfy2021/gfy/stacking_DRANN/process_data/image/1.jpg'))
    tf = picture_feature()
    print(tf.get_picture_feature("1013"))
