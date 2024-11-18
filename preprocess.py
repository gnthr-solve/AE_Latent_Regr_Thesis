

from preprocessing import preprocess_raw
from preprocessing.misc import drop_unnamed_y, drop_unnamed_Xmax
from preprocessing.normalisation import normalise_stored_tensors
#from preprocessing.investigate import investigate_tensor, investigate_index_mapping, investigate_id_match




if __name__=="__main__":
    
    #--- Preprocessing ---#
    #drop_unnamed_Xmax()
    #join_extracts()
    preprocess_raw(kind = 'key')
    preprocess_raw(kind = 'max')

    #normalise_stored_tensors()

    #investigate_tensor()
    #investigate_index_mapping()
    #investigate_id_match()