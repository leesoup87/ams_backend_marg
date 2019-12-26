import tensorflow as tf
        
def get_activity_clf_graph() : 
    '''
        get the activity_clf graph

        return : 
        g : graph contains classification model (only constants)
        name_input : name of the input tensor
        name_output : name of the prediction tensor
    '''
    with tf.Graph().as_default() as graph :
        with tf.gfile.GFile('./src/server/activity_clf/frozen.pb', 'rb') as f: 
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        x_input = tf.placeholder(shape=(None,), dtype=tf.float32, name='input')
        pred_tensor, = tf.import_graph_def(graph_def, name='',
                                 input_map={'input_wav:0': x_input},
                                 return_elements=['pred:0'])
    return graph, 'input:0', 'pred:0'
        
