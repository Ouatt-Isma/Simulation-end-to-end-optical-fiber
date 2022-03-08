import tensorflow as tf
from algorithm import *
from algorithm import gray_mapper_qam
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.utils import plot_model

#tqdm 

# define cnn model
# def define_model(params, ns):
# 	model = tf.keras.Sequential()
# 	model.add(tf.keras.layers.Conv2D(params.N/2, (1, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(params.N, 2, 1)))

#     model.add(tf.keras.layers.Conv2D(params.N/4, (2, 3), activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Flatten())
# 	model.add(Dense(params.N, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Dense(ns, activation='softmax'))
# 	# compile model
# 	opt = SGD(learning_rate=0.01, momentum=0.9)
# 	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# 	return model
 
def define_model(params, M=16, ns=16):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(int(params.N/2), (3,1), activation='relu', kernel_initializer='he_uniform', input_shape=(params.N, 2, 1)))
    model.add(tf.keras.layers.Conv2D(int(params.N/4), (3, 2), activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(params.N, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(int(params.N/2), activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(M, activation='softmax'))
    
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    #plot_model(model)
    return [tf.keras.models.clone_model(model) for i in range(ns)]
    

def define_model2(params, M=16, ns=16):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2*params.N, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.50))
    model.add(tf.keras.layers.Dense(params.N, activation='relu'))
    
    model.add(tf.keras.layers.Dense(int(params.N/2), activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.50))
    model.add(tf.keras.layers.Dense(int(params.N/4), activation='relu'))
   
    model.add(tf.keras.layers.Dense(int(params.N/8), activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(M, activation='softmax'))
    
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    #plot_model(model)
    return [tf.keras.models.clone_model(model) for i in range(ns)]
if __name__=='__main__':
    params = parameters()
    pp = parameters(_L= 1000000, _B = 10**6, _adb= 0.2, 
                    _D = 17*(10**(3)), _gama= 1.27*(10**(-3)), 
                    _lambda0=1.55*(10**(-6)), _P = 6*(10**(-3)),
                    _T=400, _N=2**10)
    params.Bn = pp.Bn
    ns=16
    """
    # pred_model(params)
    # cnt = gray_mapper_qam(16)
    # SoftMaxMapper = []
    # for i in cnt:
    #     SoftMaxMapper.append(cnt[i])
    # print(SoftMaxMapper)
    """   
    
    ### TO SEE ### 
    inp = np.load("input_signalva.npy")
    S_true = np.load("input_symbolva.npy")
    outp = np.load("output_signalva.npy")
    
    # inp1 = np.load("input_signalttt.npy")
    # S_true1= np.load("input_symbolttt.npy")
    # outp1 = np.load("output_signalttt.npy")

    # inp = np.concatenate((inp1, inp2))
    # S_true= np.concatenate((S_true1, S_true2))
    # outp= np.concatenate((outp1, outp2))
    
    # np.save("input_signalva.npy", inp)
    # np.save("input_symbolva.npy", S_true)
    # np.save("output_signalva.npy", outp)
    
    # inp = np.load("input_signalvo.npy")
    # S_true= np.load("input_symbolvo.npy")
    # outp= np.load("output_signalvo.npy")
    
 
    
    print(np.shape(inp))
    # exit(0)
    
    
    
    """
    # plt.plot(params.t, np.abs(inp[0]))
    # plt.show()
    # plt.plot(params.t, np.abs(outp[0]))
    # plt.show()
    """
    
    Data_outp = np.array([np.concatenate((np.array(out.real,ndmin=2).T,np.array(out.imag,ndmin=2).T),axis=1) for out in outp])
    Data_outp = Data_outp.reshape(len(Data_outp), params.N, 2, 1)
    S = np.array([[str(i) for i in s] for s in S_true])
    onehot_encoder = OneHotEncoder(categories='auto')
    onehot_encoder.fit(S)
    S_transform = onehot_encoder.transform(S).toarray()
    
    #scaling into [0, 1]
    max_scale = np.max(Data_outp)
    print(max_scale)
    
    (unique, counts) = np.unique(S, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)
    Data_outp/=max_scale
    
    X_train, X_test, y_train, y_test = train_test_split(Data_outp, S_transform, test_size=0.1, 
                                                        random_state=42)
    Big_model = define_model2(params)
    opt = 'adam'
    loss='categorical_crossentropy'
    epochs=20
    batch_size=64
    print(np.shape(Data_outp))
    for i in range(1,2):
        #data_set_s = S_transform[:,i:16]
        y_traini = y_train[:,16*i:16*(i+1)]
        y_testi = y_test[:,16*i:16*(i+1)]
        mod = Big_model[i]
        mod.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        
        print("training ", i+1, "th symbol")
        mod.fit(X_train, y_traini, epochs=epochs, batch_size=batch_size)
        print("evaluation")
        mod.evaluate(X_test, y_testi)
        
    print()
    # mod.evaluate(Data_outp[ind:], data_set_s[ind:])
    # print(mod.predict(Data_outp[ind:ind+1]))
    # print(data_set_s[ind])
    # print(S[ind])
    # print(onehot_encoder.transform(S[ind:ind+1]).toarray())
    
