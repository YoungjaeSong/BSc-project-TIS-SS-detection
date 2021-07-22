import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import *

def MyNN1():
    model = Sequential()
    model.add(layers.Conv1D(70, 9, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(100, 7))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(150, 7))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(230, 7))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(300, 5))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(64, activation='elu'))
    model.add(layers.Dense(4, activation='elu'))

    return model

def MyNN2():
    # Input layer
    Input_X = Input(shape=(600, 4))

    # Shared Layer
    X = Conv1D(64, 9, activation='relu')(Input_X)
    X = Dropout(0.3)(X)
    X = MaxPooling1D(pool_size=3, strides=3)(X)
    X = Conv1D(64, 7, activation='elu')(X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(pool_size=3, strides=3)(X)
    X = Conv1D(64, 7, activation='elu')(X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(pool_size=3, strides=3)(X)
    X = Flatten()(X)
    X = Dense(64)(X)
    X = Dense(64)(X)

    # Task specific layers - TIS
    X_TIS = Dense(64)(X)
    X_TIS = Dense(32)(X_TIS)
    Y_TIS1 = Dense(1, activation="sigmoid", name="TIS_prob")(X_TIS)
    Y_TIS2 = Dense(1, activation="elu", name="TIS_loc")(X_TIS)

    # Task specific layers - SS
    X_SS = Dense(64)(X)
    X_SS = Dense(32)(X_SS)
    Y_SS1 = Dense(1, activation="sigmoid", name="SS_prob")(X_SS)
    Y_SS2 = Dense(3, activation="elu", name="SS_loc")(X_SS)

    model = Model(inputs = Input_X, outputs=[Y_TIS1, Y_SS1, Y_TIS2, Y_SS2])

    return model

def MyNN3():
    # Input layer
    Input_X = Input(shape=(600, 4))

    # Shared Layer
    X = Conv1D(64, 9, activation='relu')(Input_X)
    X = Dropout(0.3)(X)
    X = MaxPooling1D(pool_size=3, strides=3)(X)
    X = Conv1D(64, 7, activation='elu')(X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(pool_size=3, strides=3)(X)
    X = Conv1D(64, 7, activation='elu')(X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(pool_size=3, strides=3)(X)
    X = Flatten()(X)
    X = Dense(64)(X)
    X = Dense(64)(X)# kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=l2(1e-4), activity_regularizer=l2(1e-5))(X)

    # Task specific layers - TIS
    X_TIS = Dense(64)(X)
    X_TIS = Dense(32)(X_TIS)
    Y_TIS1 = Dense(1, activation="sigmoid", name="TIS_prob")(X_TIS)
    Y_TIS2 = Dense(1, activation="elu", name="TIS_loc")(X_TIS)

    # Task specific layers - SS
    X_SS = Dense(64)(X)
    X_SS = Dense(32)(X_SS)
    Y_SS1 = Dense(1, activation="sigmoid", name="SS_prob")(X_SS)
    Y_SS2 = Dense(3, activation="elu", name="SS_loc")(X_SS)

    model = Model(inputs = Input_X, outputs=[Y_TIS1, Y_SS1, Y_TIS2, Y_SS2])

    return model
def MyNN4():
    Input_X = Input(shape=(600, 4))
    
    X = Conv1D(64, 4)(Input_X)
    X = MaxPooling1D(pool_size=4)(X)
    X = Conv1D(128, 4)(X)
    X = MaxPooling1D(pool_size=4)(X)
    X = Conv1D(256, 4)(X)
    X = MaxPooling1D(pool_size=4)(X)
    X = Conv1D(256, 4)(X)
    X = MaxPooling1D(pool_size=4)(X)
    # X = Conv1D(128, 4)(X)
    # X = MaxPooling1D(pool_size=4)(X)
    # X = Conv1D(64, 4)(X)
    # X = MaxPooling1D(pool_size=4)(X)
    # X = Conv1D(64, 4)(X)
    # X = MaxPooling1D(pool_size=4)(X)
    X = Dropout(0.3)(X)
    X = Flatten()(X)
    # X = Dense(128)(X)
    X = Dense(64)(X)
    X = Dropout(0.3)(X)
    Y_TISprob = Dense(1, name='TIS_prob', activation='sigmoid')(X)
    Y_SSprob = Dense(1, name='SS_prob', activation='sigmoid')(X)
    Y_TISloc = Dense(1, name='TIS_loc', activation='elu')(X)
    Y_SSloc = Dense(3, name='SS_loc', activation='elu')(X)

    model = Model(inputs=Input_X, outputs=[Y_TISprob, Y_SSprob, Y_TISloc, Y_SSloc])

    return model
def MyNN5():
    Input_X = Input(shape=(600, 4))

    # X = Dropout(0.5)(Input_X)

    X = Conv1D(16, 4)(Input_X)
    X = MaxPooling1D(pool_size=4)(X)
    X = Conv1D(32, 4)(X)
    X = MaxPooling1D(pool_size=4)(X)
    X = Conv1D(32, 4)(X)
    X = MaxPooling1D(pool_size=4)(X)
    X = Conv1D(16, 4)(X)
    X = MaxPooling1D(pool_size=4)(X)
    X = Flatten()(X)

    X_TIS = Dense(64)(X)
    X_TIS = Dropout(0.5)(X_TIS)
    X_TIS = Dense(64)(X_TIS)
    X_TIS = Dropout(0.3)(X_TIS)

    X_SS = Dense(64)(X)
    X_SS = Dropout(0.5)(X_SS)
    X_SS = Dense(64)(X_SS)
    X_SS = Dropout(0.3)(X_SS)

    Y_TISprob = Dense(1, name='TIS_prob', activation='sigmoid')(X_TIS)
    Y_SSprob = Dense(1, name='SS_prob', activation='sigmoid')(X_SS)
    Y_TISloc = Dense(1, name='TIS_loc', activation='elu')(X_TIS)
    Y_SSloc = Dense(3, name='SS_loc', activation='elu')(X_SS)

    model = Model(inputs=Input_X, outputs=[Y_TISprob, Y_SSprob, Y_TISloc, Y_SSloc])

    return model


def MyNN6():
    Input_X = Input(shape=(600, 4))
    X = Conv1D(200, 7)(Input_X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(100, 7)(X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(80, 7)(X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(70, 7)(X)
    X = MaxPooling1D(pool_size=3)(X)
    # X = Conv1D(60, 7)(X)
    # X = MaxPooling1D(pool_size=3)(X)
    # X = Conv1D(50, 7)(X)
    # X = MaxPooling1D(pool_size=3)(X)
    X = Flatten()(X)

    X = Dense(256)(X)
    X = Dropout(0.2)(X)
    X = Dense(128)(X)
    X = Dropout(0.2)(X)
    X = Dense(128)(X)
    X = Dropout(0.2)(X)

    # Task Specific Layers
    # TIS Specific layers
    X_TIS = Dense(64)(X)
    X_TIS = Dropout(0.2)(X_TIS)
    X_TIS = Dense(64)(X_TIS)
    X_TIS = Dropout(0.2)(X_TIS)

    Y_TIS_loc = Dense(1, activation='elu', name='TIS_loc')(X_TIS)
    Y_TIS_prob = Dense(1, activation='sigmoid', name='TIS_prob')(X_TIS)

    # SS specific layers
    X_SS = Dense(64)(X)
    X_SS = Dropout(0.2)(X_SS)
    X_SS = Dense(64)(X_SS)
    X_SS = Dropout(0.2)(X_SS)

    Y_SS_loc = Dense(3, activation='elu', name='SS_loc')(X_SS)
    Y_SS_prob = Dense(1, activation='sigmoid', name='SS_prob')(X_SS)

    model = Model(inputs=Input_X, outputs=[Y_TIS_prob, Y_SS_prob, Y_TIS_loc, Y_SS_loc])

    return model
    
def MyNN7():
    Input_X = Input(shape=(600, 4))
    X = Conv1D(200, 7)(Input_X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(100, 7)(X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(80, 7)(X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(70, 7)(X)
    X = MaxPooling1D(pool_size=3)(X)
    # X = Conv1D(60, 7)(X)
    # X = MaxPooling1D(pool_size=3)(X)
    # X = Conv1D(50, 7)(X)
    # X = MaxPooling1D(pool_size=3)(X)
    X = Flatten()(X)

    X = Dense(128)(X)
    X = Dropout(0.2)(X)
    X = Dense(128)(X)
    X = Dropout(0.2)(X)
    X = Dense(128)(X)
    X = Dropout(0.2)(X)

    # Task Specific Layers
    # Probability Specific layers
    X_prob = Dense(64)(X)
    X_prob = Dropout(0.2)(X_prob)
    X_prob = Dense(64)(X_prob)
    X_prob = Dropout(0.2)(X_prob)

    Y_SS_prob = Dense(1, activation='sigmoid', name='SS_prob')(X_prob)
    Y_TIS_prob = Dense(1, activation='sigmoid', name='TIS_prob')(X_prob)

    # Location specific layers
    X_loc = Dense(64)(X)
    X_loc = Dropout(0.2)(X_loc)
    X_loc = Dense(64)(X_loc)
    X_loc = Dropout(0.2)(X_loc)

    Y_SS_loc = Dense(3, activation='elu', name='SS_loc')(X_loc)
    Y_TIS_loc = Dense(1, activation='elu', name='TIS_loc')(X_loc)

    model = Model(inputs=Input_X, outputs=[Y_TIS_prob, Y_SS_prob, Y_TIS_loc, Y_SS_loc])

    return model

def MyNN8():
    Input_X = Input(shape=(600, 4))
    X = Conv1D(200, 7)(Input_X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(100, 7)(X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(80, 7)(X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(70, 7)(X)
    X = MaxPooling1D(pool_size=3)(X)
    # X = Conv1D(60, 7)(X)
    # X = MaxPooling1D(pool_size=3)(X)
    # X = Conv1D(50, 7)(X)
    # X = MaxPooling1D(pool_size=3)(X)
    X = Flatten()(X)

    X = Dense(128)(X)
    X = Dropout(0.2)(X)
    X = Dense(128)(X)
    X = Dropout(0.2)(X)
    X = Dense(128)(X)
    X = Dropout(0.2)(X)

    # Task Specific Layers
    # Probability Specific layers
    X_prob = Dense(64)(X)
    X_prob = Dropout(0.2)(X_prob)
    X_prob = Dense(64)(X_prob)
    X_prob = Dropout(0.2)(X_prob)
    X_prob = Dense(64)(X_prob)
    X_prob = Dropout(0.2)(X_prob)

    Y_SS_prob = Dense(1, activation='sigmoid', name='SS_prob')(X_prob)
    Y_TIS_prob = Dense(1, activation='sigmoid', name='TIS_prob')(X_prob)

    # Location specific layers
    X_loc = Dense(64)(X)
    X_loc = Dropout(0.2)(X_loc)
    X_loc = Dense(64)(X_loc)
    X_loc = Dropout(0.2)(X_loc)
    X_loc = Dense(64)(X_loc)
    X_loc = Dropout(0.2)(X_loc)

    Y_SS_loc = Dense(3, activation='elu', name='SS_loc')(X_loc)
    Y_TIS_loc = Dense(1, activation='elu', name='TIS_loc')(X_loc)

    model = Model(inputs=Input_X, outputs=[Y_TIS_prob, Y_SS_prob, Y_TIS_loc, Y_SS_loc])

    return model

def MyNN9():
    Input_X = Input(shape=(600, 4))
    X = Conv1D(200, 7)(Input_X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(100, 7)(X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(80, 7)(X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(70, 7)(X)
    X = MaxPooling1D(pool_size=3)(X)
    # X = Conv1D(60, 7)(X)
    # X = MaxPooling1D(pool_size=3)(X)
    # X = Conv1D(50, 7)(X)
    # X = MaxPooling1D(pool_size=3)(X)
    X = Flatten()(X)

    X = Dense(256)(X)
    X = Dropout(0.2)(X)
    X = Dense(128)(X)
    X = Dropout(0.2)(X)
    X = Dense(128)(X)
    X = Dropout(0.2)(X)
    X = Dense(128)(X)
    X = Dropout(0.2)(X)
    X = Dense(64)(X)
    X = Dropout(0.2)(X)
    X = Dense(64)(X)
    X = Dropout(0.2)(X)

    # Task Specific Layers
    # TIS Specific layers
    X_TIS = Dense(128)(X)
    X_TIS = Dropout(0.2)(X_TIS)
    X_TIS = Dense(64)(X_TIS)
    X_TIS = Dropout(0.2)(X_TIS)
    X_TIS = Dense(64)(X_TIS)
    X_TIS = Dropout(0.2)(X_TIS)
    X_TIS = Dense(64)(X_TIS)
    X_TIS = Dropout(0.2)(X_TIS)
    X_TIS = Dense(64)(X_TIS)
    X_TIS = Dropout(0.2)(X_TIS)
    X_TIS = Dense(64)(X_TIS)
    X_TIS = Dropout(0.2)(X_TIS)
    X_TIS = Dense(64)(X_TIS)
    X_TIS = Dropout(0.2)(X_TIS)
    X_TIS = Dense(64)(X_TIS)
    X_TIS = Dropout(0.2)(X_TIS)

    Y_TIS_loc = Dense(1, activation='elu', name='TIS_loc')(X_TIS)
    Y_TIS_prob = Dense(1, activation='sigmoid', name='TIS_prob')(X_TIS)

    # SS specific layers
    X_SS = Dense(64)(X)
    X_SS = Dropout(0.2)(X_SS)
    X_SS = Dense(64)(X_SS)
    X_SS = Dropout(0.2)(X_SS)
    X_SS = Dense(64)(X_SS)
    X_SS = Dropout(0.2)(X_SS)

    Y_SS_loc = Dense(3, activation='elu', name='SS_loc')(X_SS)
    Y_SS_prob = Dense(1, activation='sigmoid', name='SS_prob')(X_SS)

    model = Model(inputs=Input_X, outputs=[Y_TIS_prob, Y_SS_prob, Y_TIS_loc, Y_SS_loc])

    return model

# def MyNN10():
#     Input_X = Input(shape=(600, 4))
#     X = Conv1D(64, 2)(Input_X)
#     X = MaxPooling1D(pool_size=2)(X)
#     X = Conv1D(16, 3)(X)
#     X = MaxPooling1D(pool_size=2)(X)
#     X = Conv1D(8, 2)(X)
#     X = MaxPooling1D(pool_size=2)(X)
#     X = Flatten(X)
#     X = Dense(64)(X)
#     Y_TIS_loc = Dense(1, activation='elu', name)(X)
#     Y_TIS_prob = Dense(1, activation='sigmoid')(X)
#     Y_SS_loc = Dense(3, activation='elu')(X)
#     Y_SS_prob = Dense(1, activation='sigmoid')(X)

#     model = Model(inputs=Input_X, outputs())

def VGG16_reference_net1():
    # Shared Layer
    Input_X = Input(shape=(600, 4))
    X = Conv1D(64, 3)(Input_X)
    X = Conv1D(64, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(128, 3)(X)
    X = Conv1D(128, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(256, 3)(X)
    X = Conv1D(256, 3)(X)
    X = Conv1D(256, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Flatten()(X)
    X = Dense(1024)(X)
    X = Dropout(0.2)(X)

    # TIS specific layer
    X_TIS = Dense(1024)(X)
    X_TIS = Dropout(0.2)(X_TIS)
    X_TIS = Dense(1024)(X_TIS)
    X_TIS = Dropout(0.2)(X_TIS)
    Y_TIS_loc = Dense(1, activation='elu', name='TIS_loc')(X_TIS)
    Y_TIS_prob = Dense(1, activation='sigmoid', name='TIS_prob')(X_TIS)

    # SS specific layer
    X_SS = Dense(1024)(X)
    X_SS = Dropout(0.2)(X_SS)
    X_SS = Dense(1024)(X_SS)
    X_SS = Dropout(0.2)(X_SS)
    Y_SS_loc = Dense(3, activation='elu', name='SS_loc')(X_SS)
    Y_SS_prob = Dense(1, activation='sigmoid', name='SS_prob')(X_SS)

    model = Model(inputs=Input_X, outputs=[Y_TIS_prob, Y_SS_prob, Y_TIS_loc, Y_SS_loc])

    return model

def VGG16_reference_net2():
    # Shared Layer
    Input_X = Input(shape=(600, 4))
    X = Conv1D(64, 3)(Input_X)
    X = Conv1D(64, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(128, 3)(X)
    X = Conv1D(128, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(256, 3)(X)
    X = Conv1D(256, 3)(X)
    X = Conv1D(256, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Flatten()(X)
    X = Dense(1024)(X)
    X = Dropout(0.2)(X)

    # TIS specific layer
    X_TIS = Dense(1024)(X)
    X_TIS = Dropout(0.2)(X_TIS)
    X_TIS = Dense(1024)(X_TIS)
    X_TIS = Dropout(0.2)(X_TIS)
    X_TIS = Dense(512)(X_TIS)
    X_TIS = Dropout(0.2)(X_TIS)
    X_TIS = Dense(512)(X_TIS)
    X_TIS = Dropout(0.2)(X_TIS)
    Y_TIS_loc = Dense(1, activation='elu', name='TIS_loc')(X_TIS)
    Y_TIS_prob = Dense(1, activation='sigmoid', name='TIS_prob')(X_TIS)

    # SS specific layer
    X_SS = Dense(1024)(X)
    X_SS = Dropout(0.2)(X_SS)
    X_SS = Dense(1024)(X_SS)
    X_SS = Dropout(0.2)(X_SS)
    X_SS = Dense(512)(X_SS)
    X_SS = Dropout(0.2)(X_SS)
    Y_SS_loc = Dense(3, activation='elu', name='SS_loc')(X_SS)
    Y_SS_prob = Dense(1, activation='sigmoid', name='SS_prob')(X_SS)

    model = Model(inputs=Input_X, outputs=[Y_TIS_prob, Y_SS_prob, Y_TIS_loc, Y_SS_loc])

    return model

def VGG16_reference_net3():
    # Shared Layer
    Input_X = Input(shape=(600, 4))
    X = Conv1D(64, 3)(Input_X)
    X = Conv1D(64, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(128, 3)(X)
    X = Conv1D(128, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(256, 3)(X)
    X = Conv1D(256, 3)(X)
    X = Conv1D(256, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = Conv1D(512, 3)(X)
    X = MaxPooling1D(pool_size=2)(X)
    X = Flatten()(X)
    X = Dense(1024)(X)
    X = Dropout(0.2)(X)
    X = Dense(1024)(X)
    X = Dropout(0.2)(X)

    # TIS specific layer
    X_TIS = Dense(1024)(X)
    X_TIS = Dropout(0.2)(X_TIS)
    X_TIS = Dense(1024)(X_TIS)
    X_TIS = Dropout(0.2)(X_TIS)
    Y_TIS_loc = Dense(1, activation='elu', name='TIS_loc')(X_TIS)
    Y_TIS_prob = Dense(1, activation='sigmoid', name='TIS_prob')(X_TIS)

    # SS specific layer
    X_SS = Dense(1024)(X)
    X_SS = Dropout(0.2)(X_SS)
    X_SS = Dense(1024)(X_SS)
    X_SS = Dropout(0.2)(X_SS)
    Y_SS_loc = Dense(3, activation='elu', name='SS_loc')(X_SS)
    Y_SS_prob = Dense(1, activation='sigmoid', name='SS_prob')(X_SS)

    model = Model(inputs=Input_X, outputs=[Y_TIS_prob, Y_SS_prob, Y_TIS_loc, Y_SS_loc])

    return model

def TISRover1():
    Input_X = Input(shape=(600, 4))
    X = Conv1D(70, 7)(Input_X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Dropout(0.2)(X)
    X = Conv1D(100, 3)(X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Dropout(0.2)(X)
    X = Conv1D(150, 3)(X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Dropout(0.2)(X)
    X = Flatten()(X)
    X = Dense(512)(X)
    X = Dropout(0.2)(X)
    Y_TISprob = Dense(1, name='TIS_prob', activation='sigmoid')(X) # [TIS_Prob, SS_Prob, TIS_loc, 3 SS_pos]
    Y_SSprob = Dense(1, name='SS_prob', activation='sigmoid')(X)
    Y_TISloc = Dense(1, name='TIS_loc', activation='elu')(X)
    Y_SSloc = Dense(3, name='SS_loc', activation='elu')(X)

    model = Model(inputs=Input_X, outputs=[Y_TISprob, Y_SSprob, Y_TISloc, Y_SSloc])

    return model

def TISRover1mod():
    # Shared Layers
    Input_X = Input(shape=(600, 4))
    X = Conv1D(70, 7)(Input_X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Dropout(0.2)(X)
    X = Conv1D(100, 3)(X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Dropout(0.2)(X)
    X = Conv1D(150, 3)(X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Dropout(0.2)(X)
    X = Flatten()(X)

    # Task Specific Layers
    # TIS probability
    X_TISprob = Dense(512)(X)
    X_TISprob = Dropout(0.2)(X_TISprob)
    Y_TISprob = Dense(1, name='TIS_prob', activation='sigmoid')(X_TISprob)

    X_SSprob = Dense(512)(X)
    X_SSprob = Dropout(0.2)(X_SSprob)
    Y_SSprob = Dense(1, name='SS_prob', activation='sigmoid')(X_SSprob)

    X_TISloc = Dense(512)(X)
    X_TISloc = Dropout(0.2)(X_TISloc)
    Y_TISloc = Dense(1, name='TIS_loc', activation='elu')(X_TISloc)

    X_SSloc = Dense(512)(X)
    X_SSloc = Dropout(0.2)(X_SSloc)
    Y_SSloc = Dense(3, name='SS_loc', activation='elu')(X_SSloc)

    model = Model(inputs=Input_X, outputs=[Y_TISprob, Y_SSprob, Y_TISloc, Y_SSloc])

    return model

def PNetwork():
    # Input layer
    Input_X = Input(shape=(600, 4))

    X = Conv1D(100, 9, activation='relu')(Input_X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(150, 7, activation='relu')(X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(200, 7, activation='relu')(Input_X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(250, 7, activation='relu')(Input_X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Conv1D(300, 5, activation='relu')(Input_X)
    X = MaxPooling1D(pool_size=3)(X)
    X = Flatten()(X)

    X_TIS = Dense(256)(X)
    # X_TIS = Dropout(0.5)(X_TIS)
    X_TIS = Dense(256)(X_TIS)
    # X_TIS = Dropout(0.3)(X_TIS)

    X_SS = Dense(256)(X)
    # X_SS = Dropout(0.5)(X_SS)
    X_SS = Dense(256)(X_SS)
    # X_SS = Dropout(0.3)(X_SS)

    Y_TISprob = Dense(1, name='TIS_prob', activation='sigmoid')(X_TIS)
    Y_SSprob = Dense(1, name='SS_prob', activation='sigmoid')(X_SS)
    Y_TISloc = Dense(1, name='TIS_loc', activation='elu')(X_TIS)
    Y_SSloc = Dense(3, name='SS_loc', activation='elu')(X_SS)

    model = Model(inputs=Input_X, outputs=[Y_TISprob, Y_SSprob, Y_TISloc, Y_SSloc])

    return model


def ZNetwork():
    # Input layer
    Input_X = Input(shape=(600, 4))

    # Shared layers
    X = Conv1D(64, 9, activation='elu')(Input_X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(pool_size=3, strides=3)(X)
    X = Conv1D(64, 7, activation='elu')(X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(pool_size=3, strides=3)(X)
    X = Conv1D(64, 7, activation='elu')(X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(pool_size=3, strides=3)(X)
    X = Flatten()(X)
    X = Dense(256)(X)
    X = Dense(128)(X)

    # Task specific layers - TIS
    X_TIS = Dense(64)(X)
    X_TIS = Dense(32)(X_TIS)
    Y_TIS1 = Dense(1, activation="sigmoid", name='TIS_prob')(X_TIS)
    Y_TIS2 = Dense(1, activation="elu", name='TIS_loc')(X_TIS)

    # Task specific layers - SS
    X_SS = Dense(64)(X)
    X_SS = Dense(32)(X_SS)
    Y_SS1 = Dense(1, activation="sigmoid", name='SS_prob')(X_SS)
    Y_SS2 = Dense(3, activation="elu", name='SS_loc')(X_SS)
    
    # CAT = Concatenate()([Y_TIS1, Y_SS1, Y_TIS2, Y_SS2])

    model = Model(inputs = Input_X, outputs = [Y_TIS1, Y_SS1, Y_TIS2, Y_SS2])

    # model = Model(inputs = Input_X, outputs = [Y_TIS1, Y_SS1, Y_TIS2, Y_SS2])

    return model

def ZNetwork2():
    # Input layer
    Input_X = Input(shape=(600, 4))

    # Shared layers
    X = Conv1D(64, 9, activation='elu')(Input_X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(pool_size=3, strides=3)(X)
    X = Conv1D(64, 7, activation='elu')(X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(pool_size=3, strides=3)(X)
    X = Conv1D(64, 7, activation='elu')(X)
    X = Dropout(0.2)(X)
    X = MaxPooling1D(pool_size=3, strides=3)(X)
    X = Flatten()(X)
    X = Dense(256)(X)
    X = Dense(128)(X)

    # Task specific layers - TIS
    X_TIS = Dense(64)(X)
    X_TIS = Dense(32)(X_TIS)
    Y_TIS1 = Dense(1, activation="sigmoid", name="TIS_prob")(X_TIS)
    Y_TIS2 = Dense(1, activation="elu", name="TIS_loc")(X_TIS)

    # Task specific layers - SS
    X_SS = Dense(64)(X)
    X_SS = Dense(32)(X_SS)
    Y_SS1 = Dense(1, activation="sigmoid", name="SS_prob")(X_SS)
    Y_SS2 = Dense(3, activation="elu", name="SS_loc")(X_SS)

    # CAT = Concatenate()([Y_TIS1, Y_SS1, Y_TIS2, Y_SS2])

    # model = Model(inputs = Input_X, outputs = CAT)

    model = Model(inputs = Input_X, outputs = [Y_TIS1, Y_SS1, Y_TIS2, Y_SS2])

    return model


