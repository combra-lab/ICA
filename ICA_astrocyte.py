import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


class gpu_astro:
    def __init__(self, num_astros, max_syns_per_astro,input_morph_matrix
        , glutamate_per_synapse = 0.2                               # glutamate concentration
        , ip3_init = 0.148465                                       # steady state GCh-I value
        , C_init = 0.0698025                                        # steady state GCh-I value
        , h_init = 0.793086                                         # steady state GCh-I value
    ):


        self.num_astros = num_astros                                # compatible with only 1 astrocyte
        self.syns_per_astro = max_syns_per_astro
        self.shp = num_astros
        self.input_shp = (num_astros,max_syns_per_astro)

        # INITIAL GCH-I MODEL STATE VARIABLE VALUES
        self.ip3_init = ip3_init
        self.ca_init = C_init
        self.h_init = h_init

        # -------------- GCH-I MODEL PARAMETERS ---------------
        self.r_C = 6.0
        self.r_L = 0.11
        self.C_0 = 2.0
        self.c_1 = 0.185
        self.K_ER = 0.05                                        # set to FM value
        self.d_1 = 0.13
        self.d_2 = 1.049
        self.d_3 = 0.9434
        self.d_5 = 0.08234
        self.v_delta = 0.05                                     # set to FM value
        self.K_PLC_delta = 0.1
        self.k_delta = 1.5
        self.v_ER = 0.9
        self.r_5P = 0.05                                        # set to FM value
        self.v_3K = 2.0
        self.K_D = 0.7
        self.K_3 = 1.0
        self.a_2 = 0.2
        self.K_R = 1.3
        self.K_P = 10.0
        self.K_pi = 0.6

        # RESCALING RATE VARIABLES FROM SECONDS TO MILLISECONDS
        self.r_C = self.r_C * (1 / 1000)
        self.r_L = self.r_L * (1 / 1000)
        self.v_ER = self.v_ER * (1 / 1000)
        self.a_2 = self.a_2 * (1 / 1000)
        self.v_delta = self.v_delta * (1 / 1000)
        self.r_5P = self.r_5P * (1 / 1000)
        self.v_3K = self.v_3K * (1 / 1000)

        self.glutamate_per_syn = glutamate_per_synapse

        # PRECOMPUTED VALUES FOR GCH-I SIMULATION
        self.c_1_plus_1 = 1.0 + self.c_1


    def initialize_vars(self):

        # INITIALIZE TENSORFLOW DATA STRUCTURES FOR GCH-I

        ip3_state = tf.Variable(self.ip3_init * np.ones(self.shp), dtype=tf.float32,
                                   expected_shape=self.shp)
        ip3_store = tf.Variable(np.ones(self.shp), dtype=tf.float32,
                                   expected_shape=self.shp)
        ca_state = tf.Variable(self.ca_init * np.ones(self.shp), dtype=tf.float32,
                                  expected_shape=self.shp)
        ca_store = tf.Variable(np.ones(self.shp), dtype=tf.float32, expected_shape=self.shp)

        h_state = tf.Variable(self.h_init * np.ones(self.shp), dtype=tf.float32,
                                 expected_shape=self.shp)

        return ip3_state,ca_state,h_state,ip3_store,ca_store


    def initialize_ph(self):

        syn_input_feed = tf.placeholder(dtype=tf.float32,shape=self.input_shp)
        input_morph_feed = tf.placeholder(dtype=tf.float32,shape=self.input_shp)
        v_beta_feed = tf.placeholder(dtype=tf.float32,shape=self.input_shp)

        return syn_input_feed,input_morph_feed,v_beta_feed

    def reset_var(self,var,new_val_scalar):

        return tf.assign(var,tf.add(new_val_scalar,tf.scalar_mul(0.0,var)))


    def run_ip3_state_transition(self, L1_ip3_state, L1_ca_state, syn_inp,v_beta_var,input_morph):

        '''

        :param L1_ip3_state: IP3 CONCENTRATION OF PREVIOUS ms TIMESTEP 
        :param L1_ca_state: CA CONCENTRATION OF PREVIOUS ms TIMESTEP
        :param syn_inp: GLUTAMATE CONCENTRATION AT CURRENT ms TIMESTEP FOR SET OF SYNAPTIC INPUTS
        :param v_beta_var: GLUTAMATE RECEPTOR DENSITY (WEIGHT) OF EACH SYNAPTIC INPUT FOR SET OF SYNAPTIC INPUTS
        :param input_morph: MASK FOR EXISTING TRIPARTITE CONNECTIONS

        :return: :return: COMPUTES [IP3] FOR CURRENT ms TIMESTEP
        '''

        delta_ip3 = tf.add_n([
            tf.reduce_sum(tf.multiply(input_morph,tf.multiply(v_beta_var,
                          tf.divide(tf.pow(syn_inp, 0.7), tf.add(tf.pow(syn_inp, 0.7),
                                                                 tf.tile(tf.expand_dims(tf.pow(
                              tf.add(self.K_R,
                                     tf.scalar_mul(self.K_P,
                                                   tf.divide(L1_ca_state, tf.add(L1_ca_state, self.K_pi)))),
                              0.7),axis=1),[1,self.syns_per_astro]))))),axis=1),
            tf.multiply(tf.divide(self.v_delta, tf.add(tf.divide(L1_ip3_state, self.k_delta), 1.0)),
                        tf.divide(tf.pow(L1_ca_state, 2.0),
                                  tf.add(tf.pow(L1_ca_state, 2.0), tf.pow(self.K_PLC_delta, 2.0)))),
            tf.scalar_mul(tf.negative(self.v_3K),
                          tf.multiply(tf.divide(tf.pow(L1_ca_state, 4.0),
                                                tf.add(tf.pow(L1_ca_state, 4.0), tf.pow(self.K_D, 4.0))),
                                      tf.divide(L1_ip3_state, tf.add(L1_ip3_state, self.K_3)))),
            tf.scalar_mul(tf.negative(self.r_5P), L1_ip3_state)])

        return tf.add(L1_ip3_state, delta_ip3)


    def run_ca_state_transition(self, L1_ip3_state, L1_ca_state, L1_h_state):

        '''

        :param L1_ip3_state: IP3 CONCENTRATION OF PREVIOUS ms TIMESTEP 
        :param L1_ca_state: CA CONCENTRATION OF PREVIOUS ms TIMESTEP
        :param L1_h_state: H VALUE OF PREVIOUS ms TIMESTEP

        :return: COMPUTES [CA] FOR CURRENT ms TIMESTEP
        '''

        delta_ca = tf.add_n([tf.multiply(self.r_C, tf.multiply(tf.pow(tf.divide(L1_ip3_state,
                                                                                  tf.add(L1_ip3_state,
                                                                                         self.d_1)), 3.0),
                                                                 tf.multiply(tf.pow(tf.divide(L1_ca_state,
                                                                                              tf.add(L1_ca_state,
                                                                                                     self.d_5)),
                                                                                    3.0),
                                                                             tf.multiply(tf.pow(L1_h_state
                                                                                                , 3.0),
                                                                                         tf.subtract(self.C_0,
                                                                                                     tf.scalar_mul(
                                                                                                         self.c_1_plus_1,
                                                                                                         L1_ca_state
                                                                                                         )))))),
                             tf.scalar_mul(self.r_L,
                                           tf.subtract(self.C_0, tf.scalar_mul(self.c_1_plus_1, L1_ca_state
                                                                               ))),
                             tf.multiply(tf.negative(self.v_ER), tf.divide(tf.pow(L1_ca_state
                                                                           , 2.0), tf.add(tf.pow(L1_ca_state
                                                                                                 , 2.0),
                                                                                          tf.pow(self.K_ER,
                                                                                                 2.0))))])

        return tf.add(L1_ca_state, delta_ca)


    def run_h_state_transition(self, L1_ip3_state, L1_ca_state, L1_h_state):

        '''

        :param L1_ip3_state: IP3 CONCENTRATION OF PREVIOUS ms TIMESTEP 
        :param L1_ca_state: CA CONCENTRATION OF PREVIOUS ms TIMESTEP
        :param L1_h_state: H VALUE OF PREVIOUS ms TIMESTEP

        :return: :return: COMPUTES H FOR CURRENT ms TIMESTEP
        '''

        Q2_L1 = tf.scalar_mul(self.d_2, tf.divide(tf.add(L1_ip3_state, self.d_1), tf.add(L1_ip3_state, self.d_3)))

        delta_h = tf.divide(tf.subtract(tf.divide(Q2_L1, tf.add(Q2_L1, L1_ca_state)), L1_h_state),
                            tf.divide(1.0, tf.scalar_mul(self.a_2, tf.add(Q2_L1, L1_ca_state))))

        return tf.add(L1_h_state, delta_h)


