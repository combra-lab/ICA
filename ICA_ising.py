import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class astro_pp_model_ising:
    def __init__(self
                 , synaptic_matrix_size
                 , shrink = 1.0
                 , stretch = 1.0
                 , scaler = 1.0
                 , shift = 0.0
                 , neighbor_coupling = 1
                 , k = 1
                 , T_c = 2.26918531421
                 ):

        self.synaptic_matrix_size = synaptic_matrix_size
        self.exp_synaptic_matrix_size = (self.synaptic_matrix_size[0] + 2, self.synaptic_matrix_size[1] + 2)

        self.list1, self.list2 = self.get_gather_scatter_spins_index(self.synaptic_matrix_size)

        # PARAMETERS FOR SIMULATION ISING MODEL
        self.k = k
        self.T_c = T_c

        # PARAMETERS FOR GENERATING ISNIG COUPLINGS
        self.neighbor_coupling = -1.0 * neighbor_coupling
        self.scaler = scaler
        self.shrink = shrink
        self.stretch = stretch
        self.shift = shift


    def initialize_vars(self,initial_spins=0,initial_spin_dist=0):

        '''
        
        :param initial_spins: INITIAL SPIN STATES USED TO INITIALIZE ISING SYSTEM
        :param initial_spin_dist: CONSTANT AT 0 = UNIFORM SPIN DISTRIBUTION FOR INITIALIZATION
        
        :return: TENSORFLOW DATASTRUCTURE FOR RUNNING ISING MODEL
        '''

        ind1 = tf.constant(self.list1, dtype=tf.int32, shape=[np.shape(self.list1)[0], np.shape(self.list1)[1]])
        ind2 = tf.constant(self.list2, dtype=tf.int32, shape=[np.shape(self.list2)[0], np.shape(self.list2)[1]])

        # initial spins
        if type(initial_spins) == int:
            print('No ising spin initialization provided')
            initial_spins = np.asarray(
                ((np.random.randint(0, 2, size=(self.synaptic_matrix_size[0], self.synaptic_matrix_size[1])) * 2) - 1),
                dtype=np.float32)

        if type(initial_spin_dist) == int:
            print('No spin distribuiton provided, using uniform distribution')
            initial_spin_dist = np.ones((self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]),dtype=np.float32)


        ## initialize spin state variable
        main_spins = tf.Variable(initial_spins, dtype=tf.float32,
                                 expected_shape=[self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]],
                                 name='main_spins')


        spin_dist = tf.Variable(initial_spin_dist, dtype=tf.float32,
                                 expected_shape=[self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]],
                                 name='spin_dist')

        param_T = tf.Variable(np.ones((self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]), dtype=np.float32),
                              dtype=tf.float32,
                              expected_shape=[self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]],
                              name='param_T')

        J = tf.Variable(self.neighbor_coupling*np.ones((4,self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]), dtype=np.float32),
                              dtype=tf.float32,
                              expected_shape=[self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]],
                              name='J')

        # placeholders
        feed_temp_scalar = tf.placeholder(dtype=tf.float32, shape=[1])

        spin_feeder = tf.placeholder(dtype=tf.float32,shape=[self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]])



        return ind1, ind2, main_spins, param_T,J,spin_dist,feed_temp_scalar,spin_feeder


    def reinitialize_vars(self,ind1_list,ind2_list,main_spins_arr,param_T_arr,J_arr,spin_dist_arr,out_rescaled_arr):


        ind1 = tf.constant(ind1_list, dtype=tf.int32, shape=[np.shape(ind1_list)[0], np.shape(ind1_list)[1]])
        ind2 = tf.constant(ind2_list, dtype=tf.int32, shape=[np.shape(ind2_list)[0], np.shape(ind2_list)[1]])


        ## initialize spin state variable
        main_spins = tf.Variable(main_spins_arr, dtype=tf.float32,
                                 expected_shape=[self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]],
                                 name='main_spins')

        spin_dist = tf.Variable(spin_dist_arr, dtype=tf.float32,
                                expected_shape=[self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]],
                                name='spin_dist')

        param_T = tf.Variable(param_T_arr,
                              dtype=tf.float32,
                              expected_shape=[self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]],
                              name='param_T')

        J = tf.Variable(
            J_arr,
            dtype=tf.float32,
            expected_shape=[self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]],
            name='J')

        # placeholders
        feed_temp_scalar = tf.placeholder(dtype=tf.float32, shape=[1])

        spin_feeder = tf.placeholder(dtype=tf.float32,
                                     shape=[self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]])

        return ind1, ind2, main_spins, param_T, J, spin_dist, feed_temp_scalar, spin_feeder


    def reinitialize_spins(self,main_spins,feed_spins):

        '''
        
        :param main_spins: MATRIX WITH SPIN STATES OF ISING LATTICE
        :param feed_spins: NEW SPINS
        
        :return: NEW SPIN STATES MATRIX
        '''

        return tf.assign(main_spins,feed_spins)


    def get_magnetization(self,main_spins):

        '''
        COMPUTES MAGNETIZATION OF ISING MODEL

        :param main_spins: MATRIX WITH SPIN STATES OF ISING LATTICE
        :return: 
        '''

        return tf.reduce_mean(main_spins)

    # used
    def get_energy(self,main_spins):

        '''
        COMPUTES ENERGY OF ISING MODEL
        
        :param main_spins: MATRIX WITH SPIN STATES OF ISING LATTICE
        
        :return: SYSTEM ENERGY VALUE
        '''

        # expand main_spins matrix for periodic boundary conditions (opposing corners connect)
        exp_main_spins = tf.concat([
            tf.concat([tf.expand_dims(
                tf.expand_dims(main_spins[self.synaptic_matrix_size[0] - 1, self.synaptic_matrix_size[1] - 1], 0), 0),
                tf.expand_dims(main_spins[self.synaptic_matrix_size[0] - 1, :], 0),
                tf.expand_dims(tf.expand_dims(main_spins[self.synaptic_matrix_size[0] - 1, 0], 0), 0)], 1),
            tf.concat([tf.expand_dims(main_spins[:, self.synaptic_matrix_size[1] - 1], 1),
                       main_spins,
                       tf.expand_dims(main_spins[:, 0], 1)], 1),
            tf.concat([tf.expand_dims(tf.expand_dims(main_spins[0, self.synaptic_matrix_size[1] - 1], 0), 0),
                       tf.expand_dims(main_spins[0, :], 0),
                       tf.expand_dims(tf.expand_dims(main_spins[0, 0], 0), 0)], 1)
        ], 0)

        sum_spins_around = tf.add_n([
            exp_main_spins[0:self.exp_synaptic_matrix_size[0] - 2, 1:self.exp_synaptic_matrix_size[1] - 1],
            exp_main_spins[2:self.exp_synaptic_matrix_size[0], 1:self.exp_synaptic_matrix_size[1] - 1],
            exp_main_spins[1:self.exp_synaptic_matrix_size[0] - 1, 0:self.exp_synaptic_matrix_size[1] - 2],
            exp_main_spins[1:self.exp_synaptic_matrix_size[0] - 1, 2:self.exp_synaptic_matrix_size[1]]
        ])

        energy_cur = tf.scalar_mul(self.neighbor_coupling, tf.multiply(main_spins, sum_spins_around))

        return tf.reduce_mean(energy_cur)


    def set_temperature(self,param_T,feed_temp):

        '''
        SETS TEMPERATURE
        
        :param param_T: T VARIABLE
        :param feed_temp: NEW T
        
        :return: NEW T
        '''

        new_temp = tf.add(tf.scalar_mul(0.0, param_T), feed_temp)
        return  tf.assign(param_T, new_temp)


    def expand_spin_matrix(self, main_spins):

        '''
        EXPANDS SPIN STATE MATRIX WITH BOUNDARY IN ACCORDANCE WITH PERIODIC BOUNDARY CONDITIONS
        
        :param main_spins: MATRIX WITH SPIN STATES OF ISING LATTICE
        
        :return: MATRIX WITH EACH DIMENSTION INCREASED BY 2
        '''

        # expand main_spins matrix for periodic boundary conditions (opposing corners connect)
        exp_main_spins = tf.concat([
            tf.concat([tf.expand_dims(
                tf.expand_dims(main_spins[self.synaptic_matrix_size[0] - 1, self.synaptic_matrix_size[1] - 1], 0), 0),
                tf.expand_dims(main_spins[self.synaptic_matrix_size[0] - 1, :], 0),
                tf.expand_dims(tf.expand_dims(main_spins[self.synaptic_matrix_size[0] - 1, 0], 0), 0)], 1),
            tf.concat([tf.expand_dims(main_spins[:, self.synaptic_matrix_size[1] - 1], 1),
                       main_spins,
                       tf.expand_dims(main_spins[:, 0], 1)], 1),
            tf.concat([tf.expand_dims(tf.expand_dims(main_spins[0, self.synaptic_matrix_size[1] - 1], 0), 0),
                       tf.expand_dims(main_spins[0, :], 0),
                       tf.expand_dims(tf.expand_dims(main_spins[0, 0], 0), 0)], 1)
        ], 0)

        return exp_main_spins


    def linear_coupling_func(self,distance_metric,slope,b, power_scaling=1):

        '''
        GENERATES PRELIMINARY COUPLING VALUE J FOR EACH CONNECTION IN ISING LATTICE
        
        :param distance_metric: DISTANCE BETWEEN SPINS
        :param slope: PARAMETER FOR J COMPUTATION
        :param b: PARAMETER FOR J COMPUTATION
        :param power_scaling: PARAMETER FOR J COMPUTATION
        :return: 
        '''

        return tf.pow(tf.add(b,tf.scalar_mul(slope,distance_metric)),power_scaling)


    def compute_abs_difference(self,landscape_main,landscape_neighbor,max_dif):

        '''
        COMPUTES DIFFERENCE BETWEEN NEIGHBORING VALUES IN 2D MATRIX
        
        :param landscape_main: MATRIX WITH REAL VALUES
        :param landscape_neighbor: MATRIX WITH REAL VALUES
        :param max_dif: MAXIMUM ALLOWED DIFFERENCE
        
        :return: MATRIX OF DIFFERENCES
        '''

        return tf.clip_by_value(tf.abs(tf.subtract(landscape_main,landscape_neighbor)),0,max_dif)


    # USED
    def set_custom_coupling_v4(self,landscape,J, max_diff=2, power_scaling=1):

        '''
        COMPUTES FINAL COUPLING VALUES J FOR ISING LATTICE
        
        :param landscape: REAL SURFACE GENERATED OVER AREA OF LATTICE
        :param J: 
        :param max_diff: 
        :param power_scaling: 
        :return: 
        '''

        exp_landscape = self.expand_spin_matrix(landscape)

        ## calc difs
        # up dif
        up_dif = self.compute_abs_difference(landscape, exp_landscape[0:self.exp_synaptic_matrix_size[0] - 2,
                                                        1:self.exp_synaptic_matrix_size[1] - 1], max_diff)

        # down dif
        down_dif = self.compute_abs_difference(landscape, exp_landscape[2:self.exp_synaptic_matrix_size[0],
                                                          1:self.exp_synaptic_matrix_size[1] - 1], max_diff)

        # left dif
        left_dif = self.compute_abs_difference(landscape, exp_landscape[1:self.exp_synaptic_matrix_size[0] - 1,
                                                          0:self.exp_synaptic_matrix_size[1] - 2], max_diff)

        # right dif
        right_dif = self.compute_abs_difference(landscape, exp_landscape[1:self.exp_synaptic_matrix_size[0] - 1,
                                                           2:self.exp_synaptic_matrix_size[1]], max_diff)

        ## compute linear map
        dif_ave, dif_max, dif_min = self.get_landscape_dif_stats(landscape=landscape)

        slope = tf.divide(1,tf.subtract(dif_ave,dif_max))
        b = tf.negative(tf.multiply(slope,dif_max))

        # up
        up_coup_temp = self.linear_coupling_func(up_dif,slope=slope,b=b,power_scaling=power_scaling)
        # down
        down_coup_temp = self.linear_coupling_func(down_dif, slope=slope, b=b, power_scaling=power_scaling)
        # left
        left_coup_temp = self.linear_coupling_func(left_dif, slope=slope, b=b, power_scaling=power_scaling)
        # right
        right_coup_temp = self.linear_coupling_func(right_dif, slope=slope, b=b, power_scaling=power_scaling)

        up_coup = tf.add(J[0, :, :], tf.negative(up_coup_temp))
        down_coup = tf.add(J[1, :, :], tf.negative(down_coup_temp))
        left_coup = tf.add(J[2, :, :], tf.negative(left_coup_temp))
        right_coup = tf.add(J[3, :, :], tf.negative(right_coup_temp))

        new_J = tf.concat(
            [tf.expand_dims(up_coup, axis=0), tf.expand_dims(down_coup, axis=0), tf.expand_dims(left_coup, axis=0),
             tf.expand_dims(right_coup, axis=0)], axis=0)

        mean = tf.reduce_mean(new_J)
        new2_J = tf.multiply(new_J,tf.divide(-1.0,mean))
        new3_J = tf.subtract(tf.abs(new2_J),2)
        mean2 = tf.reduce_mean(new3_J)
        new4_J = tf.multiply(new3_J,tf.divide(-1.0,mean2))

        ### add skew
        mean2_2 = tf.reduce_mean(new4_J)
        adj_J = tf.subtract(new4_J,mean2_2+self.shift)
        skewed1_J = self.shrink*tf.clip_by_value(adj_J,-1,0) # shrink below -1 negs
        skewed2_J = self.stretch*tf.clip_by_value(adj_J,0,1) # stretch above -1 negs
        newS_J = tf.add(skewed1_J,skewed2_J)
        meanS = tf.reduce_mean(newS_J)
        newS1_J = tf.subtract(newS_J,meanS-self.neighbor_coupling)


        new5_J = tf.scalar_mul(self.scaler,newS1_J)
        mean3 = tf.reduce_mean(new5_J)
        new6_J = tf.add((-1.0-mean3),new5_J)

        return tf.assign(J, new6_J)


    def get_landscape_dif_stats(self,landscape):

        '''
        EVALUATES DIFFERENCES IN SURFACE BETWEEN NEIGHBORING SPIN LOCATIONS
        
        :param landscape: SURFACE MATRIX
        
        :return: DIFFERENCE STATS
        '''

        exp_landscape = self.expand_spin_matrix(landscape)

        up_dif = self.compute_abs_difference(landscape, exp_landscape[0:self.exp_synaptic_matrix_size[0] - 2,
                                                        1:self.exp_synaptic_matrix_size[1] - 1],max_dif=1.0)
        down_dif = self.compute_abs_difference(landscape, exp_landscape[2:self.exp_synaptic_matrix_size[0],
                                                          1:self.exp_synaptic_matrix_size[1] - 1],max_dif=1.0)
        left_dif = self.compute_abs_difference(landscape, exp_landscape[1:self.exp_synaptic_matrix_size[0] - 1,
                                                          0:self.exp_synaptic_matrix_size[1] - 2],max_dif=1.0)
        right_dif = self.compute_abs_difference(landscape, exp_landscape[1:self.exp_synaptic_matrix_size[0] - 1,
                                                           2:self.exp_synaptic_matrix_size[1]],max_dif=1.0)

        dif_ave = tf.reduce_mean(tf.stack(
            [tf.reduce_mean(up_dif), tf.reduce_mean(down_dif), tf.reduce_mean(left_dif), tf.reduce_mean(right_dif)]))
        dif_max = tf.reduce_max(tf.stack(
            [tf.reduce_max(up_dif), tf.reduce_max(down_dif), tf.reduce_max(left_dif), tf.reduce_max(right_dif)]))
        dif_min = tf.reduce_min(tf.stack(
            [tf.reduce_min(up_dif), tf.reduce_min(down_dif), tf.reduce_min(left_dif), tf.reduce_min(right_dif)]))


        return dif_ave,dif_max,dif_min


    def pre_update_computations(self, main_spins, J):

        '''
        
        :param main_spins: MATRIX WITH SPIN STATES OF ISING LATTICE
        :param J: COUPLING MATRIX
        
        :return: CHANGE IN ENERGY IF SPINS ARE FLIPPED
        '''

        exp_main_spins = self.expand_spin_matrix(main_spins)

        energy_cur = tf.add_n([
            tf.multiply(J[0,:,:],tf.multiply(main_spins,exp_main_spins[0:self.exp_synaptic_matrix_size[0] - 2, 1:self.exp_synaptic_matrix_size[1] - 1])),
            tf.multiply(J[1, :, :],tf.multiply(main_spins,exp_main_spins[2:self.exp_synaptic_matrix_size[0], 1:self.exp_synaptic_matrix_size[1] - 1])),
            tf.multiply(J[2, :, :],tf.multiply(main_spins,exp_main_spins[1:self.exp_synaptic_matrix_size[0] - 1, 0:self.exp_synaptic_matrix_size[1] - 2])),
            tf.multiply(J[3, :, :],tf.multiply(main_spins,exp_main_spins[1:self.exp_synaptic_matrix_size[0] - 1, 2:self.exp_synaptic_matrix_size[1]]))
        ])

        energy_fin = tf.add_n([
            tf.multiply(J[0, :, :], tf.multiply(tf.negative(main_spins), exp_main_spins[0:self.exp_synaptic_matrix_size[0] - 2,
                                                            1:self.exp_synaptic_matrix_size[1] - 1])),
            tf.multiply(J[1, :, :], tf.multiply(tf.negative(main_spins), exp_main_spins[2:self.exp_synaptic_matrix_size[0],
                                                            1:self.exp_synaptic_matrix_size[1] - 1])),
            tf.multiply(J[2, :, :], tf.multiply(tf.negative(main_spins), exp_main_spins[1:self.exp_synaptic_matrix_size[0] - 1,
                                                            0:self.exp_synaptic_matrix_size[1] - 2])),
            tf.multiply(J[3, :, :], tf.multiply(tf.negative(main_spins), exp_main_spins[1:self.exp_synaptic_matrix_size[0] - 1,
                                                            2:self.exp_synaptic_matrix_size[1]]))
        ])

        energy_net = tf.subtract(energy_fin, energy_cur)

        return energy_cur,energy_fin,energy_net


    def update_main_spins_1(self, ind1, main_spins, param_T, energy_net):

        '''
        
        :param ind1: INDICES OF 1/2 SET OF SPINS THAT ARE NOT NEIGHBORING TO BE UPDATED
        :param main_spins: MATRIX WITH SPIN STATES OF ISING LATTICE
        :param param_T: TEMPERATURE
        :param energy_net: CHANGE IN ENERGY
        
        :return: UPDATED SPIN STATES FOR 1/2 SET OF SPINS
        '''

        # list 1 compression
        compressed_energy_1 = tf.gather_nd(energy_net, ind1)
        compressed_spins_1 = tf.gather_nd(main_spins, ind1)
        compressed_T_1 = tf.gather_nd(param_T, ind1)

        # list 1
        compressed_decision_matrix_1 = tf.floor(tf.clip_by_value(compressed_energy_1, -1, 0)) + 1.0

        compressed_final_decision_matrix_1 = tf.subtract(
            tf.scalar_mul(2.0,
                          tf.multiply(
                              tf.add(
                                  1.0,
                                  tf.floor(
                                      tf.clip_by_value(
                                          tf.subtract(
                                              tf.random_uniform(shape=[np.shape(self.list1)[0]]),
                                              tf.exp(tf.divide(tf.negative(compressed_energy_1),
                                                               tf.scalar_mul(self.k, compressed_T_1)))), -1, 0))),
                              compressed_decision_matrix_1)),
            1.0)

        final_decision_matrix_1 = tf.add(1.0, tf.scalar_mul(2.0, tf.clip_by_value(
            tf.scatter_nd(ind1, compressed_final_decision_matrix_1,
                          shape=[self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]]), -1,
            0)))

        new_spin_state_1 = tf.multiply(main_spins, final_decision_matrix_1)

        return tf.assign(main_spins, new_spin_state_1)


    def update_main_spins_2(self, ind2, main_spins, param_T, energy_net):

        '''

        :param ind2: INDICES OF 2/2 SET OF SPINS THAT ARE NOT NEIGHBORING TO BE UPDATED
        :param main_spins: MATRIX WITH SPIN STATES OF ISING LATTICE
        :param param_T: TEMPERATURE
        :param energy_net: CHANGE IN ENERGY

        :return: UPDATED SPIN STATES FOR 2/2 SET OF SPINS
        '''

        # list 2 compression
        compressed_energy_2 = tf.gather_nd(energy_net, ind2)
        compressed_spins_2 = tf.gather_nd(main_spins, ind2)
        compressed_T_2 = tf.gather_nd(param_T, ind2)

        # list 2
        compressed_decision_matrix_2 = tf.floor(tf.clip_by_value(compressed_energy_2, -1, 0)) + 1.0

        compressed_final_decision_matrix_2 = tf.subtract(
            tf.scalar_mul(2.0,
                          tf.multiply(
                              tf.add(
                                  1.0,
                                  tf.floor(
                                      tf.clip_by_value(
                                          tf.subtract(
                                              tf.random_uniform(shape=[np.shape(self.list2)[0]]),
                                              tf.exp(tf.divide(tf.negative(compressed_energy_2),
                                                               tf.scalar_mul(self.k, compressed_T_2)))), -1, 0))),
                              compressed_decision_matrix_2)),
            1.0)

        final_decision_matrix_2 = tf.add(1.0, tf.scalar_mul(2.0, tf.clip_by_value(
            tf.scatter_nd(ind2, compressed_final_decision_matrix_2,
                          shape=[self.synaptic_matrix_size[0], self.synaptic_matrix_size[1]]), -1,
            0)))

        new_spin_state_2 = tf.multiply(main_spins, final_decision_matrix_2)

        return tf.assign(main_spins, new_spin_state_2)


    def get_gather_scatter_spins_index(self, synaptic_matrix_size):

        '''
        ASSEMBLES INDICES OF 2 SETS OF NON-NEIGHBORING SPINS
        
        :param synaptic_matrix_size: SHAPE OF ISING LATTICE
        
        :return: INDICES
        '''

        ## create spin lists for gather functions
        spin_list_1 = []
        spin_list_2 = []
        for r in range(0, synaptic_matrix_size[0]):
            for c in range(0, synaptic_matrix_size[1]):
                if r % 2 == 0 and c % 2 == 0:
                    spin_list_1.append([r, c])
                elif r % 2 == 1 and c % 2 == 1:
                    spin_list_1.append([r, c])
                elif r % 2 == 0 and c % 2 == 1:
                    spin_list_2.append([r, c])
                elif r % 2 == 1 and c % 2 == 0:
                    spin_list_2.append([r, c])

        return spin_list_1, spin_list_2
