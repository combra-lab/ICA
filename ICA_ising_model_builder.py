import sys, getopt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import ICA_support_lib as sup
import ICA_coupling_pattern as cp
import ICA_ising as ising


class astro_pp_ising_creator:
    def __init__(self):

        self.main_Path = os.getcwd()
        self.ising_model_Path = self.main_Path + '/Ising_models/'
        self.data_Path = self.main_Path + '/dataFiles/'

        sup.check_create_save_dir(self.ising_model_Path)


    def create_ising_model(self, ising_num, size_exp,shift,shrink,stretch,scaler,clusters = 500,diam_mean_var = (6, 1),amp_mean_var = (.1, 1)):

        '''
        
        :param ising_num: ISING MODEL NUMBER TO GENERATE
        :param size_exp: CONTROLS SIZE OF ISING LATTICE
        :param shift: SURFACE GENERATION PARAMETER, CONSTANT AT 0
        :param shrink: SURFACE GENERATION PARAMETER, CONSTANT AT 0
        :param stretch: SURFACE GENERATION PARAMETER, CONSTANT AT 0
        :param scaler: SURFACE GENERATION PARAMETER, CONSTANT AT 0
        :param clusters: SURFACE GENERATION PARAMETER, CONSTANT AT 500
        :param diam_mean_var: MEAN AND VARIANCE FOR THE DIAMETER OF EACH RADIAL BASIS FUNCTION FORMING THE SURFACE
        :param amp_mean_var: MEAN AND VARIANCE FOR THE AMPLITUDE OF EACH RADIAL BASIS FUNCTION FORMING THE SURFACE
        
        :return: SAVES ISING MODEL DATA
        '''


        np.random.seed(222)

        # CREATE MODEL DIRECTORY
        dataPath_model = self.ising_model_Path + '/Model_' + str(ising_num)
        sup.check_create_save_dir(dataPath_model)

        # SET SHAPE OF ISING 2D LATTICE
        shp = (2 ** size_exp, 2 ** size_exp)  # syn_space_size

        # INITIALIZED NEED CLASSES
        isi = ising.astro_pp_model_ising(synaptic_matrix_size=shp,shift=shift,shrink=shrink,stretch=stretch,scaler=scaler)
        pat = cp.astro_pp_pattern_generator(space_dims=shp)

        # CREATE LOG FOR MODEL PARAMETERS AND STATS
        log_filename = 'Log_for_Ising_Model_' + str(ising_num)
        log_fn = os.path.abspath(os.path.join(dataPath_model, log_filename))
        with open(log_fn, 'w') as f:
            f.write('LOG___ISING_MODEL_' + str(ising_num)+ '\n\n')
            f.write('DATA PATH: ' + str(dataPath_model) + '\n\n\n')

            f.write('INPUT PARAMETERS:\n\n')
            f.write('       size_exp = ' + str(size_exp) + '\n')
            f.write('          shape = ' + str(shp) + '\n\n')
            f.write('       clusters = ' + str(clusters) + '\n')
            f.write('  diam_mean_var = ' + str(diam_mean_var) + '\n')
            f.write('   amp_mean_var = ' + str(amp_mean_var) + '\n')
            f.write('          shift = ' + str(shift) + '\n')
            f.write('         shrink = ' + str(shrink) + '\n')
            f.write('        stretch = ' + str(stretch) + '\n')
            f.write('         scaler = ' + str(scaler) + '\n')

            # GENERATE 3D LANDSCAPE USING RADIAL BASIS FUNCTIONS
            params = pat.generate_pattern_landscape_parameters_normal_dist(num_of_clusters=clusters,
                                                                                diam_min_max=diam_mean_var,
                                                                                amp_min_max=amp_mean_var)
            out = pat.space_func_2d(pat.X, pat.Y, params[0], params[1], params[2], params[3])

            f.write('Initial Out Landscape <M>, Min, Max : ' + str(
                len(np.where(out >= 0)[0]) / np.size(out)) + ' , ' + str(np.amin(out)) + ' , ' + str(
                np.amax(out)) + '\n')

            # RESCALING SURFACE
            out_rescaled = np.multiply(out, np.divide(1.0, np.maximum(np.absolute(np.amin(out)),
                                                                      np.absolute(np.amax(out)))))

            f.write('Initial Out_rescaled Landscape <M>, Min, Max : ' + str(
                len(np.where(out_rescaled >= 0)[0]) / np.size(out_rescaled)) + ' , ' + str(
                np.amin(out_rescaled)) + ' , '
                    + str(np.amax(out_rescaled)) + '\n\n')

            # BINARIZE LANDSCAPE WITH THRESHOLD AT 0
            spins_matrix_rescaled = np.add(1, np.multiply(2, np.floor(np.clip(out_rescaled, -1, 0))))

            f.write(
                'spin Initialization <M> = ' + str(np.average(np.clip(spins_matrix_rescaled, 0, 1))) + '\n\n')

            # INITIALIZE VARIABLES FOR ISING MODEL GENERATION USING SURFACE AND SPINS DATA FROM ABOVE
            ind1, ind2, main_spins, param_T, J, spin_dist, feed_temp_scalar, spin_feeder = isi.initialize_vars(
                initial_spins=spins_matrix_rescaled, initial_spin_dist=out_rescaled)

            tf.set_random_seed(1234)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            # GENERATE ISING COUPLINGS
            sess.run(isi.set_custom_coupling_v4(landscape=spin_dist, J=J, power_scaling=10))

            # COLLECT STATS ON COUPLINGS
            J_out = sess.run(J)

            f.write('Ising Coupling Metrics:\n')
            f.write('   Average: ' + str(np.average(J_out)) + '\n')
            f.write('   Maximum: ' + str(np.amax(J_out)) + '\n')
            f.write('   Minimum: ' + str(np.amin(J_out)) + '\n\n')

            #### SAVING ISING MODEL ####
            data_name = 'ISING_Model_' + str(ising_num)

            sup.save_non_tf_data(  names = ['ind1','ind2','main_spins','param_T','J','spin_dist','out_rescaled']
                                 , data = [isi.list1, isi.list2, sess.run(main_spins), sess.run(param_T), sess.run(J), sess.run(spin_dist), out_rescaled]
                                 , filename = data_name
                                 , savePath = dataPath_model
                                 )

            print('Ising Model number ' + str(ising_num) + ' saved.')
            sess.close()



def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ["model_num="])
    except getopt.GetoptError:
        print('Incorrect arguments')

        sys.exit(2)

    for opt, arg in opts:
        if opt == '--model_num':
            m = int(arg)

        else:
            print('Error, exiting')
            sys.exit()


    aic = astro_pp_ising_creator()
    aic.create_ising_model(m,8,shift=0.0,shrink=0.35,stretch=4.0,scaler=1.0,clusters=500,diam_mean_var=(6,1))


if __name__ == '__main__':
    main(sys.argv[1:])

















