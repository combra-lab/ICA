import sys, getopt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import os
import ICA_support_lib as sup
import ICA_astrocyte as astr_astro
import ICA_ising as astr_ising


class astro_pp_model:
    def __init__(self, num_of_astros=1):


        self.main_Path = os.getcwd()
        self.ising_model_Path = self.main_Path + '/Ising_models/'
        self.data_Path = self.main_Path + '/dataFiles/'

        sup.check_create_save_dir(self.data_Path)

        self.astro_num = num_of_astros


    def open_ising_network(self, model):

        # OPENS ISING MODEL DATA

        filename = 'Model_'+str(model)+'/ISING_Model_' + str(model)

        names, data = sup.unpack_file(filename,self.ising_model_Path)

        ind1 = data[names.index('ind1')]
        ind2 = data[names.index('ind2')]
        main_spins = data[names.index('main_spins')]
        param_T = data[names.index('param_T')]
        J = data[names.index('J')]
        spin_dist = data[names.index('spin_dist')]
        out_rescaled = data[names.index('out_rescaled')]

        return [ind1, ind2, main_spins, param_T, J, spin_dist, out_rescaled]


    ############# VERSION FOR MULTI RUN, CAN FEED CUSTOM T_SCHE IN
    def test_ising_w_patterns_w_astro_v5_5(self, model, save_ver, t_schd_in, ms_per_temp=10000, ini_dur_ms=60000, t_slots=18,
                                         max_rate=20, v_beta_scale_factor=[3.0, 3.0], randomize_glut_amount=False,
                                         glut_level=[1.0, 1.0]):

        assert(len(glut_level) == len(v_beta_scale_factor) == self.astro_num)

        # -SEEDSET- SET RANDOM SEED
        np.random.seed()
        rnd_sd = np.random.randint(0, 1000)
        np.random.seed(rnd_sd)
        ######################## SET SEED MANUALLY ######################
        # np.random.seed(121)
        ######################## SET SEED MANUALLY ######################

        dataPath_ver = self.data_Path + 'ver_' + str(save_ver)
        sup.check_create_save_dir(dataPath_ver)
        print(dataPath_ver)

        ### START -- OPEN ISING MODEL ###
        md = self.open_ising_network(model)
        shp = np.shape(md[2])

        isi = astr_ising.astro_pp_model_ising(synaptic_matrix_size=shp)

        ind1, ind2, main_spins, param_T, J, spin_dist, feed_temp_scalar, spin_feeder = isi.reinitialize_vars(
            ind1_list=md[0], ind2_list=md[1], main_spins_arr=md[2], param_T_arr=md[3], J_arr=md[4], spin_dist_arr=md[5],
            out_rescaled_arr=md[6])

        # ISING UPDATE OPS
        energy_cur, energy_fin, energy_net = isi.pre_update_computations(main_spins=main_spins, J=J)
        update_spin_1 = isi.update_main_spins_1(ind1=ind1, main_spins=main_spins, param_T=param_T,
                                                energy_net=energy_net)
        update_spin_2 = isi.update_main_spins_2(ind2=ind2, main_spins=main_spins, param_T=param_T,
                                                energy_net=energy_net)
        mag = isi.get_magnetization(main_spins)
        e = isi.get_energy(main_spins)

        ### END -- OPEN ISING MODEL ###

        t_schd  = t_schd_in
        assert (len(t_schd) == t_slots)

        # CREATE LOG FILE FOR RECORDING SIM OUTPUTS/PARAMETERS
        log_filename = 'Data_Log_ver_' + str(save_ver)
        log_fn = os.path.abspath(os.path.join(dataPath_ver, log_filename))
        with open(log_fn, 'w') as f:

            ## COMPUTE ms BETWEEN SPIKES BASED ON ASTROCYTE TIME
            min_stim_period = int(1000 / max_rate)
            ## COMPUTE NUMBER OF ISING UPDATES PER TEMP BASED ON ASTROCYTE TIME
            ising_updates_per_temp = int(ms_per_temp / min_stim_period)
            ising_updates_per_ini = int(ini_dur_ms / min_stim_period)

            if ising_updates_per_temp * min_stim_period != ms_per_temp:
                ms_per_temp = min_stim_period * ising_updates_per_temp
            if ising_updates_per_ini * min_stim_period != ini_dur_ms:
                ini_dur_ms = min_stim_period * ising_updates_per_ini

            f.write('LOG____v_' + str(save_ver) + '\n\n')
            f.write('MODEL_______' + str(model) + '\n\n')
            f.write('DATA PATH: ' + str(dataPath_ver) + '\n\n\n')
            f.write('------------------------------------------------------------\n\n')
            f.write('INPUT PARAMETERS:\n\n')
            f.write('   ms_per_temp = ' + str(ms_per_temp) + '\n')
            f.write('   ini_dur_ini = ' + str(ini_dur_ms) + '\n')
            f.write('   t_slots = ' + str(t_slots) + '\n')
            f.write('   max_rate' + str(max_rate) + '\n')
            f.write('   v_beta_scale_factor = ' + str(v_beta_scale_factor) + '\n')
            f.write('   randomize_glut_amount = ' + str(randomize_glut_amount) + '\n')
            f.write('   glut_level = ' + str(glut_level) + '\n')
            f.write('\n\n')

            f.write('RUNTIME PARAMETERS AND STATS:\n\n')
            f.write('   Max Rate: ' + str(max_rate) + '\n')
            f.write('   min_stim_period: ' + str(min_stim_period) + '\n')
            f.write('   ising_updates_per_temp: ' + str(ising_updates_per_temp) + '\n')
            f.write('   ising_updates_per_ini: ' + str(ising_updates_per_ini) + '\n\n')

            f.write('Temp Schedule: \n')
            f.write(str(t_schd) + '\n\n')

            assert (ising_updates_per_temp * min_stim_period == ms_per_temp)
            assert (ising_updates_per_ini * min_stim_period == ini_dur_ms)


            #### INITIALIZE GCH-I MODEL ##### START
            astro = astr_astro.gpu_astro(num_astros=self.astro_num,
                                              max_syns_per_astro=shp[0] * shp[1],
                                              input_morph_matrix=np.ones(
                                                  (self.astro_num, shp[0] * shp[1]),
                                                  dtype=np.float32))


            ip3_state, ca_state, h_state, ip3_store, ca_store = astro.initialize_vars()
            syn_input_feed, input_morph_feed, v_beta_feed = astro.initialize_ph()

            # state update graph
            new_ca_var_states = astro.run_ca_state_transition(L1_ip3_state=ip3_state, L1_ca_state=ca_state,
                                                                   L1_h_state=h_state)
            store_ca_var_states = tf.assign(ca_store, new_ca_var_states)

            new_h_var_states = astro.run_h_state_transition(L1_ip3_state=ip3_state, L1_ca_state=ca_state,
                                                                 L1_h_state=h_state)
            update_h_var_states = tf.assign(h_state, new_h_var_states)

            new_ip3_var_states = astro.run_ip3_state_transition(L1_ip3_state=ip3_state, L1_ca_state=ca_state,
                                                                     syn_inp=syn_input_feed,
                                                                     v_beta_var=v_beta_feed,
                                                                     input_morph=input_morph_feed)
            update_ip3_var_states = tf.assign(ip3_state, new_ip3_var_states)

            update_ca_var_states = tf.assign(ca_state, ca_store)

            rst_ip_state = astro.reset_var(var=ip3_state,new_val_scalar=astro.ip3_init)
            rst_ca_state = astro.reset_var(var=ca_state, new_val_scalar=astro.ca_init)
            rst_h_state = astro.reset_var(var=h_state, new_val_scalar=astro.h_init)
            #### INITIALIZE GCH-I MODEL ##### END



            # -SEEDSET- SET RANDOM SEED FOR TF OPS
            # tf.set_random_seed(1234)

            ### INITIALIZE TF SESSION ###
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            f.write('------------------------------------------------------\n\n\n')

            # INITIALIZE SAVE LISTS
            mag_list = []
            e_list = []
            temp_list = []
            ca_list = []
            ising_mat_save = np.zeros((t_slots, shp[0], shp[1]), dtype=np.float32)
            spin_rate_per_t_save = np.zeros(np.shape(ising_mat_save), dtype=np.float32)
            mag_per_temp_list = []
            ms_temp = []
            ms_ising = []
            ms_time_main = []

            # INITIALIZING RANDOM SPIN STATE
            print('Setting Random Initial Spin Configuration')
            out_rescaled = sess.run(spin_dist)
            spins_matrix_rescaled1 = np.add(1, np.multiply(2, np.floor(np.clip(out_rescaled, -1, 0))))
            spins_matrix_rescaled = np.subtract(np.multiply(2, np.random.randint(0, 2, size=(np.shape(spins_matrix_rescaled1)[0], np.shape(spins_matrix_rescaled1)[1]))), 1)
            sess.run(isi.reinitialize_spins(main_spins=main_spins, feed_spins=spins_matrix_rescaled))

            ### INITIALIZATION LOOP ###
            # set temp for loop iteration
            sess.run(isi.set_temperature(param_T=param_T, feed_temp=t_schd[0]))

            print('Starting Initialization at T = ' + str(t_schd[0]))

            f.write('INITIALIZATION STATS:\n\n')
            f.write('Initialization Temperature: ' + str(t_schd[0]) + '\n')

            ### LETTING ISING SYSTEM STABALIZE
            for k in range(0,10000):
                sess.run(update_spin_1)
                sess.run(update_spin_2)


            ### GETTING INITIALIZATION RATES
            mag_ini_ave = []
            temp_spin_agg = np.zeros(shp, dtype=np.float32)
            for i in range(0, ising_updates_per_ini):
                ## Ising Spin Updates
                sess.run(update_spin_1)
                sess.run(update_spin_2)
    
                out_spins = sess.run(main_spins)
                temp_spin_agg += np.clip(out_spins, 0, 1)
                mag_ini_ave.append(sess.run(mag))

            print('Initialization Complete. Stats: ')

            # COMPUTING INITIALIZATION RATES USED TO INITIALIZE GCH-I v_beta's
            rates_raw = np.divide(temp_spin_agg, ini_dur_ms / 1000)
            rates_ini = self.format_rates(rates_raw, 0.2, max_rate)

            f.write('Sum Spin Stats: \n')
            f.write('   Average: ' + str(np.average(temp_spin_agg)) + '\n')
            f.write('   Maximum: ' + str(np.amax(temp_spin_agg)) + '\n')
            f.write('   Minimum: ' + str(np.amin(temp_spin_agg)) + '\n\n')

            f.write('Spin Rates_Raw Stats: \n')
            f.write('   Average: ' + str(np.average(rates_raw)) + '\n')
            f.write('   Maximum: ' + str(np.amax(rates_raw)) + '\n')
            f.write('   Minimum: ' + str(np.amin(rates_raw)) + '\n\n')

            f.write('Spin Rates Stats: \n')
            f.write('   Average: ' + str(np.average(rates_ini)) + '\n')
            f.write('   Maximum: ' + str(np.amax(rates_ini)) + '\n')
            f.write('   Minimum: ' + str(np.amin(rates_ini)) + '\n\n')


            ### INITIALIZATION OF V_BETA WITH GLUT_LEVEL ###
            for i in range(0,len(v_beta_scale_factor)):
                f.write('v_beta for SCALING FACTOR: ' + str(v_beta_scale_factor[i]) + ' GLUT_LEVEL: ' + str(glut_level[i]) + '\n\n')

                v_beta = self.rates_to_v_beta(rates_ini, scaling=v_beta_scale_factor[i],shp=shp)
                print('v_beta Ave, Min, Max: ', np.average(v_beta), np.amin(v_beta), np.amax(v_beta))

                f.write('v_beta Stats: \n')
                f.write('   Average: ' + str(np.average(v_beta)) + '\n')
                f.write('   Maximum: ' + str(np.amax(v_beta)) + '\n')
                f.write('   Minimum: ' + str(np.amin(v_beta)) + '\n\n')


                if i == 0:
                    ## PREPARE V_BETA FOR INPUT INTO GPU ##

                    v_beta_f = v_beta.reshape(1, -1)
                    rates_ini_f = rates_ini.reshape(1, -1)

                else:

                    v_beta_f = np.concatenate([v_beta_f, v_beta.reshape(1, -1)],axis=0)
                    rates_ini_f = np.concatenate([rates_ini_f, rates_ini.reshape(1, -1)],axis=0)

            ##### REFORMAT ASTRO INPUTS
            v_beta_f_rsc = np.divide(v_beta_f,1000)
            zero_input = np.zeros(np.shape(v_beta_f), dtype=np.float32)
            morph = np.ones(np.shape(v_beta_f), dtype=np.float32)
            glut_level_weights = np.broadcast_to(np.asarray(glut_level).reshape(-1,1),shape=np.shape(v_beta_f))

            f.write('-------------------------------------------------\n\n\n')
            f.write('SIMULATION OUTPUT AND RESULTS: \n\n')

            # T LOOP
            for i in range(0, len(t_schd)):

                print('Launching T' + str(i) + '/' + str(len(t_schd)) + ': ' + str(t_schd[i]))
                f.write('Launching T' + str(i) + '/' + str(len(t_schd)) + ': ' + str(t_schd[i]) + '\n')

                # set temp for loop iteration
                sess.run(isi.set_temperature(param_T=param_T, feed_temp=t_schd[i]))

                # RESETS GCH-I STATE VARIABLES IF T CHANGES
                if i > 0:
                    if np.absolute(t_schd[i] - t_schd[i-1]) > 0:

                        sess.run(rst_ca_state)
                        sess.run(rst_ip_state)
                        sess.run(rst_h_state)

                st = time.time()

                mag_total_per_temp = 0
                temp_spin_agg = np.multiply(temp_spin_agg, 0.0)
                # ISING LOOP
                for t in range(0, ising_updates_per_temp):

                    ## Ising Spin Updates
                    sess.run(update_spin_1)
                    sess.run(update_spin_2)

                    # GET AND REFORMAT SPINS
                    out_spins = sess.run(main_spins)
                    out_spins_fmt = np.clip(out_spins, 0, 1)
                    out_spins_f = np.broadcast_to(out_spins_fmt.reshape(1, -1), shape=np.shape(glut_level_weights))
                    out_spins_f2 = np.multiply(glut_level_weights,out_spins_f)

                    # ASTRO LOOP
                    for a in range(0, min_stim_period):

                        # COMPUTE CURRENT ASTROCYTE TIME
                        cur_ms = (i * ising_updates_per_temp * min_stim_period) + (t * min_stim_period) + a
                        ms_time_main.append(cur_ms)

                        if a == 0:

                            ### UPDATE ASTRO WITH ISING INPUT
                            sess.run(store_ca_var_states)
                            sess.run(update_h_var_states)
                            sess.run(update_ip3_var_states, feed_dict={syn_input_feed: out_spins_f2,
                                                                       v_beta_feed: v_beta_f_rsc,
                                                                       input_morph_feed: morph})
                            sess.run(update_ca_var_states)
                        else:
                            ### UPDATE ASTRO WITHOUT ISING INPUT
                            sess.run(store_ca_var_states)
                            sess.run(update_h_var_states)
                            sess.run(update_ip3_var_states,
                                     feed_dict={syn_input_feed: zero_input,
                                                v_beta_feed: v_beta_f_rsc,
                                                input_morph_feed: morph})
                            sess.run(update_ca_var_states)

                    # RECORD OUTPUTS
                    mag_list.append(sess.run(mag))
                    mag_total_per_temp += sess.run(mag)
                    e_list.append(sess.run(e))
                    temp_list.append(t_schd[i])
                    ms_ising.append(cur_ms)
                    temp_spin_agg += out_spins_fmt
                    ca_list.append(sess.run(ca_state))

                mag_per_temp_list.append(np.divide(mag_total_per_temp, ising_updates_per_temp))

                # COMPUTE AND RECORD SPIN FLIP RATES
                rates = np.divide(temp_spin_agg, ms_per_temp / 1000)
                spin_rate_per_t_save[i, :, :] = rates
                ms_temp.append(cur_ms)


                print('T_' + str(t_schd[i]) + '_Complete_in_' + str(time.time() - st) + '_sec')
                f.write('T_' + str(t_schd[i]) + '_Complete_in_' + str(time.time() - st) + '_sec\n')
            sess.close()


        data_name = 'ICA_Data_ver_' + str(save_ver)
        sup.save_non_tf_data(names = ['t_schd','mag_list','temp_list','ms_ising','ca_list','spin_rate_per_t_save','e_list']
                             , data = [t_schd,mag_list,temp_list,ms_ising,ca_list,spin_rate_per_t_save,e_list]
                             , filename=data_name
                             , savePath=dataPath_ver
                             )

        print('File  ' + str(data_name) + ' saved.')


    def rates_to_v_beta(self,rates,scaling,shp):

        '''
        
        :param rates: SPIN FLIP RATES COMPUTED BASED ON ASTROCYTE TIME
        :param scaling: MULTIPLICATIVE SCALING FACTOR 
        :param shp: SHAPE OF LATTICE
        :return: 
        '''

        p_hist_mat = np.divide(rates,1000)
        v_mat_scaled = np.multiply(np.divide(1,scaling),np.divide(1, p_hist_mat))

        return np.clip(np.divide(v_mat_scaled, shp[0] * shp[1]),0.0004,0.13)


    def format_rates(self,rates,min_rate_hz,max_rate_hz):

        # CLIPS RATE TO min_rate_hz AND max_rate_hz

        return np.clip(rates, min_rate_hz, max_rate_hz)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ["ver_num=", "ising_model_num=", "t1=", "t2="])
    except getopt.GetoptError:
        print('Incorrect arguments')

        sys.exit(2)

    for opt, arg in opts:
        if opt == '--ver_num':
            ver = int(arg)

        elif opt == '--ising_model_num':
            m = int(arg)

        elif opt == '--t1':
            t1 = float(arg)

        elif opt == '--t2':
            t2 = float(arg)

        else:
            print('Error, exiting')
            sys.exit()

    ####### UNCOMMENT IF USING GPU #######
    # GPU_NUM = input('GPU? ')
    #
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_NUM)
    # print('GPU: ' + str(os.environ["CUDA_VISIBLE_DEVICES"]))



    v_beta_scale_factor_list = [5.5]
    glut_list = [0.2]

    vbsfl = []
    gl = []
    for i in range(0, len(v_beta_scale_factor_list)):
        for j in range(0, len(glut_list)):
            vbsfl.append(v_beta_scale_factor_list[i])
            gl.append(glut_list[j])
    print(vbsfl)
    print(gl)

    t_list = [t1,t2]

    t_slots = 18
    t_schd = np.concatenate([t_list[0] * np.ones(int(t_slots / 2)), t_list[1] * np.ones(int(t_slots / 2))],
                            axis=0)  # just two temps

    am = astro_pp_model(num_of_astros=len(vbsfl))
    am.test_ising_w_patterns_w_astro_v5_5(model=m, save_ver=ver, v_beta_scale_factor=vbsfl,
                                          t_schd_in=t_schd, t_slots=t_slots, randomize_glut_amount=False,
                                          ms_per_temp=15000, max_rate=20, glut_level=gl)

if __name__ == '__main__':
    main(sys.argv[1:])
