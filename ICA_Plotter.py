import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import ICA_support_lib as sup

class plotter:
    def __init__(self,VER, MODEL):
        self.main_Path = os.getcwd()
        self.ising_model_Path = self.main_Path + '/Ising_models/'
        self.data_Path = self.main_Path + '/dataFiles/ver_'+str(VER)+'/'

        self.VER = VER
        self.M = MODEL


    def extract_data(self,):

        filename = 'ICA_Data_ver_' + str(self.VER)
        names, data = sup.unpack_file(filename=filename, dataPath=self.data_Path)
        spin_rate_per_t_save = data[names.index('spin_rate_per_t_save')]
        t_schd = data[names.index('t_schd')]
        ca_list = data[names.index('ca_list')]
        ms_ising = data[names.index('ms_ising')]
        mag_list = data[names.index('mag_list')]
        temp_list = data[names.index('temp_list')]

        return spin_rate_per_t_save, t_schd, ca_list, ms_ising, mag_list, temp_list


    def plot_spin_rates(self,ising_rates_mat,temps):

        fig = plt.figure(1,figsize=(15,8))
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)

        fig.suptitle('VER: ' + str(self.VER) + '_MODEL: ' + str(self.M))

        T1_rates = np.average(ising_rates_mat[0:9, :, :], axis=0)
        T2_rates = np.average(ising_rates_mat[9:18, :, :], axis=0)

        AX0 = ax0.imshow(T1_rates,cmap=cm.RdGy,vmin=0,vmax=20)
        AX1 = ax1.imshow(T2_rates,cmap=cm.RdGy,vmin=0,vmax=20)
        cbar = fig.colorbar(AX0,ax=[ax0,ax1],orientation='horizontal')
        cbar.ax.set_xlabel('Rate (Hz)')

        ax0.set_title('Synaptic Rates at T=' + str(temps[0]))
        ax1.set_title('Synaptic Rates at T=' + str(temps[-1]))

        fig_name = 'Synaptic Rates.png'
        fig_fn = os.path.abspath(os.path.join(self.data_Path, fig_name))
        fig.savefig(fig_fn)



        fig1 = plt.figure(2,figsize=(15,4))
        ax2 = fig1.add_subplot(121)
        ax3 = fig1.add_subplot(122)

        fig1.suptitle('VER: ' + str(self.VER) + '_MODEL: ' + str(self.M))

        y, bin_edges = np.histogram(T1_rates.reshape((1, -1)), bins=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                    density=True)
        bincenters = bin_edges + 1
        ax2.bar(bincenters[:-1], y, linewidth=3, color='k', width=1.25)

        ax2.set_xlim([0, 20])
        ax2.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
        ax2.set_yticks([0, 0.01, 0.1, 1])
        ax2.set_yscale('log')
        ax2.set_ylim([0.01, 1.0])

        y1, bin_edges1 = np.histogram(T2_rates.reshape((1, -1)), bins=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                      density=True)
        bincenters1 = bin_edges1 + 1
        ax3.bar(bincenters1[:-1], y1, linewidth=3, color='k', width=1.25)

        ax3.set_xlim([0, 20])
        ax3.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
        ax3.set_yticks([0, 0.01, 0.1, 1])
        ax3.set_yscale('log')
        ax3.set_ylim([0.01, 1.0])

        ax2.set_xlabel('Rate (Hz)')
        ax2.set_ylabel('Density')
        ax3.set_xlabel('Rate (Hz)')
        ax2.set_title('Synaptic Rate Distribution at T=' + str(temps[0]))
        ax3.set_title('Synaptic Rate Distribution at T=' + str(temps[-1]))

        fig1_name = 'Synaptic Rate Distributions.png'
        fig1_fn = os.path.abspath(os.path.join(self.data_Path, fig1_name))
        fig1.savefig(fig1_fn)

    def plot_astrocyte_response(self,ca_tm,t_tm,mag_tm,temp_tm):

        fig2 = plt.figure(0,figsize=(18,9))

        fig2.suptitle('VER: ' + str(self.VER) + '_MODEL: ' + str(self.M))

        ax0 = fig2.add_subplot(311)
        ax1 = fig2.add_subplot(312,sharex=ax0)
        ax2 = fig2.add_subplot(313,sharex=ax0)

        t_tm_sec = np.asarray(t_tm)/1000

        ax0.plot(t_tm_sec, ca_tm)
        ax0.set_ylim([0, 1.1])
        ax0.set_ylabel('Astrocyte\nCalcium\nConcentration')

        rate_arr = np.multiply(10.0, np.add(1.0, np.asarray(mag_tm)))

        ax1.plot(t_tm_sec, rate_arr)
        ax1.set_ylim([0,20])
        ax1.set_ylabel('Average\nSynaptic\nRate')

        ax2.plot(t_tm_sec, temp_tm)
        ax2.set_ylim([0.4, 3.6])
        ax2.set_ylabel('T')
        ax2.set_xlabel('Time (Seconds)')

        ax0.set_title('Astrocyte Calcium Concentration Oscillation Frequency Response to Synaptic Dynamics')

        fig2_name = 'Astrocyte Frequency Response.png'
        fig2_fn = os.path.abspath(os.path.join(self.data_Path, fig2_name))
        fig2.savefig(fig2_fn)



def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ["ver_num=", "ising_model_num="])
    except getopt.GetoptError:
        print('Incorrect arguments')

        sys.exit(2)

    for opt, arg in opts:
        if opt == '--ver_num':
            ver = int(arg)

        elif opt == '--ising_model_num':
            m = int(arg)

        else:
            print('Error, exiting')
            sys.exit()


    pt = plotter(VER=ver,MODEL=m)

    spin_rate_per_t_save, t_schd, ca_list, ms_ising, mag_list, temp_list = pt.extract_data()
    pt.plot_spin_rates(ising_rates_mat=spin_rate_per_t_save,temps=t_schd)
    pt.plot_astrocyte_response(ca_tm=ca_list,t_tm=ms_ising,mag_tm=mag_list,temp_tm=temp_list)

if __name__ == '__main__':
    main(sys.argv[1:])
