# Astrocytes learn to detect and signal deviations from critical brain dynamics

This package is a Tensorflow implementation of the astrocyte calcium response to changes in neural activity dynamics around a critical phase transition. 

The paper has been accepted at Neural Computation, available [_here_](https://).

# Citation

Vladimir A. Ivanov and Konstantinos P. Michmizos. "Astrocytes learn to detect and signal deviations from critical brain dynamics." *Neural Computation* (2022).

	@article{ivanov_2022,
	author = {Ivanov, Vladimir A. and Michmizos, Konstantinos P.},
	title = {Astrocytes learn to detect and signal deviations from critical brain dynamics},
	year = {2022},
	journal = {Neural Computation}
	}
  
## Software Installation

* Python 3.6.9
* Tensorflow 2.1 (with CUDA 11.2 using tensorflow.compat.v1)
* Numpy

## Usage

This code performs the following functions:
1. [Generate the attractor network (Ising model) with clustered couplings](#1-generate-attractor-network)
2. [Evaluate astrocyte Calcium response to change in neural dynamics](#2-evaluate-astrocyte-calcium-response)
3. [Visualize simulation output](#3-visualize-output)

### 1. Generate attractor network
To generate the attractor network (Ising model) with clustered couplings, enter the following command:

       python ICA_ising_model_builder.py

with inputs:
* `--model_num` : Integer assigned to model created.

This will create a file formatted as `ISING_Model_X` in subfolder '`/Ising_models/Model_X/`, where `X` is the model number set by input `--model_num`.

### 2. Evaluate astrocyte calcium response

To run a simulation with biophysical astrocyte driven by neural activity with dynamics transitioning from `T1` to `T2`, enter the following command:

      python ICA.py
      
with inputs:
* `--ver_num`		:	Integer assigned to simulation run.
* `--ising_model_num`	:	Integer assigned to Iisng model used.
* `--t1`		: 	Float represeting start dynamics of simulation. Astrocyte weights are initialized on neural activity from these dynamics. 
* `--t2`		:	  Float represeting ending dynamics of simulation. This controls the dynamics to which the Ising model transitions to with astrocyte already initialized on `t1`.

This will create a file formatted as `ICA_Data_ver_X` in subfolder '`/dataFiles/ver_X/`, where `X` is the simulation version number set by input `--ver_num`.

### 3. Visualize output

To visualize the output from the simulation file in the last step, enter the following command:

      python ICA_Plotter.py
      
with inputs:
* `--ver_num` 		:	Simulation verion number from which to plot.
* `--ising_model_num`	:	Integer assigned to Iisng model used.
		
This will create the following image files in subfolder `/dataFiles/ver_X` where `X` is the simulation number.
* `Astrocyte Frequency Response.png` : shows time series of astrocyte calcium concentration response to Ising synaptic activity transition from `t1` to `t2`.
* `Synaptic Rate Distributions.png` : shows distribution of synaptic rates for `t1` and `t2`.
* `Synaptic Rates.png` : shows a 2D visualization of synaptic rates.

### Other files

`ICA_ising.py` : Contains class for creating and running Ising model.

`ICA_astrocyte.py` : Contains class for initializing and running astrocyte model.

`ICA_coupling_pattern.py` : Contains class for generating surface over spin coordinates used for creating Ising model.

`ICA_support_lib.py` : Contains set of house keeping support functions.

### Example commands

        python ICA_ising_model_builder.py --model_num=1

        python ICA.py --ver_num=10 --ising_model_num=1 --t1=2.2 --t2=3.2

        python ICA_Plotter.py --ver_num=10 --ising_model_num=1   





Create an Ising model with clustered couplings using run file:
	* 'ICA_ising_model_builder.py'
		
2. Launch ICA simulation using the created Ising model using run file:
	* 'ICA.py'
		
3. Generate visualize outputs from simulation using run file:
	* 'ICA_Plotter.py'
