# For Cooper's Conflicting Constraint Checking, we can either add the limits and stuff in the loss definitions or we can do it at a separate time. I should set up some limits first. 
# Also, where/when do we want to start testing for conflicts? Can it just be right away (TM0)? I think it can... I'll just set up uninformed dummy numbers right now #$

#################################################################
# Code      Heat Generation and Transfer via PINN
# Version   2.1
# Date      2025-11-15
# Author    Cooper Evans, Dan Humfeld
# Note      This code solves the heat equation with convective BC
#           The external air temperature is co-solved
#
#################################################################
# Importing Libraries
#################################################################
#edit
import os
# Suppress C++-level warnings:
# 0 = all messages are logged (default)
# 1 = INFO messages are filtered out
# 2 = INFO and WARNING messages are filtered out
# 3 = INFO, WARNING, and ERROR messages are filtered out
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import random
import copy
from turtle import shape
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.keras import backend as k
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from time import time
start_time = time()

print(tf.config.list_physical_devices('GPU'))

#################################################################
# Inputs 
#################################################################   
LoCaTA_active = True
transfer_mode = 3
#try:
#    transfer_mode = int(sys.argv[1])
#except:
#    pass

# Training Mode: Train new = 0, continue training = 1, only load models = anything greater
train_mode = 1
#try:
#    train_mode = int(sys.argv[2])
#except:
#    pass

# Epoch and batch size control
# epochs = 2000
batch = 8192 #8000 at first, then 24000 before fine-tuning #$
# batch = 32000 #8000 at first, then 24000 before fine-tuning
# This totally belongs somewhere else! 
# Each increment, how much lower the cure duration should be than its previous increment, and the temperature at which the cure cycle is declared complete
duration_increment_factor = 0.99
T_end_C = 65.556 # 120 F "max safe temp to touch according to some standards"
# 21.852 = 65.556/(180-20) * (80-20) ####CHANGE ONCE YOU FIND THE CORRECT TEMP

# Path for model files
path_for_models = 'WorkingModelsv22/'
path_for_materials = 'Materials'

# Optimizer parameters
# learning_rate = 0.0000001 #CJE pulling this out for TM1 at 1500 epochs. It seems to have made no progress since 500 epochs
# learning_rate = 0.001 #CJE Transfer and Training Mode 0 only took ~500 epochs to get down to a loss of 1e1
# learning_rate = 0.000001 #CJE, guess based on top learning rate and comments below
learning_rate = 0.00005 #VTN CJE 1k epochs mode 0, 2k epochs mode 1,
# learning_rate = 0.0005 #VTN-2, higher learning rate makes learning TM1, TM2, and TM3 harder. Lower learning rate makes TM0 harder
# learning_rate = 0.00005 #VTN-3, higher learning rate makes learning TM1, TM2, and TM3 harder. Lower learning rate makes TM0 harder


initializer = 'glorot_uniform'

# Model hyper-parameters
dimensions = 2
nodes_per_layer = 64 #MTU
model_layers = 5 #MTU

# Operating options
plot_loss = False
time_reporting = True

# FEM results file names for Transfer Mode 0
data_file_part_T = 'WorkingModelsv22/FEM Inputs/FEM part T.csv'
data_file_tool_T = 'WorkingModelsv22/FEM Inputs/FEM tool T.csv'
data_file_air_T = 'WorkingModelsv22/FEM Inputs/FEM air T.csv'
data_file_part_DOC = 'WorkingModelsv22/FEM Inputs/FEM part DOC.csv'
# cooper
#data_file_blanket_T = 'WorkingModelsv22/FEM Inputs/FEM blanket T.csv'

#################################################################
# Transfer Mode Definitions
#################################################################
transfer_mode_definition = [
    # question: do I need to make another equation list for the blanket? I think yes. I probably also need to make an FEM blanket T. Am I gonna screw up
    # indexing everywhere if I add it to the middle? **NO becasue dictionary (yay)** Should I add it to the end or do it how it should be done? ###VTN do it how it should be done.
    {  # transfer mode 0 recommend TM 0 to be just heat equation with 1e-3 conduction. Then TM 1 is adding conduction weight to 1e-1. TM 2 adds DOC info
        'part_equation_list': ['FEM part T','FEM part DOC'],
        'tool_equation_list': ['FEM tool T'],
        'air_equation_list': ['FEM air T'],
        # cooper
        #'blanket_equation_list': ['FEM blanket T'],
        # cooper
        'loss_weightings_master': {'FEM part T': 1, 'FEM tool T': 1,
                                   'FEM air T': 1, #'FEM blanket T': 1,
                                   'FEM part DOC': 1e4},
        'non-trainable': [ 'dT_dt', 'dT_dx', 'ddoc_dt']},

    {  # transfer mode 1 recommend TM 0 to be just heat equation with 1e-3 conduction. Then TM 1 is adding conduction weight to 1e-1. TM 2 adds DOC info
        'part_equation_list': ['heat equation', 'constraint_T'],
        'tool_equation_list': ['FEM tool T'],
        'air_equation_list': ['FEM air T'],
        # cooper
        #'blanket_equation_list': ['FEM blanket T'],
        #question: I probably need to know where all the conductions and convections are so I know what to change.
        # CHECK HERE
        'loss_weightings_master': {'heat eq': 1e0, 'T_2':1e0,
                                   'conduction_2': 1e0,
                                   'convection_1': 1e0, 'convection_2': 1e0, 'convection_3': 1e0,
                                   'FEM air T': 1e0, 'FEM blanket T': 1e0},
        'non-trainable': ['doc', 'dT_dt', 'dT_dx', 'ddoc_dt']},

    {   # transfer mode 2, interrupt training to adjust weights. Conduction eventually goes to 1e0
        'part_equation_list': ['coupled heat equation','cure kinetics'], #VTN
        # 'part_equation_list': ['coupled heat equation', 'cure kinetics', 'constraint_doc_soft'],
        'tool_equation_list': ['FEM tool T'],
        'air_equation_list': ['FEM air T'],
        # cooper
        # 'blanket_equation_list': ['FEM blanket T'],
        # question: I probably need to know where all the conductions and convections are so I know what to change.
        'loss_weightings_master': { 'heat eq': 1e0,'coupled heat eq':2e0, 'doc_pde': 2e3,'constraint_doc_soft': 1e9,'T_2':1e0,
                                   'convection_1': 1e0, 'convection_2': 1e0,
                                   'doc_pde_lin': 1e4, 'doc_pde_log':1e0},
        'non-trainable': ['dT_dt', 'dT_dx', 'ddoc_dt']},

    {   # transfer mode 3, interrupt training to adjust weights. Conduction eventually goes to 1e0
        'part_equation_list': ['coupled heat equation','cure kinetics','constraint_doc_soft','constraint_T', 'minimize_cure_duration'],
        'tool_equation_list': ['constraint_A_dT_dt', 'constraint_A_T'], #, 'minimize_cure_duration'
        # 'air_equation_list': ['constraint_A_dT_dt','constraint_A_T'],
        #VTN, you can change to 'air_equation_list': ['FEM air T'],
        #'air_equation_list': ['constraint_A_dT_dt', 'constraint_A_T', 'minimize cure duration'],
        'air_equation_list': ['FEM air T'],
        # cooper
        # question: thoughts?
        # 'blanket_equation_list': ['constraint_A_dT_dt', 'constraint_A_T', 'minimize cure duration'],
# question: I'll have to change some of this too. (?)

#        'loss_weightings_master': {'doc_pde_log': 0e0, 'doc_pde_lin': 4e4,'constraint_doc_soft': 1e3, 'heat eq': 1e0, 'coupled heat eq': 2e1, # doc_pde originally 4e3, move to 1.5e6
#        # 'loss_weightings_master': {'doc_pde_log': 0e0, 'doc_pde_lin': 5e3,'constraint_doc_soft': 1e9, 'heat eq': 1e0, 'coupled heat eq': 2e0, # doc_pde originally 4e3
#                                   'T_1':1e1, 'convection_2': 1e0,
#                                   #'convection_1': 1e0, 'conduction_2': 1e0, 'convection_2': 1e0, 'convection_3': 1e0,
#                                   # 'cure_duration':5e1,
#                                   'cure_duration': 1e0, #CJE Turned off for now - minimizes how long it takes to cure.
#                                   'constraint_A_T': 1e0, #CJE Turned off for now
#                                    'constraint_A_dT_dt': 1e0 * 0, #CJE Turn off next
#                                    'FEM air T': 1e0,
#                                    'constraint_T': 1e0 #CJE Turn off next next
#                                   }, #CJE 2/18/2025 weightings

#'loss_weightings_master': {'doc_pde_log': 0e0, 'doc_pde_lin': 1.5e4,'constraint_doc_soft': 1e6, 'heat eq': 1e0, 'coupled heat eq': 2e0, # constraint_doc_soft originially 1e9
#        # 'loss_weightings_master': {'doc_pde_log': 0e0, 'doc_pde_lin': 5e3,'constraint_doc_soft': 1e9, 'heat eq': 1e0, 'coupled heat eq': 2e0, # doc_pde originally 4e3
#                                   'T_2':1e0, 'convection_1': 1e0, 'conduction_2': 1e0, 'convection_2': 1e0, 'convection_3': 1e0,
#                                   # 'cure_duration':5e1,
#                                   'cure_duration': 2e4,
#                                   'constraint_A_T':1e2,
#                                   },
#                                    #CJE old (2024) loss weightings

        'loss_weightings_master': {'FEM air T': 1e0, 'doc_pde_log': 0e0, 'doc_pde_lin': 1e4,'constraint_doc_soft': 1e4, 'coupled heat eq': 2e0, # constraint_doc_soft originially 1e9
        # 'loss_weightings_master': {'doc_pde_log': 0e0, 'doc_pde_lin': 5e3,'constraint_doc_soft': 1e9, 'heat eq': 1e0, 'coupled heat eq': 2e0, # doc_pde originally 4e3
                                   'convection_1': 1e1, 'convection_2': 1e1, 'constraint_T': 1e0,
                                   # 'cure_duration':5e1,
                                   'cure_duration': 2e4, # raw value is in hours (jk lol)
                                   'constraint_A_T': 1e-3, # raw value is in ___
                                   'constraint_A_dT_dt': 1e-3,
                                   },

        'non-trainable': ['dT_dt', 'dT_dx', 'ddoc_dt']}

]

#################################################################
# Classes
#################################################################
class Network:
    '''
    Defines the set of networks that will be used.
    To create a Network, specify the name, primary activation function and final activation function
    Creating an instance assigns the file names, makes the model and saves the model
    '''
    def __init__(self, name, component_number, dimensions, layers, nodes, primary_activation, final_activation, transfer_mode, train_mode, path_for_models):
        self.name = name
        self._primary_activation = primary_activation
        self._final_activation = final_activation
        self._dimensions = dimensions
        self._output_model_file_name = path_for_models + 'mode_' + str(transfer_mode) + '_c' + str(component_number) + '_' + self.name + '.keras'
        if ((train_mode == 0) and (transfer_mode > 0)):
            self.input_model_transfer_mode = transfer_mode - 1
        else:
            self.input_model_transfer_mode = transfer_mode
        self._input_model_file_name = path_for_models + 'mode_' + str(self.input_model_transfer_mode) + '_c' + str(component_number) + '_' + self.name + '.keras'
        # Make or load model
        self.make_model(self._dimensions, layers, nodes)
        if ((transfer_mode > 0) or (train_mode >= 1)):
            self.load_model() ###Marker2
        self.save_model()
        #self.model.summary()

    def make_model(self, dimensions, layers, nodes):
        model_inputs = [keras.layers.Input(shape=(1,)) for input_variable in range(dimensions)]
        if (dimensions == 1):
            layered_buildup = model_inputs[0]
        else:
            layered_buildup = keras.layers.concatenate(model_inputs)
        for layer_number in range(layers-1):
            # layered_buildup = keras.layers.Dense(nodes, activation = self._primary_activation, kernel_initializer = initializer, bias_initializer = initializer)(layered_buildup)
            layered_buildup = keras.layers.Dense(nodes, kernel_regularizer=regularizers.l2(1e-3), activation = self._primary_activation, kernel_initializer = initializer, bias_initializer = initializer)(layered_buildup) ###a
        layered_buildup = keras.layers.Dense(1, activation = self._final_activation, kernel_initializer = initializer, bias_initializer = initializer)(layered_buildup)
        self.model = keras.models.Model(model_inputs, [layered_buildup])
        self.model_optimizer = keras.optimizers.Adam(learning_rate)
        self.model.compile(loss = 'mse', optimizer = self.model_optimizer)
        self.model.optimizer.build(self.model.trainable_variables)
    
    def load_model(self):
        #print(self._input_model_file_name)
        self.model = keras.models.load_model(self._input_model_file_name) ###Marker1
        self.model_optimizer = keras.optimizers.Adam(learning_rate) #self.model.optimizer - 12/16 OLD CE
        #print("completed ",self._input_model_file_name)

    def save_model(self):
        self.model.save(self._output_model_file_name)

    def values(self, inputs_lists):
        if (self._dimensions == len(inputs_lists)):
            return self.model(inputs_lists)
        elif (self._dimensions == 1) and (len(inputs_lists) > 1):
            return(self.model(inputs_lists[-1]))
        else:
            return None

class Material:
    ''' Class to hold material properties '''
    def __init__(self, name):
        self.name = name
        self.determine_file_name()
        self.load_properties()
        self.derive_properties()

    def determine_file_name(self):
        self.file_name = path_for_materials + '/material_' + self.name + '.txt'

    def load_properties(self):
        # There is no file type / format defined for these. For the moment keeping them hard-coded but this should be improved soon
        if (self.name == 'Air' or 'Tool'):
            self.properties = {'phase': 'fluid'}

        if (self.name == 'Aluminum'):
            self.properties = {'phase': 'solid'}
            self.properties['k'] = 160 *1e2*(3600*3600*3600)         #MTU (kg*m/s^2*m/s)/mK -> (kg*mm/h^2*mm/h)/mmK
            self.properties['rho'] = 2770 *1e-2*1e-2*1e-2       #MTU kg/m3 -> kg/mm3
            self.properties['cp'] = 875 *1e2*1e2*(3600*3600)         #MTU kg*m/s^2*m/(kg K) -> kg*mm/hr^2*mm/(kg K)

        if (self.name == 'Composite Complicated'): #We will use this for the project #CJE from NASA Paper with 80% fiber volume fraction
            self.properties = {'phase': 'solid'}
            self.properties['k'] = (0.204 + 0.228) / 2 * 1e2 * (3600 * 3600 * 3600)  # Thermal Conductivity (kg*m/s^2*m/s)/mK -> (kg*cm/h^2*cm/h)/cmK
            self.properties['rho'] = (1440 + 1410) / 2 * 1e-2 * 1e-2 * 1e-2  # Density kg/m3 -> kg/cm3
            self.properties['cp'] = (1057 + 876) / 2 * 1e2 * 1e2 * (3600 * 3600)  # Specific Heat kg*m/s^2*m/(kg K) -> kg*cm/hr^2*cm/(kg K)
            self.properties['resin_heat_of_reaction'] = 1.57e5 * 1e2 * 1e2 * (3600 * 3600)  # FROM POWERPOINT kg*m/s^2*m/kg -> kg*mm/h^2*mm/kg
            self.properties['resin_volume_fraction'] = 0.51  # unitless
            self.properties['ck'] = {}
            self.properties['ck']['model'] = 'Toray'  # Later should change that to "modified autocatalytic" or something
            self.properties['ck']['AA_1'] = 14240 * 3600  # MTU 1/s -> 1/hour
            self.properties['ck']['AA_2'] = 453684 * 3600  # MTU 1/s -> 1/hour
            self.properties['ck']['Ea_1'] = 66435. * 1e2 * 1e2 * (3600 * 3600)  # MTU kg*m/s^2*m/mol -> kg*cm/hr^2*cm/mol
            self.properties['ck']['Ea_2'] = 73063. * 1e2 * 1e2 * (3600 * 3600)  # MTU kg*m/s^2*m/mol -> kg*cm/hr^2*cm/mol
            self.properties['ck']['mm_1'] = 0  # unitless
            self.properties['ck']['nn_1'] = 1  # unitless
            self.properties['ck']['mm_2'] = 1  # unitless
            self.properties['ck']['nn_2'] = 2.5  # unitless
            self.properties['ck']['R'] = 8.3141 * 1e2 * 1e2 * (3600 * 3600)  # MTU kg*m/s^2*m/mol.K -> kg*mm/hr^2*mm/mol.K

            # Diffusion Term Constants THESE WILL PROBABLY HAVE TO CHANGE!!!!
            self.properties['ck']['Ad'] = 4e12 *3600                             # unitless
            self.properties['ck']['Ed'] = 6e4  *1e2*1e2*(3600*3600)                            # unitless
            self.properties['ck']['b'] = 0.52                              # unitless
            self.properties['ck']['w'] = 0.00008                              # unitless
            self.properties['ck']['g'] = 0.025                              # unitless

            self.properties['ck']['Tg0'] = 8.0+273                              # Kelvin
            self.properties['ck']['Tginf'] = 200+273                              # Kelvin
            self.properties['ck']['lam'] = 0.8                              # unitless

        if (self.name == 'Composite Insulator'):
            self.properties = {'phase': 'solid'}
            self.properties['k'] = 0.47/6 *1e3*(3600*3600*3600)         #MTU (kg*m/s^2*m/s)/mK -> (kg*mm/h^2*mm/h)/mmK
            self.properties['rho'] = 1573 *1e-9       #MTU kg/m3 -> kg/mm3
            self.properties['cp'] = 967 *1e6*(3600*3600)         #MTU kg*m/s^2*m/(kg K) -> kg*mm/hr^2*mm/(kg K)
            self.properties['resin_heat_of_reaction'] = 600*1000 *1e3*1e3*(3600*3600)         #MTU kg*m/s^2*m/kg -> kg*mm/h^2*mm/kg
            self.properties['resin_volume_fraction'] = 0.35            # unitless
            self.properties['ck'] = {}           # unitless
            self.properties['ck']['model'] = '8552'                    # Later should change that to "modified Ahrennius" or something
            self.properties['ck']['AA'] = 152800. *60*60          #MTU 1/s -> 1/hour
            self.properties['ck']['Ea'] = 66500. *1e3*1e3*(3600*3600)           #MTU kg*m/s^2*m/mol -> kg*mm/hr^2*mm/mol
            self.properties['ck']['mm'] = 0.8129
            self.properties['ck']['nn'] = 2.736
            self.properties['ck']['R'] = 8.3141 *1e3*1e3*(3600*3600)            #MTU kg*m/s^2*m/mol.K -> kg*mm/hr^2*mm/mol.K
# cooper
        if (self.name == 'Thin Blanket'):
            self.properties = {'phase': 'solid'}

        if (self.name == 'Thin Tool'):
                self.properties = {'phase': 'solid'}

    def derive_properties(self):
        # Derived / consolidated properties
        try:
            self.properties['a'] = self.properties['k'] / self.properties['rho'] / self.properties['cp']      # mm^2/h
        except KeyError:
            pass
        try:
            self.properties['b'] = self.properties['resin_heat_of_reaction'] * self.properties['resin_volume_fraction'] * self.properties['rho'] / self.properties['rho'] / self.properties['cp']  # K
        except KeyError:
            pass
        '''
        with self.properties as p:
            try:
                p['a'] = p['k'] / p['rho'] / p['cp']      # m2/s
            except KeyError:
                pass
            try:
                p['b'] = p['resin_heat_of_reaction'] * p['resin_volume_fraction'] * p['rho'] / p['rho'] / p['cp']  # K
            except KeyError:
                pass
        '''

class Component:
    ''' A Component is meant to be the extent of a body of material, e.g. a volume in 3+1D or a layer in 1+1D '''
    component_number = 0
    def __init__(self, material_name, thickness, time_min, time_max, network_definition, equation_list, transfer_mode, train_mode, path_for_models, non_trainable_list):
        self.material_name = material_name
        Component.component_number += 1
        self.component_number = Component.component_number
        self.material = Material(material_name)
        self.thickness = thickness
        self.time_min = time_min
        self.time_max = time_max
        self.normalization = [thickness, (time_max - time_min)]
        self.equation_list = equation_list
        self.network_definition = copy.deepcopy(network_definition)
        self.non_trainable_list = non_trainable_list
        self._prune_network_definition()
        self._define_network(transfer_mode, train_mode, path_for_models) ###Marker3
        self._target_duration = target_dur
        self.bias = 1  #$

    def _prune_network_definition(self):
        if self.material.properties['phase'] == 'fluid':
            self.network_list = [name for name in self.network_definition]
            for name in self.network_list:
                if (name not in ['T', 'dT_dt']):
                    del self.network_definition[name]
        if ('b' not in self.material.properties):
            try:
                del self.network_definition['doc']
                del self.network_definition['ddoc_dt']
            except KeyError:
                pass
        if (self.thickness == 0):
            for name in self.network_definition:
                self.network_definition[name]['dimensions'] -= 1
            try:
                del self.network_definition['dT_dx']
            except KeyError:
                pass
             #MTU, remove dT_dx

    def _prune_equation_list(self):
        # Not implemented
        if ('dT_dx' not in self.network_definition):
            try:
                self.equation_list.remove('heat equation')
            except KeyError:
                pass
        if ('dT_dx' not in self.network_definition):
            try:
                self.equation_list.remove('coupled heat equation')
            except KeyError:
                pass
        if ('doc' not in self.network_definition):
            try:
                self.equation_list.remove('cure_kinetics')
            except KeyError:
                pass
        if ('b' not in self.material.properties):
            if ('heat equation' in self.equation_list) and ('coupled heat equation' in self.equation_list):
                self.equation_list.remove('coupled heat equation')
            self.equation_list = ['heat equation' if equation == 'coupled heat equation' else equation for equation in self.equation_list]

    def _define_network(self, transfer_mode, train_mode, path_for_models):
        self.net = {}
        for name in self.network_definition:
            self.net[name] = Network(name, self.component_number, 
                self.network_definition[name]['dimensions'], self.network_definition[name]['layers'], self.network_definition[name]['nodes'], 
                self.network_definition[name]['activation_function'], self.network_definition[name]['activation_final'],
                transfer_mode, train_mode, path_for_models)

    def define_residual_set(self, batch):
        self.x_arr = np.random.uniform(0, self.thickness, batch) #MTU
        self.x_feed = np.column_stack((self.x_arr))
        self.x_feed = tf.Variable(self.x_feed.reshape(len(self.x_feed[0]),1), trainable=True, dtype=tf.float32)
        self.t_arr = np.random.uniform(t_min, t_max, batch) #MTU
        self.t_feed = np.column_stack((self.t_arr))
        self.t_feed = tf.Variable(self.t_feed.reshape(len(self.t_feed[0]),1), trainable=True, dtype=tf.float32)
        # Used for transfer_mode 0; different temperature cycle for cure kinetics in the non-touching problems.
        if ('cure kinetics non-touching' in self.equation_list):
            self.Tck_inf_batch = []
            for j in range (0,batch):
                self.Tck_inf_batch.append(0.75*(T_max-T_min))
            self.Tck_inf_feed = np.column_stack((self.Tck_inf_batch))
            self.Tck_inf_feed = tf.Variable(self.Tck_inf_feed.reshape(batch,1), trainable=True, dtype=tf.float32)

    #Track input to Air NN

    def define_functions(self):
        self.functions = {}
        if 'T' in self.net:
            self.functions['T_equ'] = tf.multiply(self.net['T'].values([self.x_feed, self.t_feed]), tf.tanh(30.0 * (self.t_feed-t_min)/(t_max-t_min))) + T0 * (1.0 - (self.t_feed-t_min)/(t_max-t_min)) #MTU
            # self.functions['T_equ'] = tf.multiply(self.net['T'].values([self.x_feed, self.t_feed]), one_feed) + zero_feed #MTU Comment, fix later for transition between FEM and non-FEM
        if 'doc' in self.net:
            self.functions['doc_equ'] = tf.multiply(self.net['doc'].values([self.x_feed, self.t_feed]), tf.tanh(30.0 * (self.t_feed-t_min)/(t_max-t_min))) + doc_0 * k.relu(1.0 - 4.0 * (self.t_feed-t_min)/(t_max-t_min)) #MTU
            # self.functions['doc_equ'] = tf.minimum(self.functions['doc_equ'], 0.999) #MTU shunt to 1 if greater than 1

            self.functions['doc_equ'] = tf.minimum(self.functions['doc_equ'],0.999)

        #Bypass neural networks
        if 'ddoc_dt' in self.net:
            a=1
        if 'dT_dt' in self.net:
            a = 1
        if 'dT_dx' in self.net:
            a = 1

    def define_derivative_functions(self, tape_1):
        if 'T_equ' in self.functions:
            self.functions['dT_dx'] = tape_1.gradient(self.functions['T_equ'], [self.x_feed, self.t_feed])[0]
            self.functions['dT_dx_equ'] = self.functions['dT_dx']

            self.functions['dT_dt'] = tape_1.gradient(self.functions['T_equ'], [self.x_feed, self.t_feed])[1]
            self.functions['dT_dt_equ'] = self.functions['dT_dt']

        if 'doc_equ' in self.functions:
            self.functions['ddoc_dt'] = tape_1.gradient(self.functions['doc_equ'], [self.t_feed])[0]
            self.functions['ddoc_dt_equ'] = self.functions['ddoc_dt'] #VTN

    def define_double_derivative_functions(self, tape_2):
        if self.functions['dT_dx_equ'] is not None:
            self.functions['d2T_dx2'] = tape_2.gradient(self.functions['dT_dx_equ'], [self.x_feed])[0]

    def define_trench_width(self):
        self.loss_limits[name]

    def define_body_losses(self): #$
        self.loss_lists = {}
        self.hard_constraints = {}
        self.governing_eqs = {}
        self.soft_constraints = {}
        if 'heat equation' in self.equation_list:
            self.loss_lists['heat eq'] = (k.square(tf.math.scalar_mul(self.material.properties['a'],self.functions['d2T_dx2']) - self.functions['dT_dt_equ'])) #MTU
            #self.governing_eqs['heat eq'] = k.mean(k.square(tf.math.scalar_mul(self.material.properties['a'], self.functions['d2T_dx2']) - self.functions['dT_dt_equ']))  # MTU
            self.governing_eqs['heat eq'] = k.mean(k.square(k.abs(tf.math.scalar_mul(self.material.properties['a'], self.functions['d2T_dx2']) - self.functions['dT_dt_equ'])+self.bias))
        if 'metal heat equation' in self.equation_list:
            self.loss_lists['metal heat eq'] = (k.square(self.functions['d2T_dx2'] - tf.math.scalar_mul((self.material.properties['a'])**-1,self.functions['dT_dt_equ']))) #MTU
            self.governing_eqs['metal heat eq'] = k.mean(k.square(self.functions['d2T_dx2'] - tf.math.scalar_mul((self.material.properties['a']) ** -1, self.functions['dT_dt_equ'])))  # MTU
        if 'coupled heat equation' in self.equation_list:
            # self.loss_lists['coupled heat eq'] = k.square(tf.math.scalar_mul(self.material.properties['a']*((t_max-t_min)/self.thickness**1),self.functions['d2T_dx2']) + tf.math.scalar_mul(self.material.properties['b']*(1/(T_max-T_min)),self.functions['ddoc_dt_equ']) - self.functions['dT_dt_equ'])
            self.loss_lists['coupled heat eq'] = (k.square(tf.math.scalar_mul(self.material.properties['a'], self.functions['d2T_dx2']) + tf.math.scalar_mul(self.material.properties['b'], self.functions['ddoc_dt_equ']) - tf.math.scalar_mul(1, self.functions['dT_dt_equ']))) #MTU NEED CHECK
            self.governing_eqs['coupled heat eq'] = k.mean(k.square(k.abs(tf.math.scalar_mul(self.material.properties['a'], self.functions['d2T_dx2']) + tf.math.scalar_mul(self.material.properties['b'], self.functions['ddoc_dt_equ']) - tf.math.scalar_mul(1, self.functions['dT_dt_equ']))+self.bias))
        if 'dT_dt' in self.equation_list:
            self.loss_lists['dT_dt'] = (k.square(zero_feed))  # MTU
        if 'dT_dx' in self.equation_list:
            self.loss_lists['dT_dx'] = (k.square(zero_feed))  # MTU
            # question: Can we please talk through this?
        if 'constraint_T' in self.equation_list:
            self.loss_lists['constraint_T'] = (k.square(tf.reshape(tf.convert_to_tensor([0]*batch, dtype=np.float32), [batch,1])))
            self.hard_constraints['constraint_T'] = k.mean(k.square(tf.reshape(tf.convert_to_tensor([0] * batch, dtype=np.float32), [batch, 1])))
            for inst in constraint_T:
                x_mask = tf.cast(tf.multiply(tf.reshape(((self.x_feed > inst[0]).numpy()) * 1, [batch, 1]), tf.reshape(((self.x_feed < inst[1]).numpy()) * 1, [batch, 1])),dtype = tf.float32) #MTU
                t_mask = tf.cast(tf.multiply(tf.reshape(((self.t_feed > inst[2]).numpy()) * 1, [batch, 1]), tf.reshape(((self.t_feed < inst[3]).numpy()) * 1, [batch, 1])),dtype = tf.float32) #MTU
                range_mask = k.relu(inst[4] - self.functions['T_equ']) + k.relu(self.functions['T_equ'] - inst[5])                   # 0 if inside range, linear outside of range
                weight_mask = one_feed * inst[6]
                loss_term_inst = tf.multiply(tf.multiply(tf.multiply(x_mask, t_mask), range_mask), weight_mask)
                loss_term_inst1 = k.square(loss_term_inst)
                loss_term_inst = k.square(k.abs(loss_term_inst) + self.bias) #VANISH
                self.loss_lists['constraint_T'] = self.loss_lists['constraint_T'] + loss_term_inst1
                self.hard_constraints['constraint_T'] = self.hard_constraints['constraint_T'] + loss_term_inst
            self.hard_constraints['constraint_T'] = k.mean(self.hard_constraints['constraint_T'])

        if 'constraint_T_up' in self.equation_list:
            self.loss_lists['constraint_T_up'] = tf.reshape(tf.convert_to_tensor([0]*batch, dtype=np.float32), [batch,1])
            self.hard_constraints['constraint_T_up'] = tf.reshape(tf.convert_to_tensor([0] * batch, dtype=np.float32), [batch, 1])
            for inst in constraint_T_up:
                x_mask = tf.reshape(tf.multiply(k.relu(self.x_feed - inst[0]), k.relu(inst[1] - self.x_feed)), [batch, 1])   # 0 if below x_min, 0 if above x_max, parabolic downwards inbetween
                x_mask = x_mask / (0.25 * (inst[1]-inst[0])**3)    # This is squared to normalize the range, then divided once more to make the average of the residuals more like 1, even when the constraint is levied only over a narrow range of normalized t 0 to 1.
                t_mask = tf.reshape(tf.multiply(k.relu(self.t_feed - inst[2]), k.relu(inst[3] - self.t_feed)), [batch, 1])   # 0 if below t_min, 0 if above t_max, parabolic downwards inbetween
                t_mask = t_mask / (0.25 * (inst[3]-inst[2])**3)                                                                     # This is squared to normalize the range, then divided once more to make the average of the residuals more like 1, even when the constraint is levied only over a narrow range of normalized t 0 to 1.
                range_mask = k.relu(inst[4] - self.functions['T_equ']) + k.relu(self.functions['T_equ'] - inst[5])                   # 0 if inside range, linear outside of range
                weight_mask = one_feed * inst[6]
                loss_term_inst = tf.multiply(tf.multiply(tf.multiply(x_mask, t_mask), range_mask), weight_mask) 
                loss_term_inst = k.mean(k.square(loss_term_inst))
                self.loss_lists['constraint_T_up'] = self.loss_lists['constraint_T_up'] + loss_term_inst
                self.hard_constraints['constraint_T_up'] = self.hard_constraints['constraint_T_up'] + loss_term_inst
            self.hard_constraints['constraint_T_up'] = k.mean(self.hard_constraints['constraint_T_up'])
        if 'constraint_A_T' in self.equation_list:
            self.loss_lists['constraint_A_T'] = tf.reshape(tf.convert_to_tensor([0]*batch, dtype=np.float32), [batch,1])
            self.hard_constraints['constraint_A_T'] = k.mean(k.square(tf.reshape(tf.convert_to_tensor([0] * batch, dtype=np.float32), [batch, 1])))
            count = 0
            for inst in constraint_A_T:
                # x_mask = tf.cast(tf.multiply(tf.reshape(((self.x_feed > inst[0]).numpy()) * 1, [batch, 1]), tf.reshape(((self.x_feed < inst[1]).numpy()) * 1, [batch, 1])),dtype = tf.float32) #MTU
                t_mask = tf.cast(tf.multiply(tf.reshape(((self.t_feed > inst[2]).numpy()) * 1, [batch, 1]), tf.reshape(((self.t_feed < inst[3]).numpy()) * 1, [batch, 1])),dtype = tf.float32) #MTU
                range_mask = k.relu(inst[4] - self.functions['T_equ']) + k.relu(self.functions['T_equ'] - inst[5])                   # 0 if inside range, linear outside of range
                weight_mask = one_feed * inst[6]
                loss_term_inst = tf.multiply(tf.multiply(t_mask, range_mask), weight_mask)
                loss_term_inst = k.square(k.relu(loss_term_inst) + self.bias) #VANISH
                # t_mask = tf.reshape(tf.multiply(k.relu(self.t_feed - inst[2]), k.relu(inst[3] - self.t_feed)), [batch, 1])                                  # 0 if below t_min, 0 if above t_max, parabolic downwards inbetween
                # t_mask = t_mask / (0.25 * (inst[3]-inst[2])**3) # This is squared to normalize the range, then divided once more to make the average of the residuals more like 1, even when the constraint is levied only over a narrow range of normalized t 0 to 1. Should be varied of 0 to 1 divided by the number of residuals that fall into that range.
                # range_mask = k.relu(inst[4] - self.functions['T_equ']) + k.relu(self.functions['T_equ'] - inst[5])                   # 0 if inside range, linear outside of range
                # weight_mask = one_feed * inst[6]
                # loss_term_inst = tf.multiply(tf.multiply(t_mask, range_mask), weight_mask)
                # loss_term_inst = k.square(loss_term_inst)
                self.loss_lists['constraint_A_T'] = self.loss_lists['constraint_A_T'] + loss_term_inst
                self.hard_constraints['constraint_A_T'] = self.hard_constraints['constraint_A_T'] + loss_term_inst
                count += 1
            self.hard_constraints['constraint_A_T'] = self.hard_constraints['constraint_A_T'] - ((count - 1) * self.bias**2)
            self.hard_constraints['constraint_A_T'] = k.mean(self.hard_constraints['constraint_A_T'])
            #print(self.hard_constraints['constraint_A_T'])
        if 'constraint_dT_dt' in self.equation_list:
            self.loss_lists['constraint_dT_dt'] = tf.reshape(tf.convert_to_tensor([0]*batch, dtype=np.float32), [batch,1])
            self.hard_constraints['constraint_dT_dt'] = tf.reshape(tf.convert_to_tensor([0] * batch, dtype=np.float32), [batch, 1])
            for inst in constraint_dT_dt:
                x_mask = tf.reshape(tf.multiply(k.relu(self.x_feed - inst[0]), k.relu(inst[1] - self.x_feed)), [batch, 1])   # 0 if below x_min, 0 if above x_max, parabolic downwards inbetween
                x_mask = x_mask / (0.25 * (inst[1]-inst[0])**3)    # This is squared to normalize the range, then divided once more to make the average of the residuals more like 1, even when the constraint is levied only over a narrow range of normalized t 0 to 1.
                t_mask = tf.reshape(tf.multiply(k.relu(self.t_feed - inst[2]), k.relu(inst[3] - self.t_feed)), [batch, 1])   # 0 if below t_min, 0 if above t_max, parabolic downwards inbetween
                t_mask = t_mask / (0.25 * (inst[3]-inst[2])**3)                                                                     # This is squared to normalize the range, then divided once more to make the average of the residuals more like 1, even when the constraint is levied only over a narrow range of normalized t 0 to 1.
                rate_mask = k.relu(inst[4] - self.functions['dT_dt_equ']) + k.relu(self.functions['dT_dt_equ'] - inst[5])           # 0 if inside range, linear outside of range, but can be very small since dT/dt can be small
                weight_mask = one_feed * inst[6]
                loss_term_inst = tf.multiply(tf.multiply(tf.multiply(x_mask, t_mask), rate_mask), weight_mask)
                loss_term_inst = k.mean(k.square(k.relu(loss_term_inst+self.bias))) - self.bias**2 #VANISH
                # not debugged
                self.loss_lists['constraint_dT_dt'] = self.loss_lists['constraint_dT_dt'] + loss_term_inst
                self.hard_constraints['constraint_dT_dt'] = self.hard_constraints['constraint_dT_dt'] + loss_term_inst
            self.hard_constraints['constraint_dT_dt'] = k.mean(self.hard_constraints['constraint_dT_dt']) + self.bias
        if 'constraint_A_dT_dt' in self.equation_list:
            self.loss_lists['constraint_A_dT_dt'] = tf.reshape(tf.convert_to_tensor([0]*batch, dtype=np.float32), [batch,1])
            self.hard_constraints['constraint_A_dT_dt'] = k.mean(k.square(tf.reshape(tf.convert_to_tensor([0] * batch, dtype=np.float32), [batch, 1])))
            for inst in constraint_A_dT_dt:
                x_mask = tf.reshape(tf.multiply(k.relu(self.x_feed - inst[0]), k.relu(inst[1] - self.x_feed)), [batch, 1])                                  # 0 if below x_min, 0 if above x_max, parabolic downwards inbetween
                x_mask = x_mask / (0.25 * (inst[1]-inst[0])**2)
                range_mask = tf.multiply(k.relu(self.functions['T_equ'] - inst[2]), k.relu(inst[3] - self.functions['T_equ']))      # 0 if below T_min, 0 if above T_max, parabolic downwards inbetween
                range_mask = range_mask / (0.25 * (inst[3]-inst[2])**2)
                rate_mask = k.relu(inst[4] - self.functions['dT_dt_equ']) + k.relu(self.functions['dT_dt_equ'] - inst[5])           # 0 if inside range, linear outside of range, but can be very small since dT/dt can be small
                weight_mask = one_feed * inst[6]
                loss_term_inst = tf.multiply(tf.multiply(tf.multiply(x_mask, range_mask), rate_mask), weight_mask) 
                loss_term_inst = k.square(k.relu(loss_term_inst) + self.bias) #VANISH
                #print(self.material_name, "A_dT_dt loss term inst:", loss_term_inst)
                self.loss_lists['constraint_A_dT_dt'] = self.loss_lists['constraint_A_dT_dt'] + loss_term_inst
                self.hard_constraints['constraint_A_dT_dt'] = self.hard_constraints['constraint_A_dT_dt'] + loss_term_inst
            self.hard_constraints['constraint_A_dT_dt'] = k.mean(self.hard_constraints['constraint_A_dT_dt'])
            #print(self.hard_constraints['constraint_A_dT_dt'])
        if 'constraint_doc_soft' in self.equation_list: #$
            self.loss_lists['constraint_doc_soft'] = tf.reshape(tf.convert_to_tensor([0]*batch, dtype=np.float32), [batch,1])
            self.hard_constraints['constraint_doc_soft'] = k.mean(k.square(tf.reshape(tf.convert_to_tensor([0] * batch, dtype=np.float32), [batch, 1])))
            #print("btw this value is ", self.hard_constraints['constraint_doc_soft'])
            if 'doc' in self.net:
                for inst in constraint_doc_soft:
                    x_mask = tf.reshape(tf.multiply(k.relu(self.x_feed - inst[0]), k.relu(inst[1] - self.x_feed)), [batch, 1])                                  # 0 if below x_min, 0 if above x_max, parabolic downwards inbetween
                    x_mask = x_mask / (0.25 * (inst[1]-inst[0])**3)  #This is squared to normalize the range, then divided once more to make the average of the residuals more like 1, even when the constraint is levied only over a narrow range of normalized t 0 to 1.
                    t_mask = tf.reshape(tf.multiply(k.relu(self.t_feed - inst[2]), k.relu(inst[3] - self.t_feed)), [batch, 1])                                  # 0 if below t_min, 0 if above t_max, parabolic downwards inbetween
                    t_mask = t_mask / (0.25 * (inst[3]-inst[2])**3)  #This is squared to normalize the range, then divided once more to make the average of the residuals more like 1, even when the constraint is levied only over a narrow range of normalized t 0 to 1.
                    range_mask = k.relu(inst[4] - self.functions['doc_equ']) + k.relu(self.functions['doc_equ'] - inst[5])                   # 0 if inside range, linear outside of range
                    weight_mask = one_feed * inst[6]
                    loss_term_inst = tf.multiply(tf.multiply(tf.multiply(x_mask, t_mask), range_mask), weight_mask)
                    loss_term_inst = k.square(k.relu(loss_term_inst) + self.bias) #VANISH
                    self.loss_lists['constraint_doc_soft'] = self.loss_lists['constraint_doc_soft'] + loss_term_inst
                    self.hard_constraints['constraint_doc_soft'] = self.hard_constraints['constraint_doc_soft'] + loss_term_inst
            self.hard_constraints['constraint_doc_soft'] = k.mean(self.hard_constraints['constraint_doc_soft'])
            #print(self.hard_constraints['constraint_doc_soft'])
        if 'constant_T' in self.equation_list:
            self.loss_lists['constant_T'] = (k.square(self.functions['dT_dx']))
            self.hard_constraints['constant_T'] = k.mean(k.square(self.functions['dT_dx']))
        if 'ddoc_dt' in self.equation_list:
            #self.loss_lists['ddoc_dt'] = k.square(self.functions['ddoc_dt'] - self.functions['ddoc_dt_equ'])      # Old method
            self.loss_lists['ddoc_dt'] = (k.square(zero_feed))  # MTU
        if 'doc_monotonic' in self.equation_list:
            self.loss_lists['doc_monotonic'] = (k.square(k.relu(-1*self.functions['ddoc_dt'])))  # MTU
            self.hard_constraints['doc_monotonic'] = k.mean(k.square(k.relu(-1 * self.functions['ddoc_dt'])))
        if 'cure kinetics non-touching' in self.equation_list:
            # self.loss_lists['doc_PDE'] = k.square((tf.math.log(self.material.properties['ck']['AA']*(t_max-t_min))+(-self.material.properties['ck']['Ea']/(self.material.properties['ck']['R']*(273.15+T_min+self.Tck_inf_feed*(T_max-T_min))))+self.material.properties['ck']['mm']*tf.math.log(self.functions['doc_equ'])+self.material.properties['ck']['nn']*tf.math.log(1-self.functions['doc_equ']) - tf.math.log(1+tf.math.exp(self.material.properties['ck']['C'] * (self.functions['doc_equ'] - self.material.properties['ck']['xC0'] - (self.functions['T_equ']*(T_max-T_min))* self.material.properties['ck']['xCT'])))) - tf.math.log(self.functions['ddoc_dt_equ']) + k.relu(tf.math.log(tf.math.scalar_mul(doc_0, zero_feed)) - tf.math.log(self.functions['doc_equ'])))
            self.loss_lists['doc_PDE'] = (k.square((tf.math.log(self.material.properties['ck']['AA'])+(-self.material.properties['ck']['Ea']/(self.material.properties['ck']['R']*(273.15+T_min+self.Tck_inf_feed*(T_max-T_min))))+self.material.properties['ck']['mm']*tf.math.log(self.functions['doc_equ'])+self.material.properties['ck']['nn']*tf.math.log(1-self.functions['doc_equ']) - tf.math.log(1+tf.math.exp(self.material.properties['ck']['C'] * (self.functions['doc_equ'] - self.material.properties['ck']['xC0'] - (273.5+self.functions['T_equ'])* self.material.properties['ck']['xCT'])))) - tf.math.log(self.functions['ddoc_dt_equ']) + k.relu(tf.math.log(tf.math.scalar_mul(doc_0, zero_feed)) - tf.math.log(self.functions['doc_equ'])))) #MTU
            self.governing_eqs['doc_PDE'] = k.mean(k.square(k.abs((tf.math.log(self.material.properties['ck']['AA']) + (-self.material.properties['ck']['Ea'] / (self.material.properties['ck']['R'] * (273.15 + T_min + self.Tck_inf_feed * (T_max - T_min)))) + self.material.properties['ck']['mm'] * tf.math.log(
                self.functions['doc_equ']) + self.material.properties['ck']['nn'] * tf.math.log(1 - self.functions['doc_equ']) - tf.math.log(
                1 + tf.math.exp(self.material.properties['ck']['C'] * (self.functions['doc_equ'] - self.material.properties['ck']['xC0'] - (273.5 + self.functions['T_equ']) * self.material.properties['ck']['xCT'])))) - tf.math.log(self.functions['ddoc_dt_equ']) + k.relu(
                tf.math.log(tf.math.scalar_mul(doc_0, zero_feed)) - tf.math.log(self.functions['doc_equ']))) + self.bias))
        # if 'cure kinetics' in self.equation_list:
        #     # self.loss_lists['doc_PDE'] = k.square((tf.math.log(self.material.properties['ck']['AA']*(t_max-t_min))+(-self.material.properties['ck']['Ea']/(self.material.properties['ck']['R']*(273.15+T_min+self.functions['T_equ']*(T_max-T_min))))+self.material.properties['ck']['mm']*tf.math.log(self.functions['doc_equ'])+self.material.properties['ck']['nn']*tf.math.log(1-self.functions['doc_equ'])  - tf.math.log(1+tf.math.exp(self.material.properties['ck']['C'] * (self.functions['doc_equ'] - self.material.properties['ck']['xC0'] - (T_min+self.functions['T_equ']*(T_max-T_min)) * self.material.properties['ck']['xCT'])))) - tf.math.log(self.functions['ddoc_dt_equ']) + k.relu(tf.math.log(tf.math.scalar_mul(doc_0, zero_feed)) - tf.math.log(self.functions['doc_equ'])))
        #     # self.loss_lists['doc_PDE'] = k.square((tf.math.log(self.material.properties['ck']['AA'])+(-self.material.properties['ck']['Ea']/(self.material.properties['ck']['R']*(273.5+self.functions['T_equ'])))+self.material.properties['ck']['mm']*tf.math.log(self.functions['doc_equ'])+self.material.properties['ck']['nn']*tf.math.log(1-self.functions['doc_equ'])  - tf.math.log(1+tf.math.exp(self.material.properties['ck']['C'] * (self.functions['doc_equ'] - self.material.properties['ck']['xC0'] - (T_min+self.functions['T_equ']+273.5) * self.material.properties['ck']['xCT'])))) - tf.math.log(self.functions['ddoc_dt_equ']) + k.relu(tf.math.log(tf.math.scalar_mul(doc_0, zero_feed)) - tf.math.log(self.functions['doc_equ']))) #MTU
        #     K = self.material.properties['ck']['AA']*tf.math.exp(-self.material.properties['ck']['Ea']/(self.material.properties['ck']['R']*(273.15+self.functions['T_equ'])))
        #     numerator = K * tf.pow(self.functions['doc_equ'],self.material.properties['ck']['mm']) * tf.pow(1-self.functions['doc_equ'],self.material.properties['ck']['nn'])
        #     denominator = 1 + tf.math.exp(self.material.properties['ck']['C']*(self.functions['doc_equ']-(self.material.properties['ck']['xC0']+self.material.properties['ck']['xCT']*(273.15+self.functions['T_equ']))))
        #     self.loss_lists['doc_PDE'] = k.square(tf.divide(numerator,denominator) - self.functions['ddoc_dt_equ'])
        #     # self.loss_lists['doc_PDE'] = k.square((tf.math.log(self.material.properties['ck']['AA'])+(-self.material.properties['ck']['Ea']/(self.material.properties['ck']['R']*(273.5+self.functions['T_equ'])))+self.material.properties['ck']['mm']*tf.math.log(self.functions['doc_equ'])+self.material.properties['ck']['nn']*tf.math.log(1-self.functions['doc_equ'])  - tf.math.log(1+tf.math.exp(self.material.properties['ck']['C'] * (self.functions['doc_equ'] - self.material.properties['ck']['xC0'] - (T_min+self.functions['T_equ']+273.5) * self.material.properties['ck']['xCT'])))) - tf.math.log(self.functions['ddoc_dt_equ'])) #MTU

        if 'cure kinetics' in self.equation_list:
            current_temp = (273.15+self.functions['T_equ'])
            alpha = self.functions['doc_equ']
            K1 = self.material.properties['ck']['AA_1'] * np.exp(-self.material.properties['ck']['Ea_1'] / (self.material.properties['ck']['R'] * current_temp))
            K2 = self.material.properties['ck']['AA_2'] * np.exp(-self.material.properties['ck']['Ea_2'] / (self.material.properties['ck']['R'] * current_temp))
            Tg = self.material.properties['ck']['Tg0'] + self.material.properties['ck']['lam'] * alpha * (self.material.properties['ck']['Tginf'] - self.material.properties['ck']['Tg0']) / (1 - 0.2 * alpha)
            kd = self.material.properties['ck']['Ad'] * np.exp(-self.material.properties['ck']['Ed'] / (self.material.properties['ck']['R'] * current_temp) - self.material.properties['ck']['b'] / (self.material.properties['ck']['w'] * (current_temp - Tg) + self.material.properties['ck']['g']))
            K1_eff = K1 * kd / (K1 + kd)
            K2_eff = K2 * kd / (K2 + kd)
            rate = K1_eff * (1 - alpha) ** (self.material.properties['ck']['nn_1']) + K2_eff * (alpha ** (self.material.properties['ck']['mm_2'])) * ((1 - alpha) ** (self.material.properties['ck']['nn_2']))
            # rate = tf.convert_to_tensor(rate)
            # self.loss_lists['doc_pde'] = k.square(rate-self.functions['ddoc_dt_equ'])
            self.loss_lists['doc_pde_log'] = (k.square(tf.math.log(k.relu(rate)+1e-5) - tf.math.log(k.relu(self.functions['ddoc_dt_equ'])+1e-5)))
            self.loss_lists['doc_pde_lin'] = (k.square(rate - self.functions['ddoc_dt_equ']))
            self.governing_eqs['doc_pde_log'] = k.mean(k.square(k.abs(tf.math.log(k.relu(rate) + 1e-5) - tf.math.log(k.relu(self.functions['ddoc_dt_equ']) + 1e-5))+ self.bias))
            self.governing_eqs['doc_pde_lin'] = k.mean(k.square(k.abs(rate - self.functions['ddoc_dt_equ'])+self.bias))

        if 'cure kinetics autocatalytic' in self.equation_list: #MTU
            self.loss_lists['doc_PDE'] = (k.square(tf.math.log(self.material.properties['ck']['AA'])+(-self.material.properties['ck']['Ea']/(self.material.properties['ck']['R']*(273.15+self.functions['T_equ'])))+self.material.properties['ck']['mm']*tf.math.log(self.functions['doc_equ'])+self.material.properties['ck']['nn']*tf.math.log(1-self.functions['doc_equ']) \
                - tf.math.log(self.functions['ddoc_dt_equ']))) #######################PROBLEM??????????#$
            self.governing_eqs['doc_PDE'] = k.mean(k.square(
                tf.math.log(self.material.properties['ck']['AA']) + (-self.material.properties['ck']['Ea'] / (self.material.properties['ck']['R'] * (273.15 + self.functions['T_equ']))) + self.material.properties['ck']['mm'] * tf.math.log(self.functions['doc_equ']) + self.material.properties['ck'][
                    'nn'] * tf.math.log(1 - self.functions['doc_equ']) \
                - tf.math.log(self.functions['ddoc_dt_equ'])))
        if 'cure kinetics linear' in self.equation_list:
            # self.loss_lists['doc_PDE_lin'] = k.square((self.material.properties['ck']['AA']*(t_max-t_min))*tf.math.exp(-self.material.properties['ck']['Ea']/(self.material.properties['ck']['R']*(273.15+T_min+self.functions['T_equ']*(T_max-T_min))))*(self.functions['doc_equ'])**(self.material.properties['ck']['mm'])*(1-self.functions['doc_equ'])**(self.material.properties['ck']['nn']) / (1+tf.math.exp(self.material.properties['ck']['C'] * (self.functions['doc_equ'] - self.material.properties['ck']['xC0'] - (T_min+self.functions['T_equ']*(T_max-T_min)) * self.material.properties['ck']['xCT'])))) - tf.math.log(self.functions['ddoc_dt_equ'])
            self.loss_lists['doc_PDE_lin'] = (k.square((self.material.properties['ck']['AA'])*tf.math.exp(-self.material.properties['ck']['Ea']/(self.material.properties['ck']['R']*(273.15+self.functions['T_equ'])))*(self.functions['doc_equ'])**(self.material.properties['ck']['mm'])*(1-self.functions['doc_equ'])**(self.material.properties['ck']['nn']) / (1+tf.math.exp(self.material.properties['ck']['C'] * (self.functions['doc_equ'] - self.material.properties['ck']['xC0'] - (273.5+T_min+self.functions['T_equ']) * self.material.properties['ck']['xCT'])))) - tf.math.log(self.functions['ddoc_dt_equ'])) #MTU
            self.governing_eqs['doc_PDE_lin'] = k.mean(k.square(k.abs(
                (self.material.properties['ck']['AA']) * tf.math.exp(-self.material.properties['ck']['Ea'] / (self.material.properties['ck']['R'] * (273.15 + self.functions['T_equ']))) * (self.functions['doc_equ']) ** (self.material.properties['ck']['mm']) * (1 - self.functions['doc_equ']) ** (
                self.material.properties['ck']['nn']) / (1 + tf.math.exp(self.material.properties['ck']['C'] * (self.functions['doc_equ'] - self.material.properties['ck']['xC0'] - (273.5 + T_min + self.functions['T_equ']) * self.material.properties['ck']['xCT'])))) - tf.math.log(
                self.functions['ddoc_dt_equ']) + self.bias))

        if ('target_cure_duration' in self.equation_list) or ('minimize_cure_duration' in self.equation_list) or ('remain optimized' in self.equation_list):
            DOC_end_min = 0.8
            num_temp = 400
            test_z = 0
            if (self.component_number in [1, 2]): # Air, Tool
                test_z = 0
            if (self.component_number in [3]): # Part
                test_z = self.thickness
            if (False): #3D
                zero_test_xyz = np.column_stack([np.zeros(num_temp), np.zeros(num_temp), test_z * np.ones(num_temp)])
            elif (True): #1D
                zero_test_xyz = np.column_stack([test_z * np.ones(num_temp)])
            else:
                zero_test_xyz = np.column_stack([test_z * np.ones(num_temp)])  # This is for fluids, which are _geometry.dimensions = 0; the Network.values() function will igmore the zero_test_xyz but it must be defined

            self.t_test = np.linspace(t_min, t_max, num_temp, dtype=np.float32)
            doc_index = next((index for index, value in enumerate(self.net['doc'].values([zero_test_xyz, self.t_test])) if value >= DOC_end_min * 0.98), -1)
            #print(f"The Index of the end of cure (DOC) is {doc_index}")  ###
            self.t_test = np.linspace(self.t_test[doc_index], t_max, num_temp, dtype=np.float32)  # VTN
            self.functions['T_test_equ'] = tf.multiply(self.net['T'].values([zero_test_xyz, self.t_test]), np.tanh(30.0 * self.t_test / (t_max - t_min)).reshape(len(self.t_test), 1)) + (((T0)) * (1.0 - self.t_test / (t_max - t_min))).reshape(len(self.t_test), 1)  # MTU
            first_index = next((index for index, value in enumerate(self.functions['T_test_equ']) if value <= T_end), -1)
            # first_index = next((len(self.functions['T_test_equ']) - 1 - index for index, value in enumerate(reversed(self.functions['T_test_equ'])) if value < T_end), -1)
            # self.end_of_cure = self.t_test[first_index] + (self.t_test[first_index] - self.t_test[first_index - 1]) * (T_end - self.functions['T_test_equ'][first_index]) / (self.functions['T_test_equ'][first_index] - self.functions['T_test_equ'][first_index - 1])
            if first_index == -1:
                self.end_of_cure = t_max + (self.functions['T_test_equ'][first_index] / (10 * T_end))  # may want to change self_target_duration to something else so that you can plug in the t_max for terget duration to start
                #print('if')
            else:
                self.end_of_cure = self.t_test[first_index] + (self.t_test[first_index] - self.t_test[first_index - 1]) * (T_end - self.functions['T_test_equ'][first_index]) / (self.functions['T_test_equ'][first_index] - self.functions['T_test_equ'][first_index - 1])
                # self.end_of_cure = self._target_duration * self.functions['T_test_equ'][first_index]/T_end ## Considering some way to create a multiple of of T_test_equ rather than a tiny slope aspect.
                #print('else')
            self.loss_lists['cure_duration'] = k.square(k.relu(self.end_of_cure - self._target_duration))  # If end of cure < target duration, 0 loss, otherwise loss.
            self.soft_constraints['cure_duration'] = k.mean(k.square(self.end_of_cure))
            #print(f"The Index of the end of cure (T_end) is {first_index}")  ###
            '''old?
            num_temp = 400
            #component numnbers: 1 -> air    2 -> tool   3 -> part
            if ('1D' == '3D'):
                if (self.component_number in [1, 2]):
                    test_z = [0]
                elif (self.component_number == 3):
                    test_z = self.thickness
                else:
                    print('SUPER DUPER FAIL')
                zero_test_xyz = np.column_stack([np.zeros(num_temp), np.zeros(num_temp), test_z * np.ones(num_temp)])

            if ('1D' == '1D'):
                if (self.component_number in [1, 2]): #lowkey might want to change this to screen based on thickness regardless of which part its attached to.
                    test_z = [0]
                elif (self.component_number == 3):
                    if (self.thickness == 0):
                        test_z = [0]
                    else:
                        test_z = []
                        num_test_points = 3 #bottom, middle, top
                        for num in range(num_test_points):
                            test_z.append(self.thickness * (num / (num_test_points - 1)))
                else:
                    print('SUPER DUPER FAIL 2')
                #zero_test_xyz = test_z * np.ones(num_temp)
            #zero_test_xyz = np.column_stack(np.zeros((num_temp, 3)))
            self.t_test = np.linspace(3.25, 3.75, num_temp, dtype=np.float32)       # VTN # CHANGE FOR LOCATA
            #print("Test Z: ", test_z)
            #print("t_test: ", self.t_test)
            #print("Component name: ", self.material_name)
            if (len(test_z) != 1):
                spots = []
                T_subtest = []
                final_list = []
                for spot in range(len(test_z)):
                    #x_test[spot] = test_z[spot] * np.ones(len(self.t_test))
                    T_subtest.append(tf.multiply(self.net['T'].values([test_z[spot] * np.ones(len(self.t_test)), self.t_test]), np.tanh(30.0 * self.t_test / (t_max - t_min)).reshape(len(self.t_test), 1)) + (((T0)) * (1.0 - self.t_test / (t_max - t_min))).reshape(len(self.t_test), 1))
                    spots.append(spot)
                for index in range(len(T_subtest[0])):
                    column_values = [row[index] for row in T_subtest]  # get the column
                    final_list.append(max(column_values))

            self.functions['T_test_equ'] = final_list #CJE
            first_index = next((index for index, value in enumerate(self.functions['T_test_equ']) if value < T_end), -1)
            self.end_of_cure = self.t_test[first_index] + (self.t_test[first_index] - self.t_test[first_index - 1]) * (T_end - self.functions['T_test_equ'][first_index]) / (self.functions['T_test_equ'][first_index] - self.functions['T_test_equ'][first_index - 1])
            # self.loss_lists['cure_duration'] = k.square(k.relu(self._target_duration - self.end_of_cure))
            self.loss_lists['cure_duration'] = (k.square((self._target_duration - self.end_of_cure)))
            self.soft_constraints['cure_duration'] = k.mean(k.square((self.end_of_cure))) #&&
            #print("End of cure [hrs]: ", self.end_of_cure)
            #print("T_test_equ: ", np.array(self.functions['T_test_equ']))
            #print("First Index", first_index)

        #if 'minimize cure duration' in self.equation_list:
        #    self.t_test = np.linspace(self._target_duration*duration_increment_factor**2, self._target_duration/duration_increment_factor**2, 400, dtype=np.float32)       # VTN
        #    self.functions['T_test_equ'] = tf.multiply(self.net['T'].values([self.t_test, self.t_test]), np.tanh(30.0 * self.t_test/(t_max-t_min)).reshape(len(self.t_test),1)) + (((T0)) * (1.0 - self.t_test/(t_max-t_min))).reshape(len(self.t_test),1) #MTU
        #    first_index = next((index for index, value in enumerate(self.functions['T_test_equ']) if value < T_end), -1)
        #    end_of_cure = self.t_test[first_index] + (self.t_test[first_index] - self.t_test[first_index-1])*(T_end - self.functions['T_test_equ'][first_index])/(self.functions['T_test_equ'][first_index]-self.functions['T_test_equ'][first_index-1])
            # if (self.functions['T_test_equ'][first_index]<T_end):
            # print(first_index)
            # print(np.array(end_of_cure))
        #    self.loss_lists['cure_duration'] = k.square(self._target_duration - end_of_cure)
            # else:
            #     self.loss_lists['cure_duration'] = [0]
            #self._target_duration = max(0.1*t_max, min(t_max, duration_increment_factor * k.get_value(end_of_cure)))
            '''
        
        if 'FEM part T' in self.equation_list:
            T_prediction_list = tf.multiply(self.net['T'].values([data_part_T_x, data_part_T_t]), tf.tanh(30.0 * data_part_T_t/(t_max-t_min))) + ((T0)) * (1.0 - data_part_T_t/(t_max-t_min)) #MTU
            # T_prediction_list = self.net['T'].values([data_part_T_x, data_part_T_t])
            self.loss_lists['FEM part T'] = (k.square(T_prediction_list - data_part_T_T))
            self.governing_eqs['FEM part T'] = k.mean(k.square(k.abs(T_prediction_list - data_part_T_T) + self.bias))
        if 'FEM tool T' in self.equation_list:
            T_prediction_list = tf.multiply(self.net['T'].values([data_tool_T_x, data_tool_T_t]), tf.tanh(30.0 * data_tool_T_t/(t_max-t_min))) + ((T0)) * (1.0 - data_tool_T_t/(t_max-t_min)) #MTU
            # T_prediction_list = self.net['T'].values([data_tool_T_x, data_tool_T_t])
            self.loss_lists['FEM tool T'] = (k.square(T_prediction_list - data_tool_T_T))
            self.governing_eqs['FEM tool T'] = k.mean(k.square(k.abs(T_prediction_list - data_tool_T_T) + self.bias))
        if 'FEM air T' in self.equation_list:
            T_prediction_list = tf.multiply(self.net['T'].values([data_air_T_x, data_air_T_t]), tf.tanh(30.0 * data_air_T_t/(t_max-t_min))) + ((T0)) * (1.0 - data_air_T_t/(t_max-t_min)) #MTU
            # T_prediction_list = self.net['T'].values([data_air_T_x, data_air_T_t])
            self.loss_lists['FEM air T'] = (k.square(T_prediction_list - data_air_T_T))
            self.governing_eqs['FEM air T'] = k.mean(k.square(k.abs(T_prediction_list - data_air_T_T) + self.bias))
            #print("The FEM air T is ", self.governing_eqs['FEM air T'])
        if 'FEM blanket T' in self.equation_list:
            T_prediction_list = tf.multiply(self.net['T'].values([data_blanket_T_x, data_blanket_T_t]), tf.tanh(30.0 * data_blanket_T_t / (t_max - t_min))) + ((T0)) * (1.0 - data_blanket_T_t / (t_max - t_min))  # MTU
            # T_prediction_list = self.net['T'].values([data_air_T_x, data_air_T_t])
            self.loss_lists['FEM blanket T'] = (k.square(T_prediction_list - data_blanket_T_T))
            self.governing_eqs['FEM blanket T'] = k.mean(k.square(k.abs(T_prediction_list - data_blanket_T_T) + self.bias))
        if 'FEM part DOC' in self.equation_list:
            DOC_prediction_list = tf.multiply(self.net['doc'].values([data_part_DOC_x, data_part_DOC_t]), tf.tanh(30.0 * data_part_DOC_t/(t_max-t_min))) + doc_0 * k.relu(1.0 - 4.0 * data_part_DOC_t/(t_max-t_min)) #MTU
            # DOC_prediction_list = self.net['doc'].values([data_part_DOC_x, data_part_DOC_t])
            self.loss_lists['FEM part DOC'] = (k.square(DOC_prediction_list - data_part_DOC_DOC))
            self.governing_eqs['FEM part DOC'] = k.mean(k.square(k.abs(DOC_prediction_list - data_part_DOC_DOC) + self.bias))
        #for name in self.loss_lists:
        #    print(self.material_name, name, self.loss_lists[name][0])

    def reset_loss_weightings(self):           
        self.losses = {**self.loss_lists}
        if not LoCaTA_active:
            for name in self.loss_lists:
                #print(name)
                if (len(self.loss_lists[name]) == 1):
                    self.losses[name] = self.loss_lists[name][0]
                else:
                    self.losses[name] = k.mean(self.loss_lists[name])
        self.loss_weightings = {}
        for name in self.losses:
            self.loss_weightings[name] = 1

    def calculate_weighted_losses(self):
        self.losses_weighted = {}
        for name in self.losses:
            self.losses_weighted[name] = self.losses[name] * self.loss_weightings[name]
        self.loss_total = sum([self.losses_weighted[name] for name in self.losses_weighted])
        #print(self.loss_total)

        if math.isnan(self.loss_total):
            print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    ''' this might still work - Doesn't need to because of the new training methodology
    def calculate_weighted_losses(self, stage):
        self.losses_weighted = {}
        for name in self.losses:
            self.losses_weighted[name] = self.losses[name] * self.loss_weightings[name]
            if name in self.hard_constraints:
                self.hard_constraints[name] = self.losses_weighted[name]
            elif name in self.governing_eqs:
                self.governing_eqs[name] = self.losses_weighted[name]
            elif name in self.soft_constraints:
                self.soft_constraints[name] = self.losses_weighted[name]
    '''
    def trench_losses(self, loss_limit):
        for name in self.hard_constraints:
            if self.hard_constraints[name] <= (loss_limit[name]) + self.bias**2: #self.loss_limits[name]:
                self.hard_constraints[name] = self.hard_constraints[name] - k.get_value(self.hard_constraints[name])    
        for name in self.governing_eqs:
            if self.governing_eqs[name] <= (loss_limit[name]) + self.bias**2: #self.loss_limits[name]:
                self.governing_eqs[name] = self.governing_eqs[name] - k.get_value(self.governing_eqs[name])
            #if name == 'FEM air T':
                #print('FEM air T post-trenching is ', self.governing_eqs[name])
            #else:
                #print(name, " ", self.governing_eqs[name])

    def optimize_targets(self):
        global optimized
        global loss_threshold_for_duration_minimization
        global total_loss_threshold_for_duration_minimization
        if ('minimize_cure_duration' in self.equation_list) and (self.loss_total <= total_loss_threshold_for_duration_minimization) and (self.losses['cure_duration'] < loss_threshold_for_duration_minimization):
            self._target_duration = self._target_duration * duration_increment_factor
            optimized = True
            #if loss_threshold_for_duration_minimization > cure_loss_min:
            #    loss_threshold_for_duration_minimization = loss_threshold_for_duration_minimization * threshold_increment_factor
            #if total_loss_threshold_for_duration_minimization > total_loss_min:
            #    total_loss_threshold_for_duration_minimization = total_loss_threshold_for_duration_minimization * threshold_increment_factor
            print(f"\n\nNew Target Duration: {self._target_duration}\nNew cure_duration threshold: {loss_threshold_for_duration_minimization}\nNew total_loss threshold: {total_loss_threshold_for_duration_minimization}")
            print(f"The total loss is {self.loss_total} and the cure duration loss is {self.losses['cure_duration']}")
            
    def train_models(self, tape_3):
        self.gradients = {}
        for name in self.net:
            #local_copy_of_weights = [self.net[name].model.trainable_variables[i].numpy() for i in range(len(self.net[name].model.trainable_variables))]
            if name not in self.non_trainable_list:
                try:
                    self.gradients[name] = tape_3.gradient(self.loss_total, self.net[name].model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                except:
                    print("Failed on ", self.material_name,"!!! NOT ACTUALLY TRAINING!!!")
                ##print("stage 2: ", self.gradients[name])
                self.net[name].model_optimizer.apply_gradients((grad, var) for (grad, var) in zip(self.gradients[name], self.net[name].model.trainable_variables) if grad is not None)

    def save_models(self):
        for name in self.net:
            self.net[name].save_model()

class Boundary:
    '''
    Defines a boundary between two Components 
    While in 1+1D, the following are implicit: component1(internal_x=1) is co-located with component2(internal_x=0)
    '''
    boundary_number = 0

    def __init__(self, component1, component2, time_min, time_max, equation_list, htc=0):
        self.component1 = component1
        self.component2 = component2
        self.time_min = time_min
        self.time_max = time_max
        self.equation_list = equation_list
        self.h = htc
        Boundary.boundary_number += 1
        self.boundary_number = Boundary.boundary_number
    
    def define_residual_set(self, batch):
        self.t_arr = np.random.uniform(t_min, t_max, batch) #MTU
        self.t_feed = np.column_stack((self.t_arr))
        self.t_feed = tf.Variable(self.t_feed.reshape(len(self.t_feed[0]),1), trainable=True, dtype=tf.float32)
        self.c1_x_feed = tf.scalar_mul(self.component1.thickness, one_feed)
        self.c2_x_feed = tf.scalar_mul(self.component2.thickness, one_feed)

    def define_functions(self):
        self.functions = {}
        self.functions['T_equ_comp_1'] = tf.multiply(self.component1.net['T'].values([self.c1_x_feed, self.t_feed]), tf.tanh(30.0 * (self.t_feed-t_min)/(t_max-t_min))) + T0 * (1.0 - (self.t_feed-t_min)/(t_max-t_min)) #MTU
        self.functions['T_equ_comp_2'] = tf.multiply(self.component2.net['T'].values([zero_feed, self.t_feed]), tf.tanh(30.0 * (self.t_feed-t_min)/(t_max-t_min))) + T0 * (1.0 - (self.t_feed-t_min)/(t_max-t_min)) #MTU

    def define_derivative_functions(self, tape_1):
        if self.component1.material.properties['phase'] == 'solid':
            self.functions['dT_dx_equ_comp_1'] = tape_1.gradient(self.functions['T_equ_comp_1'], [self.c1_x_feed, self.t_feed])[0]
        if self.component2.material.properties['phase'] == 'solid':
            self.functions['dT_dx_equ_comp_2'] = tape_1.gradient(self.functions['T_equ_comp_2'], [zero_feed, self.t_feed])[0]

            # question: why zero feed?

    def define_boundary_losses(self):
        self.loss_lists = {}
        if 'thermal contact' in self.equation_list:
            self.loss_lists['T'] = k.mean(k.square(self.functions['T_equ_comp_1'] - self.functions['T_equ_comp_2']))
        if 'conductive heat transfer' in self.equation_list:
            self.loss_lists['T'] = k.mean(k.square(self.functions['T_equ_comp_1'] - self.functions['T_equ_comp_2']))
            # self.loss_lists['conduction'] = k.square((self.component1.material.properties['k']*self.functions['dT_dx_equ_comp_1'] - self.component2.material.properties['k']*self.functions['dT_dx_equ_comp_2']))
            self.loss_lists['conduction'] = k.mean(k.square((self.component1.material.properties['k']/self.component2.material.properties['k']*self.functions['dT_dx_equ_comp_1'] - self.functions['dT_dx_equ_comp_2'])))
        if 'convective heat transfer' in self.equation_list:
            # Assume that convective heat transfer is only defined where one component is fluid and the other is solid
            if self.component1.material.properties['phase'] == 'fluid':
                self.fluid_T = self.functions['T_equ_comp_1']
                self.solid_T = self.functions['T_equ_comp_2']
                #self.solid_dT_dx = -self.functions['dT_dx_equ_comp_2']
                self.solid_dT_dx = self.functions['dT_dx_equ_comp_2']               # Seems correct
                self.solid_k = self.component2.material.properties['k']
                self.solid_L = self.component2.thickness
                self.loss_lists['convection'] = k.mean(k.square(-1*(self.fluid_T - self.solid_T) - self.solid_k*self.solid_dT_dx/self.h)) #MTU NEEDS CHECK #CHANGE BACK TO K.MEAN FOR LOCATA!!!!!!
            else:
                self.fluid_T = self.functions['T_equ_comp_2']
                self.solid_T = self.functions['T_equ_comp_1']
                self.solid_dT_dx = self.functions['dT_dx_equ_comp_1']              # Previously used version, thought it was correct. it's not. Why not?
                #self.solid_dT_dx = -self.functions['dT_dx_equ_comp_1']
                self.solid_k = self.component1.material.properties['k']
                self.solid_L = self.component1.thickness
                self.loss_lists['convection'] = k.mean(k.square(+1*(self.fluid_T - self.solid_T) - self.solid_k*self.solid_dT_dx/self.h)) #MTU NEEDS CHECK #CHANGE BACK TO K.MEAN FOR LOCATA!!!!!!
                # question: what is air equality
        #if 'air equality' in self.equation_list:
        #    self.loss_lists['air equality'] = (k.square(self.functions['T_equ_comp_1'] - self.functions['T_equ_comp_2']))

    def distribute_boundary_losses(self):
        for loss_list in self.loss_lists:
            if 'conduction' in self.loss_lists: #send conduction to just component 1
                self.component1.loss_lists[loss_list + "_" + str(self.boundary_number)] = self.loss_lists[loss_list]
                self.component2.loss_lists[loss_list + "_" + str(self.boundary_number)] = self.loss_lists[loss_list]

            if 'convection' in self.loss_lists: #split convection to just the solids
                if self.component1.material.properties['phase'] == 'solid':
                    self.component1.loss_lists[loss_list + "_" + str(self.boundary_number)] = self.loss_lists[loss_list]
                    self.component1.governing_eqs[loss_list + "_" + str(self.boundary_number)] = self.loss_lists[loss_list]
                else:
                    self.component2.loss_lists[loss_list + "_" + str(self.boundary_number)] = self.loss_lists[loss_list]
                    self.component2.governing_eqs[loss_list + "_" + str(self.boundary_number)] = self.loss_lists[loss_list]
            if 'T' in self.loss_lists: #send conduction to dependent component, hardcoded rn
                self.component2.loss_lists[loss_list + "_" + str(self.boundary_number)] = self.loss_lists[loss_list]

#################################################################
# Model names and definitions
#################################################################
prediction_names = ['T', 'doc']#, 'dT_dt', 'dT_dx', 'ddoc_dt']
network_definition = {
    'T':        {'dimensions': dimensions, 'layers': model_layers, 'nodes': nodes_per_layer, 'activation_function': 'swish', 'activation_final': 'linear'}, #MTU, change activation_final
    'dT_dt':    {'dimensions': dimensions, 'layers': model_layers, 'nodes': nodes_per_layer, 'activation_function': 'swish', 'activation_final': 'linear'},
    'dT_dx':    {'dimensions': dimensions, 'layers': model_layers, 'nodes': nodes_per_layer, 'activation_function': 'swish', 'activation_final': 'linear'},
    'doc':      {'dimensions': dimensions, 'layers': model_layers, 'nodes': nodes_per_layer, 'activation_function': 'tanh', 'activation_final': 'sigmoid'},
    'ddoc_dt':  {'dimensions': dimensions, 'layers': model_layers, 'nodes': nodes_per_layer, 'activation_function': 'swish', 'activation_final': 'mish'} #MTU, change
}
#question: do I need to add 'thermal contact' to this list?
equation_list = ['heat equation', 'coupled heat equation', 'cure kinetics non-touching', 'cure kinetics', 'dT_dt', 'dT_dx', 'ddoc_dt', 'convective heat tansfer below', 'conductive heat transfer',
    'convective heat tansfer above', 'thermal contact', 'constraint_T', 'constraint_dT_dt', 'constraint_A_dT_dt']     # Note that some are body and some are boundary
#Make these transfer_mode dependent:

part_equation_list = transfer_mode_definition[transfer_mode]['part_equation_list']
tool_equation_list = transfer_mode_definition[transfer_mode]['tool_equation_list']
air_equation_list  = transfer_mode_definition[transfer_mode]['air_equation_list']
# blanket_equation_list  = transfer_mode_definition[transfer_mode]['blanket_equation_list']
loss_weightings_master = transfer_mode_definition[transfer_mode]['loss_weightings_master']

### CONFLICTING CONSTRAINTS ADDITION ### #$
speed_scale = 1. #$
trench_width = {#0.1 * speed_scale
    # GEs      #$
    'coupled heat eq': 1e0 * speed_scale,
    'doc_pde_log': 1.e-1 * speed_scale, # 1e0
    'doc_pde_lin': 2.e-2 * speed_scale, # 2e-02 for bias = 1
    'convection_1': 3.e-1 * speed_scale,
    'convection_2': 3.e-1 * speed_scale,
    # HCs
    'FEM air T': 3.e-3 * speed_scale,
    'constraint_T': 0. * speed_scale,
    'constraint_A_T': 0. * speed_scale,
    'constraint_A_dT_dt': 0. * speed_scale,
    'constraint_doc_soft': 0. * speed_scale

#    'cure_duration': 1. * speed_scale # Soft Constraints don't get trenchified
#    'loss_': 1. * speed_scale
#    'loss_': 1. * speed_scale
}

loss_limits = {**trench_width} #$


#################################################################
# Heat Problem
#################################################################
T0 = 20                 # C   
T_min = 10               # C
T_max = 220            # C
#h1 = 50 *(3600*3600*3600)              # MTU (kg*m/s^2*m/s)/m^2K -> (kg*mm/hr^2*mm/hr)/mm^2K
#h2 = 100 *(3600*3600*3600)                # MTU (kg*m/s^2*m/s)/m^2K -> (kg*mm/hr^2*mm/hr)/mm^2K
h1 = 400 *(3600*3600*3600) #question: again, are these heat transfer coefficients? Where did they come from?
# h2 = 5 *(3600*3600*3600)
###### CJE Trying to walk down HTC to get part to cure/heat all the way through. Based on "light insulating blanket" from 2/12/2025 GT email
# h2 = 4.75, then 4.5, then 4.0, then 3.5, then 3.0, then 2.5, then 2.0, then 1.5. Could step a little faster at the beginning with a 0.0005 learning rate
h2 = 2 *(3600*3600*3600)
doc_0 = 0.05            # Unitless, initial degree of cure
# question: is thickness infinite the same as no thickness? why does that work?
thickness_infinite = 0.000 # mm, just simulating this much of the air above and below, although not using the correct physics to simulate it
thickness_tool = 0.000 # CJE for Blanket Problem 0.009 *100    # MTU cm, aka 0.354in
# thickness_part = 0.01288 *100    # MTU cm, aka 0.591in
thickness_part = 1.288 #0.008 *100 # MTU cm, aka 0.314in
# thickness_part = 0.017 *100 # MTU cm, aka 0.314in
T_end = T_end_C #MTU

#CHECKPOINT - 1/31/2025
#################################################################
# Time Array
#################################################################
t_min = 0               # s
t_max = 4.5             # hours CJE of output, not of "cycle"
target_dur = t_max
loss_threshold_for_duration_minimization = 4.5e-1 # Initial optimization threshold for cure duration loss. This is approximately 76 seconds away from target
cure_loss_min = 1e-1 # This sets a soft min for the cure loss threshold (can be slightly lower (<2%) due to the way it's calculated. This is approximately 36 seconds away from target, (0.5% error at 2 hours)
total_loss_threshold_for_duration_minimization = 5e0 # Initial optimization threshold for total loss
total_loss_min = 8e-1 # This sets a soft min for the total loss threshold (can be slightly lower (<2%) due to the way it's calculated. This should be the MAXIMUM weighted loss you would tolerate in an acceptable model.
duration_increment_factor = 0.99
threshold_increment_factor = 0.98
dur_low_limit = 2 # hours. Sets lower limit for target_dur
#################################################################
# Component and Boundary Definition
#################################################################
x_min = -thickness_infinite
x_max = thickness_tool + thickness_part + thickness_infinite
components = {
    'air': Component('Air', thickness_infinite, t_min, t_max, network_definition, air_equation_list, transfer_mode, train_mode, path_for_models, transfer_mode_definition[transfer_mode]['non-trainable']),
    'tool': Component('Tool', thickness_infinite, t_min, t_max, network_definition, tool_equation_list, transfer_mode, train_mode, path_for_models, transfer_mode_definition[transfer_mode]['non-trainable']),
    'part': Component('Composite Complicated', thickness_part, t_min, t_max, network_definition, part_equation_list, transfer_mode, train_mode, path_for_models, transfer_mode_definition[transfer_mode]['non-trainable']),
    # cooper
    #'blanket': Component('Thin Blanket', thickness_infinite, t_min, t_max, network_definition, blanket_equation_list, transfer_mode, train_mode, path_for_models, transfer_mode_definition[transfer_mode]['non-trainable']),
}
boundaries = [
    Boundary(components['tool'], components['part'], t_min, t_max, ['convective heat transfer'], h1), # CJE Should we put this in as TC or Conductive HT?
    Boundary(components['part'], components['air'], t_min, t_max, ['convective heat transfer'], h2),
    # cooper
    #question: what to do with h2 at the end? delete bc it's conductive not convective? #VTN, yes, delete
    #Boundary(components['part'], components['blanket'], t_min, t_max, ['thermal contact']),
]
if (transfer_mode in [0]):
    boundaries = []

#################################################################
# Constraints (hard coded for now)
#################################################################
#question: can we talk through these constraints?
# constraint_T is tuples of [x_min, x_max, t_min, t_max, T_min, T_max, weighting], to allow weightings; these are soft Dirchelet BCs. The x are already normalized in this list.
# This needs to change character to be a region of {x,t}-space which means masking the residuals for membership in the region
# Note that v1 of the code used x, t rather than x_min, x_max, t_min, t_max while updating the code mind the difference
# The "be hot during hours 1-3" condition with the low weighting is meant to encourage the optimizer towards states where it heats up quickly; 
# will need to knock that weighting down to zero at some transfer_mode. Actually, turning that off.
constraint_T = [[thickness_tool, thickness_part, t_min, t_max, 20, 170, 10]] #MTU #VALIDATION
# constraint_T = [[0, 30, t_min, t_max, T_min, 200, 10],[0, 30, 2, 4, 100, 200, 10]] #MTU

# Temperature constraint for training mode 0 when there's nothing else to force the part to a high temperature
constraint_T_up = [[0, 30, 2*60*60, 4*60*60, (340-32)*5/9, (360-32)*5/9, 1]] #MTU

# Constraint_T for air. Really this should be done differently. This is a soft constraint; there's also a hard constraint mentioned below.
# constraint_A_T = [[0, 1, t_max*0.99, t_max, T0*0.99, T0, 0.1]]
# constraint_A_T = [[0, 30, t_max*0.99, t_max, T0*0.99, T0, 0.1],[0, 20, t_min, t_max, T0, T_max, 10]] #MTU
# constraint_A_T = [[0, 30, t_max*0.99, t_max, T0*0.99, T0, .1], [0, 20, 1, target_dur-1, 180, T_max, 10],[0, 20, t_min, t_max, T0, T_max, 10]] #VTN
# constraint_A_T = [[0, 30, t_max*0.99, t_max, T0*0.99, T0, 1], [0, 20, 1, 1.75, 70, T_max, 1], [0, 20, t_min, t_max, T0, T_max, 1000]] #CJE
constraint_A_T = [[0, thickness_infinite + thickness_tool + thickness_part, t_max*0.99, t_max, T0, T_end*1.01, 1], #I think I have to weight this higher because its a shorter time segment CJE #VALIDATION
                  #[0, thickness_infinite + thickness_tool + thickness_part, 1, 1.75, 70, T_max, 1], #tried removing, lets see what happens
                  [0, thickness_infinite + thickness_tool + thickness_part, t_min, t_max, 20, 170, 10]] #CJE #VALIDATION

# constraint_A_T = [[0, 1, t_min, t_max, T0, T_max, 10]]


# constraint_doc_soft is tuples of [x_min, x_max, t_min, t_max, doc_min, doc_max, weighting], soft enforced conditions on DOC
# constraint_doc_soft = [[0, 30, t_max*0.9, t_max, 0.8055, 0.8490, 1e4]] #MTU
# constraint_doc_soft = [[0, 30, t_max*0.9, t_max, 0.89, 0.95, 1e4]] #MTU
#constraint_doc_soft = [[0, 30, 3*1.1, t_max, 0.94, 0.965, 1e4]]
# constraint_doc_soft = [[0, 30, target_dur + -0.25, t_max, 0.95, 0.99, 1e4]] #VTN, originally adding half an hour, this worked
# constraint_doc_soft = [[0, 30, target_dur + -0.25, t_max, 0.875, 0.95, 1e4]] #VTN, originally adding half an hour
constraint_doc_soft = [[0, 30, t_max-0.01, t_max, 0.8, 0.99, 1e4]] #VTN, originally adding half an hour #CJE modifying because it might not be possible given the boundary conditions #VALIDATION

# constraint_dT_dt is tuples of [x_min,x_max,t_min,t_max,dT/dt_min,dT/dt_max,weighting], soft enforced conditions on dT_dt
# constraint_dT_dt = [[0, 30, (130-32)*5/9, (300-32)*5/9, 1.*5/9/60,   5.*5/9/60, 1]]
constraint_dT_dt = [[0, 30, t_min, t_max, 1.*5/9/60*3600,   5.*5/9/60*3600, 1]] #MTU ###############Test this next


# constraint_A_dT_dt is tuples of [x_min,x_max,T_min,T_max,dT/dt_min,dT/dt_max]. 
#constraint_A_dT_dt = [[0, 30, T_min, T_max, -6.*5/9/60, 10.*5/9/60, 1]] #MTU
constraint_A_dT_dt = [[0, 30, T_min, T_max, -300, 300 , 1]] #VALIDATION
# for all x, for all temperatures create this rate of change the air temp cannot change more than 300 celcius an hour


# constraint_T_hard is tuples of [x_min,x_max,t,T]; these are hard-enforced Dirchelet BCs.
constraint_T_hard = [[0, 30, 0, T0]] #MTU
# This isn't used in the code below, but is used in the manual determination of the particular (g) and factor (f) functions
# T_g(x,t) = ((T0-T_min)/(T_max-T_min)) * (1-t)   # (normalized time)
# T_f(x,t) = tanh(30t)
    
# dT_dt_g(x,t) = 0
# dT_dt_f(x,t) = 1
    
# dT_dx_g(x,t) = 0
# dT_dx_f(x,t) = 1
    
# constraint_A_hard is tuples of [x_min,x_max,t,T], but the x is ignored as it has no position dependence. These are hard-enforced. There are no soft-enforced constraint_A's at all right now.
#constraint_A_hard = [[0, 1, 0, T0]]
# A_g(x,t) = ((T0-T_min)/(T_max-T_min)) * (1-t)
# A_f(x,t) = tanh(30t)

# constraint_doc_hard is tuples of [x_min, x_max, t, doc]. These are hard-enforced.
constraint_doc_hard = [[0, 30, 0, doc_0]]
# This isn't used in the code below, but is used in the manual determination of the particular (g) and factor (f) functions
# doc_g(x,t) = doc_0 * relu(1-4t)   # (normalized time)
# doc_f(x,t) = tanh(30t)

# ddocdt_g(x,t) = 0
# ddocdt_f(x,t) = 1

#################################################################
# Normalize Constraints #MTU
#################################################################
# constraint_T = [[point[0],point[1],point[2]/t_max,point[3]/t_max,(point[4]-T_min)/(T_max-T_min),(point[5]-T_min)/(T_max-T_min),point[6]] for point in constraint_T]
# constraint_T_up = [[point[0],point[1],point[2]/t_max,point[3]/t_max,(point[4]-T_min)/(T_max-T_min),(point[5]-T_min)/(T_max-T_min),point[6]] for point in constraint_T_up]
# constraint_A_T = [[point[0],point[1],point[2]/t_max,point[3]/t_max,(point[4]-T_min)/(T_max-T_min),(point[5]-T_min)/(T_max-T_min),point[6]] for point in constraint_A_T]
# constraint_T_hard = [[point[0],point[1],point[2]/t_max,(point[3]-T_min)/(T_max-T_min)] for point in constraint_T_hard]
# constraint_doc_hard = [[point[0],point[1],point[2]/t_max,point[3]] for point in constraint_T_hard]
# #constraint_dT_dt = [[point[0],point[1],(point[2]-T_min)/(T_max-T_min),(point[3]-T_min)/(T_max-T_min),(point[4])/((T_max-T_min)/(t_max-t_min)),(point[5])/((T_max-T_min)/(t_max-t_min)),point[6]] for point in constraint_dT_dt]
# constraint_doc_soft = [[point[0],point[1],point[2]/t_max,point[3]/t_max,point[4],point[5],point[6]] for point in constraint_doc_soft]
# constraint_A_dT_dt = [[point[0],point[1],(point[2]-T_min)/(T_max-T_min),(point[3]-T_min)/(T_max-T_min),(point[4])/((T_max-T_min)/(t_max-t_min)),(point[5])/((T_max-T_min)/(t_max-t_min)),point[6]] for point in constraint_A_dT_dt]

#################################################################
# File names
#################################################################
loss_history = path_for_models + 'loss_history_mode_' + str(transfer_mode) + '.csv'

#################################################################
# Load Normalized Mode 0 Constraints
#################################################################
load_FEM = transfer_mode in [0, 1, 2, 3]
if load_FEM:
    data_part_T_df = pd.read_csv(data_file_part_T, dtype=np.float32)
    data_part_T_t = np.array(data_part_T_df['t_norm'])
    data_part_T_x = np.array(data_part_T_df['x_norm'])
    data_part_T_T = np.array(data_part_T_df['T_norm'])
    data_part_T_t = data_part_T_t.reshape(len(data_part_T_t),1)
    data_part_T_x = data_part_T_x.reshape(len(data_part_T_x),1)
    data_part_T_T = data_part_T_T.reshape(len(data_part_T_T),1)

    data_tool_T_df = pd.read_csv(data_file_tool_T, dtype=np.float32)
    data_tool_T_t = np.array(data_tool_T_df['t_norm'])
    data_tool_T_x = np.array(data_tool_T_df['x_norm'])
    data_tool_T_T = np.array(data_tool_T_df['T_norm'])
    data_tool_T_t = data_tool_T_t.reshape(len(data_tool_T_t),1)
    data_tool_T_x = data_tool_T_x.reshape(len(data_tool_T_x),1)
    data_tool_T_T = data_tool_T_T.reshape(len(data_tool_T_T),1)

    data_air_T_df = pd.read_csv(data_file_air_T, dtype=np.float32)
    data_air_T_t = np.array(data_air_T_df['t_norm'])
    data_air_T_x = np.array(data_air_T_df['x_norm'])
    data_air_T_T = np.array(data_air_T_df['T_norm'])
    data_air_T_t = data_air_T_t.reshape(len(data_air_T_t),1)
    data_air_T_x = data_air_T_x.reshape(len(data_air_T_x),1)
    data_air_T_T = data_air_T_T.reshape(len(data_air_T_T),1)

    # Cooper
    # data_blanket_T_df = pd.read_csv(data_file_blanket_T, dtype=np.float32)
    # data_blanket_T_t = np.array(data_blanket_T_df['t_norm'])
    # data_blanket_T_x = np.array(data_blanket_T_df['x_norm'])
    # data_blanket_T_T = np.array(data_blanket_T_df['T_norm'])
    # data_blanket_T_t = data_blanket_T_t.reshape(len(data_blanket_T_t),1)
    # data_blanket_T_x = data_blanket_T_x.reshape(len(data_blanket_T_x),1)
    # data_blanket_T_T = data_blanket_T_T.reshape(len(data_blanket_T_T),1)


    data_part_DOC_df = pd.read_csv(data_file_part_DOC, dtype=np.float32)
    data_part_DOC_t = np.array(data_part_DOC_df['t_norm'])
    data_part_DOC_x = np.array(data_part_DOC_df['x_norm'])
    data_part_DOC_DOC = np.array(data_part_DOC_df['DOC'])
    data_part_DOC_t = data_part_DOC_t.reshape(len(data_part_DOC_t),1)
    data_part_DOC_x = data_part_DOC_x.reshape(len(data_part_DOC_x),1)
    data_part_DOC_DOC = data_part_DOC_DOC.reshape(len(data_part_DOC_DOC),1)

#################################################################
# Main Code
#################################################################
# question: I shouldn't need to change things in here, right? 
if ((train_mode == 0) or (train_mode == 1)):
    train_model = True #$
    print("Training mode = ", train_mode)
    print("Transfer mode = ", transfer_mode)

    #Create Graph
    training_process_data = {'epochs': [], 'loss': [], 'time': [], 'loss counter': []}
    if plot_loss:
        lossplot = plt.figure()
        lossplot.show()
        lossplot.patch.set_facecolor((0.1,0.1,0.1))
        axes = plt.gca()
        axes.set_xlim(0, 10)
        axes.set_ylim(0, +1)
        axes.set_facecolor((0.1,0.1,0.1))
        axes.spines['bottom'].set_color((0.9,0.9,0.9))
        axes.spines['top'].set_color((0.9,0.9,0.9))
        axes.spines['left'].set_color((0.9,0.9,0.9))
        axes.spines['right'].set_color((0.9,0.9,0.9))
        axes.xaxis.label.set_color((0.9,0.9,0.9))
        axes.yaxis.label.set_color((0.9,0.9,0.9))
        axes.tick_params(axis='x', colors=(0.9,0.9,0.9))
        axes.tick_params(axis='y', colors=(0.9,0.9,0.9))
        line, = axes.plot(training_process_data['epochs'], training_process_data['loss'], 'r-') 

    min_loss = 1e6 #$
    min_man_loss = 1e6 #$
    min_stage_loss = 1e6 #$
    min_past_thresh_loss = 1e6 #$
    current_loss = min_stage_loss
    loss_counter = 0 #$
    threshold_crossed = False #$
    current_epoch = 0 #$
    stage = 1 #$
    last_time = time()

    zero_feed = np.column_stack(np.zeros(batch))
    zero_feed = tf.Variable(zero_feed.reshape(len(zero_feed[0]),1), trainable=True, dtype=tf.float32)

    one_feed = np.column_stack(np.ones(batch))
    one_feed = tf.Variable(one_feed.reshape(len(one_feed[0]),1), trainable=True, dtype=tf.float32)

    while train_model:
        model_saved = False #$
        # Create tensors to feed to TF
        for component in components:
            components[component].define_residual_set(batch)
        for boundary in boundaries:
            boundary.define_residual_set(batch)

        with tf.GradientTape(persistent=True) as tape_3:
            with tf.GradientTape(persistent=True) as tape_2:
                with tf.GradientTape(persistent=True) as tape_1:
                    # Watch parameters
                    for component in components:
                        tape_1.watch(components[component].x_feed)
                        tape_1.watch(components[component].t_feed)
                        if ('cure kinetics non-touching' in components[component].equation_list):
                            tape_1.watch(components[component].Tck_inf_feed)
                    for boundary in boundaries:
                        tape_1.watch(boundary.t_feed)
                        tape_1.watch(boundary.c1_x_feed)
                        tape_1.watch(boundary.c2_x_feed)
                    tape_1.watch(zero_feed)
                    tape_1.watch(one_feed)

                    # Define functions
                    for component in components:
                        components[component].define_functions()
                    for boundary in boundaries:
                        boundary.define_functions()

                # Watch parameters
                for component in components:
                    tape_2.watch(components[component].x_feed)
                    tape_2.watch(components[component].t_feed)
                    if ('cure kinetics non-touching' in components[component].equation_list):
                        tape_2.watch(components[component].Tck_inf_feed)
                for boundary in boundaries:
                    tape_2.watch(boundary.t_feed)
                    tape_2.watch(boundary.c1_x_feed)
                    tape_2.watch(boundary.c2_x_feed)
                tape_2.watch(zero_feed)
                tape_2.watch(one_feed)

                # Take derivitives
                for component in components:
                    components[component].define_derivative_functions(tape_1)
                for boundary in boundaries:
                    boundary.define_derivative_functions(tape_1)

            #take double derivatives
            for component in components:
                components[component].define_double_derivative_functions(tape_2)

            # Model losses
            for component in components:
                components[component].define_body_losses()
            for boundary in boundaries:
                boundary.define_boundary_losses()
                boundary.distribute_boundary_losses()

                # Define loss weightings
            for component in components:
                components[component].reset_loss_weightings()
                for name in loss_weightings_master:
                    try:
                        components[component].loss_weightings[name] = loss_weightings_master[name]
                    except KeyError:
                        pass
                #print('Component: ', component)
                #components[component].calculate_weighted_losses(stage)
                if not LoCaTA_active:
                    components[component].calculate_weighted_losses()

            # Trenchify
            if LoCaTA_active:
                for component in components:
                    components[component].trench_losses(loss_limits)
            ''' THIS IS THE PROBLEM RIGHT HERE LOOK NO FURTHER. THERE IS A DISCONNECTED GRADIENT ISSUE THAT STEMS FROM RIGHT HERE (I think it was because I was tabbed too far back and wasn't in the gradient tape. Let's see!
            for component in components:
                if stage == 1:
                    components[component].stage1losses = {**components[component].hard_constraints}
                    print(components[component].stage1losses)
                    for name in components[component].stage1losses:
                        print("losses_weighted: ", components[component].losses_weighted[name])
                        print("stage1losses: ", components[component].stage1losses[name])
                    components[component].loss_total = sum([components[component].stage1losses[name] for name in components[component].stage1losses])
                    print(components[component].loss_total)
                if stage == 2:
                    components[component].stage2losses = {**components[component].hard_constraints, **components[component].governing_eqs}
                    components[component].loss_total = sum([components[component].losses_weighted[name] for name in components[component].stage2losses])
                if stage == 3:
                    components[component].stage3losses = {**components[component].losses}
                    components[component].loss_total = sum([components[component].losses_weighted[name] for name in components[component].stage3losses])
                if math.isnan(components[component].loss_total):
                    print("a")
            '''
            '''    
            # Stage loss divisions
            if stage == 1:
                X_losses = {**hard_constraints}
                F_losses = {**hard_constraints}
                M_losses = {**hard_constraints}
                stage_losses = {**hard_constraints}

            if stage == 2:
                X_losses = {**governing_eqs, **hard_constraints}
                F_losses = {**governing_eqs, **hard_constraints}
                M_losses = {**governing_eqs, **hard_constraints}
                del X_losses['loss_PDE']
                del M_losses['loss_PDE']
                del F_losses['loss_MEQ']
                stage_losses = {**governing_eqs, **hard_constraints}

            if stage == 3:
                X_losses = {**governing_eqs, **hard_constraints, **soft_constraints}
                F_losses = {**governing_eqs, **hard_constraints, **soft_constraints}
                M_losses = {**governing_eqs, **hard_constraints, **soft_constraints}
                del X_losses['loss_PDE']
                del M_losses['loss_PDE']
                del F_losses['loss_MEQ']
                stage_losses = {**governing_eqs, **hard_constraints, **soft_constraints}

            X_loss_total = sum([X_losses[name] for name in X_losses])
            F_loss_total = sum([F_losses[name] for name in F_losses])
            M_loss_total = sum([M_losses[name] for name in M_losses])
            stage_loss_total = sum([stage_losses[name] for name in stage_losses])
            stage_loss = k.get_value(stage_loss_total)
            '''
            # Loss totals
            if not LoCaTA_active:
                loss_total = sum([components[component].loss_total for component in components])
                current_loss = k.get_value(loss_total)
            if LoCaTA_active:
                loss_total= 0
                passage_granted = False
                while not passage_granted:
                    for component in components:
                        if stage == 1:
                            components[component].stage_losses = {**components[component].hard_constraints}
                        if stage == 2:
                            components[component].stage_losses = {**components[component].governing_eqs, **components[component].hard_constraints}
                        if stage == 3:
                            components[component].stage_losses = {**components[component].governing_eqs, **components[component].hard_constraints, **components[component].soft_constraints}
                        components[component].loss_total = sum([components[component].stage_losses[name] for name in components[component].stage_losses])
                        loss_total += components[component].loss_total
                    current_loss = k.get_value(loss_total)
                    stage_loss = current_loss # CHANGE IN THE FUTURE
                    passage_granted = True
                    if (stage == 1) and (current_loss == 0):
                        print("Moving on from stage 1 to stage 2! \nCurrent Loss: ", current_loss, "\n")
                        stage += 1
                        min_stage_loss = 10000
                        passage_granted = False
                        current_loss = 1

                    if (stage == 2) and (current_loss == 0):
                        print("Moving on from stage 2 to stage 3! \nCurrent Loss: ", current_loss, "\n")
                        stage += 1
                        min_stage_loss = 10000
                        threshold_crossed = True
                        #for component in components:
                        #    components[component].bias += 1
                        passage_granted = False
        # Optimize: adjust targets for loss calculations where applicable
        if not LoCaTA_active:
            for component in components:
                components[component].optimize_targets()


        # Save Model #$
        if (min_stage_loss > current_loss): # or optimized: CHANGE
            model_saved = True
            min_stage_loss = current_loss
            #print('Saving models')
            for component in components:
                components[component].save_models()
                #print('Saving model ' + str(components[component].component_number)) #CJE for debugging
            # optimized = False CHANGE

        # Keep going or not #$
        if model_saved:
            loss_counter = 0
        else:
            loss_counter += 1
        if not threshold_crossed:
            if loss_counter > 20000:
                print("Conflicting Constraints Detected!")
                sys.exit()
        elif loss_counter > 15000:
                print("Training Completed")
                sys.exit()
            

        # Train the model
        for component in components:
            if components[component].material_name != 'Air':
                components[component].train_models(tape_3)

        # Take a break and report
        if (current_epoch % 100 == 0) or (model_saved):
            print("\n\n\nStep " + str(current_epoch) + " -------------------------------")
            if not LoCaTA_active:
                print("Unweighted -------------------------------")
                for component in components:
                    print(component + ": ", [name + ": " + "{:.3e}".format(k.get_value(components[component].losses[name])) for name in components[component].losses])
                print("Weighted -------------------------------")
                for component in components:
                    print(component + ": ", [name + ": " + "{:.3e}".format(k.get_value(components[component].losses_weighted[name])) for name in components[component].losses_weighted])
            if LoCaTA_active:
                print("\n----- Governing Equations -----")
                for component in components:
                    #print(name + ": ", "{:.3e}".format(k.get_value(governing_eqs[name])))  # , "  Target: ", "{:.3e}".format(model_accuracy[name]))
                    print(component + ": ", [name + ": " + "{:.3e}".format(k.get_value(components[component].governing_eqs[name])) for name in components[component].governing_eqs])
                print("\n----- Hard Constraints -----")
                for component in components:
                    #print(name + ": ", "{:.3e}".format(k.get_value(hard_constraints[name])))  # , "  Target: ", "{:.3e}".format(model_accuracy[name]))
                    print(component + ": ", [name + ": " + "{:.3e}".format(k.get_value(components[component].hard_constraints[name])) for name in components[component].hard_constraints])
                print("\n----- Soft Constraints -----")
                for component in components:
                    #print(name + ": ", "{:.3e}".format(k.get_value(soft_constraints[name])))
                    print(component + ": ", [name + ": " + "{:.3e}".format(k.get_value(components[component].soft_constraints[name])) for name in components[component].soft_constraints])
            print("-------------------------------")
            print("Stage ", str(stage))
            print("Threshold Crossed: \t", str(threshold_crossed))
            print("Loss Counter: \t\t", str(loss_counter))
            if LoCaTA_active:
                print("Stage Loss:\t\t", "{:.3e}".format(current_loss))
            if not LoCaTA_active:
                print("Total Loss:\t\t", "{:.3e}".format(current_loss))

            if (time_reporting):
                print("Calculation time for last period: ", "{:.1f}".format(round(time() - last_time, 1)))
            last_time = time()

            
            #Report
            current_time = round(time() - start_time,2) 
            training_process_data['epochs'].append(current_epoch)
            training_process_data['loss'].append(current_loss)
            training_process_data['time'].append(current_time)
            training_process_data['loss counter'].append(loss_counter)
            if plot_loss:
                line.set_xdata(training_process_data['epochs'])
                line.set_ydata(training_process_data['loss'])
                plt.draw()
                plt.pause(1e-17)
                plt.xlim(0,current_epoch)
                plt.ylim(current_loss/5, current_loss*10)  

            #np.savetxt(loss_history, np.column_stack((training_process_data['time'],training_process_data['epochs'],training_process_data['loss'])), comments="", header="Time(s),Epoch,Total Loss", delimiter=',', fmt='%f')
            np.savetxt(loss_history, np.column_stack((training_process_data['epochs'],training_process_data['loss'], training_process_data['loss counter'])), comments="", header="Epoch,Total Loss,Loss Counter", delimiter=',', fmt='%f')
        current_epoch += 1


        # print(k.get_value(components['part'].losses['doc_pde_lin']))
       # if (k.get_value(components['part'].losses['doc_pde_lin']) < 1.5e-4) and transfer_mode in [3]:
           # break



# question: what is this? I also shouldn't need to change this, right?
#################################################################
# Predicting
#################################################################
# Inputs
time_divisions = 121
nodes_per_component = 21
x_feed = np.arange(nodes_per_component)/(nodes_per_component-1)
one_feed = np.ones(nodes_per_component)
dt = int((t_max-t_min)*60/(time_divisions-1)) #MTU convert to minutes
results_initialized = np.ones(1)*(x_min)
for component in components:
    if (components[component].thickness == 0):
        results_initialized = np.append(results_initialized, results_initialized[-1])
    else:
        results_initialized = np.append(results_initialized, results_initialized[-1] + np.arange(nodes_per_component)/(nodes_per_component-1)*components[component].thickness)
results_initialized = np.insert(results_initialized, 0, 0)

reports = {}
report_time = []
results = {}
for name in prediction_names:
    results[name] = results_initialized.copy()
for i in range(0, int((t_max-t_min)*60)+1*dt, dt):
    print('time =', i/60) #reports in hours
    predictions = {}
    predictions['A'] = []
    for name in prediction_names:
        predictions[name] = []
    component_predictions = {}
    report_time.append(i/60)
    for component in components:
        if (components[component].thickness == 0):
            nodes_this_component = 1
        else:
            nodes_this_component = nodes_per_component
        t_feed = np.ones(nodes_this_component)*i/60 #MTU
        if components[component].material.name == 'Air':
            if predictions['A'] == []:
                # predictions['A'] = components[component].net['T'].values([np.ones(1), np.ones(1)*(t_min + i/(t_max-t_min))])[0][0] * tf.tanh(30.0 * (t_min + i/(t_max-t_min))) + ((T0-T_min)/(T_max-T_min)) * (1 - (t_min + i/(t_max-t_min)))
                predictions['A'] = components[component].net['T'].values([np.zeros(1), np.ones(1)*float(i/60)])[0][0] * tf.tanh(30.0 * (float(i/60)/(t_max-t_min))) + ((T0)) * (1 - (float(i/60)/(t_max-t_min))) #MTU
                # predictions['A'] = components[component].net['T'].values([np.zeros(1), np.ones(1)*float(i/60)])[0][0]

        #print(tf.multiply(components[component].net['T'].values([x_feed, t_feed]), tf.reshape(tf.cast(tf.convert_to_tensor(tf.tanh(30.0 * t_feed)), tf.float32),shape=(1,nodes_this_component))))
        component_predictions['T'] = tf.multiply(components[component].net['T'].values([x_feed * components[component].thickness, t_feed]), tf.reshape(tf.cast(tf.convert_to_tensor(tf.tanh(30.0 * (t_feed-t_min)/(t_max-t_min))), tf.float32),shape=(1,nodes_this_component))).numpy()[:,0] + ((T0)) * (1.0 - (t_feed-t_min)/(t_max-t_min)) #MTU
        # component_predictions['T'] = np.reshape(components[component].net['T'].values([x_feed * components[component].thickness, t_feed]).numpy(), (nodes_this_component))
        component_predictions['T'] = np.ones(nodes_this_component) + component_predictions['T']
        predictions['T'].extend(component_predictions['T'].tolist())
        try:
            component_predictions['doc'] = tf.multiply(components[component].net['doc'].values([x_feed * components[component].thickness, t_feed]), tf.reshape(tf.cast(tf.convert_to_tensor(tf.tanh(30.0 * (t_feed-t_min)/(t_max-t_min))), tf.float32),shape=(1,len(components[component].net['T'].values([x_feed, (t_feed-t_min)/(t_max-t_min)]))))).numpy()[:,0] + doc_0 * k.relu(1.0 - 4.0 * (t_feed-t_min)/(t_max-t_min)).numpy() #MTU
            # component_predictions['doc'] = np.reshape(components[component].net['doc'].values([x_feed * components[component].thickness, t_feed]).numpy(), (nodes_this_component))
        except:
            component_predictions['doc'] = np.zeros(shape=(nodes_this_component))
        predictions['doc'].extend(component_predictions['doc'].tolist())
        # try:
        #     component_predictions['dT_dt'] = np.reshape(components[component].net['dT_dt'].values([x_feed * components[component].thickness, t_feed]).numpy(), (nodes_this_component)).tolist()
        #     predictions['dT_dt'].extend(component_predictions['dT_dt'])
        # except:
        #     pass
        # try:
        #     component_predictions['dT_dx'] = np.reshape(components[component].net['dT_dx'].values([x_feed * components[component].thickness, t_feed]).numpy(), (nodes_this_component))
        # except:
        #     component_predictions['dT_dx'] = np.zeros(shape=(nodes_this_component))
        # predictions['dT_dx'].extend(component_predictions['dT_dx'].tolist())
        # #.shape=(1,nodes_this_component))).numpy()[:,0]
        # #print(component_predictions['T'])
        # #component_predictions['T'] = np.ones(nodes_this_component)*T_min + component_predictions['T']*(T_max - T_min)
        # #predictions['T'].extend(component_predictions['T'].tolist())
        # try:
        #    component_predictions['ddoc_dt'] = tf.multiply(components[component].net['ddoc_dt'].values([x_feed* components[component].thickness, t_feed]), 1).numpy()[:,0]
        # except:
        #    component_predictions['ddoc_dt'] = np.zeros(shape=(nodes_this_component))
        # predictions['ddoc_dt'].extend(component_predictions['ddoc_dt'].tolist())
    for name in prediction_names:
        reports[name] = np.insert(predictions[name], 0, k.get_value(predictions['A']))
        reports[name] = np.insert(reports[name], 0, i/60)
        results[name] = np.column_stack((results[name], reports[name]))
        
for name in prediction_names:
    predictions_file_name = path_for_models + 'predictions_mode_' + str(transfer_mode) + '_' + name + '.csv'
    np.savetxt(predictions_file_name, results[name], delimiter=',') 
# if True:
#     df = pd.read_csv('Working Models v22/predictions_mode_3_T.csv', header=None)
#
#     x_values = df.iloc[0, 1:115]
#     y_values = df.iloc[2, 1:115]
#
#     # Create a plot
#     plt.figure(figsize=(10, 5))
#     plt.plot(x_values, y_values)
#     plt.grid(True)
#     plt.show()
print("Job's done")