from exm_deeplearn_lib.exmsyn_models import unet_like
from exm_deeplearn_lib.exmsyn_compile import gen_unet_batch_v2
from exm_deeplearn_lib.exmsyn_network import masked_binary_crossentropy
from exm_deeplearn_lib.exmsyn_network import masked_accuracy, masked_error_pos, masked_error_neg
from exm_deeplearn_lib.exmsyn_network import DeepNeuralNetwork
from keras.optimizers import SGD
import pickle


file_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_DNN/RETRAIN_2020/data/RAWDATA_SYNAPSEMASKS_FORTRAINING_compiled_nrrd/'
img_mask_junk_names = [(file_path+'antennallobe1_ch1_nc82_20180625_500mMsalt_5544-5045-9568_RAWDATA.nrrd', file_path+'antennallobe1_ch1_nc82_20180625_500mMsalt_5544-5045-9568_SYNAPSEMASKS.nrrd', file_path+'junk.nrrd'), \
    (file_path+'ellipsoidbody1_ch2_nc82_minus_ch1_pC1x0p02_20180917_6921-3570-6594_RAWDATA.nrrd',file_path+'ellipsoidbody1_ch2_nc82_minus_ch1_pC1x0p02_20180917_6921-3570-6594_SYNAPSEMASKS.nrrd', file_path+'junk.nrrd'), \
    (file_path+'lateralhorn1_ch2_nc82_20200613_2765-6756-6642_RAWDATA.nrrd', file_path+'lateralhorn1_ch2_nc82_20200613_2765-6756-6642_SYNAPSEMASKS.nrrd', file_path+'junk.nrrd'), \
    (file_path+'mushroombody2_ch2_nc82_20200623_2496-2182-2017_RAWDATA.nrrd', file_path+'mushroombody2_ch2_nc82_20200623_2496-2182-2017_SYNAPSEMASKS.nrrd', file_path+'junk.nrrd'), \
    (file_path+'opticlobe1_ch0_nc82_20180503_4228-3823-4701_RAWDATA.nrrd', file_path+'opticlobe1_ch0_nc82_20180503_4228-3823-4701_SYNAPSEMASKS.nrrd', file_path+'junk.nrrd'), \
    (file_path+'antennallobe0_ch1_nc82_20180625_500mMsalt_5793-2782-11411_RAWDATA.nrrd', file_path+'antennallobe0_ch1_nc82_20180625_500mMsalt_5793-2782-11411_SYNAPSEMASKS.nrrd', file_path+'junk.nrrd'), \
    (file_path+'ellipsoidbody0_ch2_nc82_20180917_7527-3917-6681_RAWDATA.nrrd', file_path+'ellipsoidbody0_ch2_nc82_20180917_7527-3917-6681_SYNAPSEMASKS.nrrd', file_path+'junk.nrrd'), \
    (file_path+'mushroombody0_ch2_nc82_20180917_6210-6934-5492_RAWDATA.nrrd', file_path+'mushroombody0_ch2_nc82_20180917_6210-6934-5492_SYNAPSEMASKS.nrrd', file_path+'junk.nrrd'), \
    (file_path+'protocerebrum0_ch2_nc82_20180917_13904-10064-4442_RAWDATA.nrrd', file_path+'protocerebrum0_ch2_nc82_20180917_13904-10064-4442_SYNAPSEMASKS.nrrd', file_path+'junk.nrrd'), \
    (file_path+'opticlobe2_ch0_nc82_20200103_4068-2260-4085_RAWDATA.nrrd', file_path+'opticlobe2_ch0_nc82_20200103_4068-2260-4085_SYNAPSEMASKS.nrrd', file_path+'junk.nrrd')]

input_shape = (64,64,64)
model = unet_like(input_shape)
sgd_opti = SGD(lr=0.001, momentum=0.9, decay=0.00005, nesterov=True)
compile_args = {'optimizer':sgd_opti, 'loss':masked_binary_crossentropy, 'metrics':[masked_accuracy, masked_error_pos, masked_error_neg]}
network = DeepNeuralNetwork(model, compile_args=compile_args)

batch_sz = 16
n_gpus = 1
generator = gen_unet_batch_v2(img_mask_junk_names, crop_sz=(64,64,64), mask_sz=(24,24,24), batch_sz=batch_sz*n_gpus)
save_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_DNN/saved_unet_model_2020/model_2020_4/'
history = network.train_network(generator=generator, steps_per_epoch=100, epochs=3000, n_gpus=n_gpus, save_name=None)

with open(save_path+'history_rawdata_gen2_lr1e-3_sgd_batch64_steps100_epochs3000.pkl', 'wb') as f:
    pickle.dump(history.history, f)

network.save_whole_network(save_path+'unet_model_2020_4')
# network.save_architecture(save_path+'unet')