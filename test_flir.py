import os

net = "res50"
dataset = "flir_vis"
dataset_t = "flir_tr"
start_epoch = 1
max_epochs = 1
s = 1

for i in range(start_epoch, max_epochs + 1):
    model_dir = "./models/{}/{}/target_{}_eta_0.1_local_True_global_True_gamma_3_session_{}_epoch_{}_step_4000.pth".format(net,"vis2tr",dataset_t,s,i)

    command = "python test_net.py --dataset {} --dataset_t {} --net {}  --load_name {}".format(
        dataset,dataset_t,net,model_dir
    )
    os.system(command)
