import os
import csv
import torch
import tools
import numpy as np
import data_load
import Resnet
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Func
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from transformer import transform_train, transform_test, transform_target, transform_train_FMN, transform_test_FMN



for dataset_number in range(1,3):
    #main config
    asm_ds = ['cifar', 'FashionMINIST0.5', 'FashionMINIST0.6', 'cifar10']
    training_target_ds = asm_ds[dataset_number]
    version = f'RESNET34_T_estimation_{training_target_ds}'

    training_iteration_eval = 10 #number of training loop

    Model =  Resnet.ResNet18_grey #estimation
    Model_2 =  Resnet.ResNet34_grey #training


    #lr steps config

    #config 1 Resnet18
    lr_c1_est = [5, 10]
    lr_c1_train = [4, 6, 8, 11, 13, 15, 20, 25]

    #config 2
    lr_c2_est = [5, 10]
    lr_c2_train = [4, 6, 8, 13, 20, 50]




    est_lr_steps = lr_c1_est
    train_lr_steps = lr_c2_train




    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device is : " + device)


    #Configuration

    cifar_config = {
        'path' : "./dataset/CIFAR.npz",
        'dataset' : 'cifar',
        'n_epoch_training' : 70,  #200/100
        'n_epoch_estimate' : 25,  #25/35/50
        'num_classes' : 3,
        'batch_size' : 64,
        'train_val_split_rate' : 0.8,
        'lr' : 0.01,  #try 0.001/0.002/0.01
        'worker_number' : 1,
        'weight_decay_est' : 0.001,  #works as a regularizer
        'weight_decay_train' : 0.01,  #works as a regularizer
        'model' : 'resnet18'
    }


    FMN5_config = {
        'path' : "./dataset/FashionMNIST0.5.npz",
        'dataset' : 'FashionMINIST0.5',
        'n_epoch_training' : 70,  #200/100
        'n_epoch_estimate' : 25,  #25/35/50
        'num_classes' : 3,
        'batch_size' : 64,
        'train_val_split_rate' : 0.8,
        'lr' : 0.01,  #try 0.001/0.002/0.01
        'worker_number' : 1,
        'weight_decay_est' : 0.001,  #works as a regularizer
        'weight_decay_train' : 0.01,  #works as a regularizer
        'model' : 'resnet18'
    }


    FMN6_config = {
        'path' : "./dataset/FashionMNIST0.6.npz",
        'dataset' : 'FashionMINIST0.6',
        'n_epoch_training' : 70,  #200/100
        'n_epoch_estimate' : 25,  #25/35/50
        'num_classes' : 3,
        'batch_size' : 64,
        'train_val_split_rate' : 0.8,
        'lr' : 0.01,  #try 0.001/0.002/0.01
        'worker_number' : 1,
        'weight_decay_est' : 0.001,  #works as a regularizer
        'weight_decay_train' : 0.01,  #works as a regularizer
        'model' : 'resnet18'
    }


    #set config
    config_dict = None

    if training_target_ds == 'cifar':
        config_dict = cifar_config

    elif training_target_ds == 'FashionMINIST0.5':
        config_dict = FMN5_config

    elif training_target_ds == 'FashionMINIST0.6':
        config_dict = FMN6_config






    print("Training data is {} with following parameters".format(config_dict['dataset']))
    print(config_dict)




    #start actual training
    for i in range(1, training_iteration_eval + 1): #random seed from 1 to 10


        print("\n Iteration : %d"%(i))


        #iteration logging data
        est_train_history = []
        est_val_history = []

        real_train_history = []
        real_val_history = []
        real_test_history = []

        #setup main data for each iteration
        random_seed = i




        #directory to save informations
        #mkdir
        model_save_dir = 'model' + '/' + training_target_ds + '/' + 'Version_%s'%(version) + '/' + 'Iteration_%d'%(i)

        if not os.path.exists(model_save_dir):
            os.system('mkdir -p %s'%(model_save_dir))


        prob_save_dir = 'prob' + '/' + training_target_ds + '/' + 'Version_%s'%(version) + '/' + 'Iteration_%d'%(i)

        if not os.path.exists(prob_save_dir):
            os.system('mkdir -p %s'%(prob_save_dir))


        matrix_save_dir = 'matrix' + '/' + training_target_ds + '/' + 'Version_%s'%(version) + '/' + 'Iteration_%d'%(i)

        if not os.path.exists(matrix_save_dir):
            os.system('mkdir -p %s'%(matrix_save_dir))


        logging_data_dir = 'log' + '/' + training_target_ds + '/' + 'Version_%s'%(version) + '/' + 'Iteration_%d'%(i)

        if not os.path.exists(logging_data_dir):
            os.system('mkdir -p %s'%(logging_data_dir))



        #load training data
        train_data = data_load.FMN_dataset(path=config_dict['path'], train=True, transform=transform_train_FMN(config_dict['dataset']), target_transform=transform_target,
                                           random_seed=random_seed)

        val_data = data_load.FMN_dataset(path=config_dict['path'], train=False, transform=transform_train_FMN(config_dict['dataset']), target_transform=transform_target,
                                         random_seed=random_seed)

        test_data = data_load.FMN_test_dataset(path=config_dict['path'], transform=transform_test_FMN(config_dict['dataset']), target_transform=transform_target)

        estimate_state = True

        est_model = Model(config_dict['num_classes'])


        #set up data_loader
        train_loader = DataLoader(dataset=train_data,
                                batch_size=config_dict['batch_size'],
                                shuffle=True, #improve the robustness of the algorithm
                                num_workers=config_dict['worker_number'],
                                drop_last=False)


        estimate_loader = DataLoader(dataset=train_data,
                                batch_size=config_dict['batch_size'],
                                shuffle=False,
                                num_workers=config_dict['worker_number'],
                                drop_last=False)


        val_loader = DataLoader(dataset=val_data,
                                batch_size=config_dict['batch_size'],
                                shuffle=False,
                                num_workers=config_dict['worker_number'],
                                drop_last=False)


        test_loader = DataLoader(dataset=test_data,
                                batch_size=config_dict['batch_size'],
                                num_workers=config_dict['worker_number'],
                                drop_last=False)


        #probability matrix
        probability_matrix = torch.zeros((len(train_data), config_dict['num_classes']))
        batch_index_num = int(len(train_data) / config_dict['batch_size'] )
        print('The amount of batch is : {}'.format(batch_index_num))
        highest_est_val_acc = 0
        highest_est_val_acc_epoch_n = 0



        #optimizer
        optimizer_adam = optim.Adam(est_model.parameters(), lr=config_dict['lr'])
        optimizer_SGD = optim.SGD(est_model.parameters(), lr=config_dict['lr'], momentum=0.9) #, weight_decay=config_dict['weight_decay_est']
        #scheduler
        est_scheduler = MultiStepLR(optimizer_SGD, milestones=est_lr_steps, gamma=0.5)

        #loss
        loss_func_ce = nn.CrossEntropyLoss()



        #cuda
        if torch.cuda.is_available:
            print("using GPU, move to device : " + device)
            est_model = est_model.to(device)
            loss_func_ce = loss_func_ce.to(device)
        else:
            print("using cpu")



        print("\nstart training the NN for estimating transition matrix")

        for epoch in range(config_dict['n_epoch_estimate']):

            print('epoch {} with LR : {}'.format(epoch + 1, est_scheduler.get_last_lr()))

            #enter training mode
            est_model.train()

            #logging part
            train_loss = 0.
            train_acc = 0.
            val_loss = 0.
            val_acc = 0.

            #training of the model
            for batch_x, batch_y in train_loader:

                #load the batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                #zeroing the gradient
                optimizer_SGD.zero_grad()

                #forward pass
                out = est_model(batch_x) #no revision
                loss = loss_func_ce(out, batch_y)

                #batch update on the model
                loss.backward() #Computes the gradient of current tensor w.r.t. graph leaves.
                optimizer_SGD.step() #


                #loss logging
                train_loss += loss.item()
                pred = torch.max(out, 1)[1]
                train_correct = (pred == batch_y).sum()
                train_acc += train_correct.item()

            #update the scheduler
            est_scheduler.step()

            #training output
            epoch_training_loss = train_loss / (len(train_data))*config_dict['batch_size']
            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(epoch_training_loss, train_acc / (len(train_data))))
            est_train_history.append( [epoch_training_loss, train_acc / (len(train_data)) ] )

            #validation of current epoch
            with torch.no_grad(): # run faster, less vrm and faster computation
                est_model.eval()
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    out = est_model(batch_x)
                    loss = loss_func_ce(out, batch_y)
                    val_loss += loss.item()
                    #print(out) # out just the output of the network, using forward pass
                    pred = torch.max(out, 1)[1] #verified, basically this will print out the prediction
                    #print(pred)
                    val_correct = (pred == batch_y).sum()
                    #print(val_correct)
                    val_acc += val_correct.item()
                    #print(val_correct.item())

            #print("Validation : number of corrected classification : {} and total number of the set : {} ".format(val_acc, len(val_data)))
            current_val_acc = val_acc / (len(val_data))
            print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data))*config_dict['batch_size'], current_val_acc ))
            est_val_history.append( [val_loss / (len(val_data))*config_dict['batch_size'], current_val_acc] )



            #last epoch, save the output probability epoch == (config_dict['n_epoch_estimate'] - 1)
            if current_val_acc > highest_est_val_acc :
                highest_est_val_acc = current_val_acc
                highest_est_val_acc_epoch_n = epoch
                with torch.no_grad():
                    for batch_index, (batch_x, batch_y) in enumerate(estimate_loader):
                        batch_x = batch_x.to(device)
                        out = est_model(batch_x)
                        out = Func.softmax(out, dim=1)
                        out = out.cpu() #Returns a copy of this object in CPU memory. Cuz we do computation in CPU memory(system RAM)

                        #store this batch of probability into the probability matrix
                        if batch_index <= batch_index_num: #cuz the batch_index_number always round up(using int())
                            probability_matrix[ (batch_index * config_dict['batch_size']) : ((batch_index+1) * config_dict['batch_size']) , :] = out
                        else:
                            print("Unexpected behaviours !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            #[index_num*args.batch_size, len(train_data), :] = out  magic code for spare

        #checking output of the probability matrix
        is_zero = False

        for i in range(probability_matrix.shape[0]):
            for j in range(probability_matrix.shape[1]):
                if probability_matrix[i,j] == 0:
                    is_zero = True
        print("Zeros in probability matrix : {}".format(is_zero))
        print(probability_matrix)


        #get the transition matrix
        transition_matrix_no_norm = tools.find_transition_matrix(probability_matrix, config_dict['num_classes'], filter_outlier=True)
        transition_matrix_norm = tools.norm(transition_matrix_no_norm)

        print("Using probability matrix from epoch {} with val acc {}".format(highest_est_val_acc_epoch_n + 1, highest_est_val_acc))

        print("Both transition matrix before transpose")
        print("Transition matrix before norm")
        print(transition_matrix_no_norm)
        print("Transition matrix after norm")
        print(transition_matrix_norm)

        #saving probability matrix
        prob_matrix_save_path = prob_save_dir + '/' + 'probability_matrix.npy'
        np.save(prob_matrix_save_path, probability_matrix)

        #save transition norm matrix
        trans_matrix_save_path = matrix_save_dir + '/' + 'transition_matrix_norm.npy'
        np.save(trans_matrix_save_path, transition_matrix_norm)

        #save transition no norm matrix
        trans_matrix_save_path = matrix_save_dir + '/' + 'transition_matrix_no_norm.npy'
        np.save(trans_matrix_save_path, transition_matrix_no_norm)



        #save all training log
        with open(logging_data_dir+'/est_model_log.csv', 'w') as f:
            writer = csv.writer(f)
            for i in range(len(est_train_history)) :
                l = ['Epoch', i+1]
                lt = ['Training data : ', 'loss', est_train_history[i][0], 'accuracy', est_train_history[i][1]]
                lv = ['Val data : ', 'loss', est_val_history[i][0], 'accuracy', est_val_history[i][1]]

                writer.writerow(l)
                writer.writerow(lt)
                writer.writerow(lv)



        #training the model using the transition matrix
        print("\n\n Finished Estimating transition matrix, Start training the classifier using Transition matrix")


        #setup everything again for actual training
        train_model = Model_2(config_dict['num_classes'])

        #optimizer
        train_optimizer_SGD = optim.SGD(train_model.parameters(), lr=config_dict['lr'], momentum=0.9) #, weight_decay=config_dict['weight_decay_train']
        #scheduler
        train_scheduler = MultiStepLR(train_optimizer_SGD, milestones=train_lr_steps, gamma=0.3)


        #set up loss function for training

        #stock ce
        train_loss_ce = nn.CrossEntropyLoss()

        # #reweight ce
        # train_loss_ce_reweight = reweight_loss()


        #load transition matrix into torch
        transition_matrix_cuda_no_transpose = torch.from_numpy(transition_matrix_no_norm).float().to(device) #for reweight loss

        transition_matrix_cuda = torch.from_numpy(transition_matrix_no_norm).float().to(device)
        transition_matrix_cuda = transition_matrix_cuda.t()  #transpose

        print("Tranpose matrix :")
        print(transition_matrix_cuda)


        #cuda
        if torch.cuda.is_available:
            print("using GPU, move to device : " + device)
            train_model = train_model.to(device)
            train_loss_ce = train_loss_ce.to(device)
            #train_loss_ce_reweight = train_loss_ce_reweight.to(device)
        else:
            print("using cpu")


        for epoch in range(config_dict['n_epoch_training']):

            print('epoch {} with LR {}'.format(epoch + 1, train_scheduler.get_last_lr()))

            train_model.train()

            #logging part
            train_loss = 0.
            train_acc = 0.
            val_loss = 0.
            val_acc = 0.
            eval_loss = 0.
            eval_acc = 0.

            #mini batch training
            for batch_x, batch_y in train_loader:

                #load data into gpu
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                #zeroing the optimizer
                train_optimizer_SGD.zero_grad()

                #compute the normal output
                out = train_model(batch_x)

                #softmax the out
                prob = Func.softmax(out, dim=1)

                #print("before",prob.shape)

                #speculation part
                prob = prob.t() #convert into column format

                #print("after",prob.shape)

                #apply the transition matrix
                out_apply_T = torch.matmul(transition_matrix_cuda, prob)
                out_apply_T = out_apply_T.t()


                # #reweight ce
                # loss = train_loss_ce_reweight(out, transition_matrix_cuda_no_transpose, batch_y)

                #stock ce
                loss = train_loss_ce(out_apply_T, batch_y)


                #update model
                loss.backward()
                train_optimizer_SGD.step()

                #loss logging
                train_loss += loss.item()
                pred = torch.max(out_apply_T, 1)[1]
                train_correct = (pred == batch_y).sum()
                train_acc += train_correct.item()


            #decay the lr
            train_scheduler.step()

            #training output
            epoch_training_loss = train_loss / (len(train_data))*config_dict['batch_size']
            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(epoch_training_loss, train_acc / (len(train_data))))
            real_train_history.append( [epoch_training_loss, train_acc / (len(train_data)) ] )

            #validation of the training process
            with torch.no_grad():

                train_model.eval()
                for batch_x, batch_y in val_loader:

                    #load data into gpu
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    #forward pass
                    out = train_model(batch_x)
                    prob = Func.softmax(out, dim=1)
                    prob = prob.t()
                    out_apply_T = torch.matmul(transition_matrix_cuda, prob)
                    out_apply_T = out_apply_T.t()

                    # #reweight ce
                    # loss = train_loss_ce_reweight(out, transition_matrix_cuda_no_transpose, batch_y)

                    #stock ce
                    loss = train_loss_ce(out_apply_T, batch_y)

                    #datalogging
                    val_loss += loss.item()
                    pred = torch.max(out_apply_T, 1)[1]
                    val_correct = (pred == batch_y).sum()
                    val_acc += val_correct.item()

            #validation output
            print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data))*config_dict['batch_size'], val_acc / (len(val_data))))
            real_val_history.append( [val_loss / (len(val_data))*config_dict['batch_size'], val_acc / (len(val_data))] )


            #testing during each epoch
            with torch.no_grad():

                train_model.eval()
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    out = train_model(batch_x)

                    #stock ce
                    loss = train_loss_ce(out, batch_y)

                    #evaluation
                    eval_loss += loss.item()
                    pred = torch.max(out, 1)[1]
                    eval_correct = (pred == batch_y).sum()
                    eval_acc += eval_correct.item()

                #testing output
                print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data))*config_dict['batch_size'], eval_acc / (len(test_data))))
                real_test_history.append([ eval_loss / (len(test_data))*config_dict['batch_size'] , eval_acc / (len(test_data)) ])


            #save all training log
            with open(logging_data_dir+'/real_model_log.csv', 'w') as f:
                writer = csv.writer(f)
                for i in range(len(real_train_history)) :
                    l = ['Epoch', i+1]
                    lt = ['Training data : ', 'loss', real_train_history[i][0], 'accuracy', real_train_history[i][1]]
                    lv = ['Val data : ', 'loss', real_val_history[i][0], 'accuracy', real_val_history[i][1]]
                    lts = ['test data : ', 'loss', real_test_history[i][0], 'accuracy', real_test_history[i][1]]

                    writer.writerow(l)
                    writer.writerow(lt)
                    writer.writerow(lv)
                    writer.writerow(lts)



            #save config
            with open('model' + '/' + training_target_ds + '/' + 'Version_%s'%(version) + '/' +'/model_config.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow([training_target_ds])
                for k in config_dict.keys():
                    writer.writerow([k, config_dict[k]])