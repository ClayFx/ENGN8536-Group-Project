import torch
import numpy as np
import os
from dataloader.MPII import VideoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import time
import torch.nn.functional as F
from src import model
from src import util
from src.body import Body
import cv2
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion_mse = torch.nn.MSELoss()
criterion_L1 = torch.nn.L1Loss()
criterion_ce = torch.nn.CrossEntropyLoss()

def write_log(net, epoch, test_dataloader, criterion, train_accuracy, train_loss, save_dir, log_mode):
    with torch.no_grad():
        test_true_positive = 0
        total = 0
        for batch_idx, (test_data, test_labels) in enumerate(test_dataloader):
            test_res_logit = net(test_data)
            test_loss = criterion(test_res_logit, test_labels.long())
            ans = np.argmax(test_res_logit.detach().numpy(), axis=1)
            for i in range(len(test_labels)):
                if ans[i] == test_labels[i]:
                    test_true_positive += 1
                total += 1
    test_accuracy = test_true_positive * 100 / total

    with open(save_dir, mode=log_mode) as file:
        file.write(f"Epoch {epoch} with Hidden Neuron {net.n_hidden_layers} : Training Loss is {train_loss} | Training Accuracy is {train_accuracy}%\n")
        file.write(f"                Test Loss is {test_loss} | Test Accuracy is {test_accuracy}%\n")
        file.write("\n")

def extract_outputs(oriImg, Mconv7_stage6_L1, Mconv7_stage6_L2):
    scale_search = [0.5]
    boxsize = 368
    stride = 8
    padValue = 128
    multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
    scale = multiplier[0]

    oriImg = oriImg.cpu().numpy()
    oriImg = oriImg.squeeze(0)
    # print(oriImg.shape)
    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)

    heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))  # output 1 is heatmaps
    heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

    # paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1, 2, 0))  # output 0 is PAFs
    paf = np.transpose(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))  # output 0 is PAFs
    paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

    return heatmap, paf

def train(model, optimizer, dataloaders, epochs=20):
    trainloader, testloader = dataloaders

    best_testing_accuracy = 0.0
    # training
    with tqdm(range(1, epochs + 1), ncols=100, ascii=True) as tq:
        for epoch in tq:
            model.train()

            # total_count = torch.tensor([0.0])
            # correct_count = torch.tensor([0.0])
            # batch_time = time.time(); iter_time = time.time()
            for i, data in enumerate(trainloader):
                # Only works for batch_number == 1.
                imgs, _, label = data
                imgs, label = imgs.to(device), label.to(device)
                # if epoch == 5 and i == 0:
                #     cls_scores = model(imgs, with_dyn=args.with_dyn, visualize=True)
                #     print(labels)
                # else:
                #     cls_scores = model(imgs, with_dyn=args.with_dyn)
                
#                 Mconv7_stage6_L1, Mconv7_stage6_L2, heatmap = model(imgs)
#                 lastest_stage6_L1, lastest_stage6_L2 = Mconv7_stage6_L1[:,-1,:,:,:], Mconv7_stage6_L2[:,-1,:,:,:]
#                 print(lastest_stage6_L1.size(), lastest_stage6_L2.size(), heatmap.size())
#                 print(label.size()) # torch.Size([1, 256, 256])

#                 loss = criterion_ce(heatmap.view(-1, 19, 256*256).float(), label.view(-1, 256*256).long())
                
                next_paf, next_heatmap, pre_paf, pre_heatmap = model(imgs)
                # loss_ce = criterion_ce(upsampled_heatmap.view(-1, 19, 256*256).float(), label.view(-1, 256*256).long())

                loss_mse_paf = criterion_L1(next_paf[:, :-1, :, :, :], pre_paf[:, 1:, :, :, :])
                loss_mse_hm = criterion_L1(next_heatmap[:, :-1, :, :, :], pre_heatmap[:, 1:, :, :, :])
                
                loss = loss_mse_paf + loss_mse_hm
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tq.set_description(
                    'Train Epoch: {} [{}/{} ]\t Loss: {:.6f}'.format(epoch, i * len(imgs),
                                                                     len(trainloader.dataset), loss.item()))
                # print(len(list(filter(lambda p: p.requires_grad, model.parameters()))))
                if i % 100 == 0 and i != 0:
                    print('')
                    print('epoch:{}, iter:{}, time:{:.2f}, loss:{:.5f}'.format(epoch, i,
                        time.time()-iter_time, loss.item()))
                    iter_time = time.time()
                
                # total_count += labels.size(0)
                # with torch.no_grad():
                #     predict = torch.argmax(cls_scores, dim=1)
                #     correct_count += (predict.cpu() == labels.cpu()).sum()
            
            model_parameter_checkpoint_path = f'./ckpt/{epoch}_parameter_checkpoint.pth'
            opt_parameter_checkpoint_path = f'./ckpt/{epoch}_opt_parameter_checkpoint.pth'
            # 1) Saving all learnable parameters of the model and optimizer
            torch.save(model.state_dict(), model_parameter_checkpoint_path)
            torch.save(optimizer.state_dict(), opt_parameter_checkpoint_path)
            
#             training_accuracy = (correct_count / total_count).item()

#             batch_time = time.time() - batch_time
#             print(' ')
#             print('[epoch {} | time:{:.2f} | loss:{:.5f}]'.format(epoch, batch_time, loss.item()))
#             print('-------------------------------------------------')

#             if epoch % 1 == 0:
#                 testing_accuracy = evaluate(model, testloader)
#                 print('testing accuracy: {:.3f}'.format(testing_accuracy))
#                 print('training accuracy: {:.3f}'.format(training_accuracy))
#                 # Record the accuracy
#                 with open(f"{epoch}_logging.txt", mode="a+") as file:
#                     file.write('Training: epoch %d | Train Loss: %.6f | Train Accuracy: %.3f%% | Test Accuracy: %.3f%% \n' %
#                            (epoch, loss.item(), 100. * training_accuracy, 100. * testing_accuracy))

#                 if testing_accuracy > best_testing_accuracy:
#                     model_parameter_checkpoint_path = f'./ckpt/{epoch}_parameter_checkpoint.pth'
#                     opt_parameter_checkpoint_path = f'./ckpt/{epoch}_opt_parameter_checkpoint.pth'
#                     # 1) Saving all learnable parameters of the model and optimizer
#                     torch.save(model.state_dict(), model_parameter_checkpoint_path)
#                     torch.save(optimizer.state_dict(), opt_parameter_checkpoint_path)

#                     best_testing_accuracy = testing_accuracy
#                     ### -----------------------------------------------------------------
#                     print('new best model saved at epoch: {}'.format(epoch))
#     print('-------------------------------------------------')
#     print('best testing accuracy achieved: {:.3f}'.format(best_testing_accuracy))


def evaluate(model, testloader):
    model.eval()
    total_count = torch.tensor([0.0]); correct_count = torch.tensor([0.0])
    for i, data in enumerate(testloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)

        total_count += labels.size(0)

        with torch.no_grad():
            heatmap, paf = model(imgs)

            # predict = torch.argmax(cls_scores, dim=1)
            # correct_count += (predict.cpu() == labels.cpu()).sum()
    testing_accuracy = correct_count / total_count
    return testing_accuracy.item()


def resume(model, optimizer):
    checkpoint_path = './ckpt/{}_checkpoint.pth'
    assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)
    ### load the model and the optimizer --------------------------------
    model_parameter_checkpoint_path = './ckpt/{}_parameter_checkpoint.pth'
    opt_parameter_checkpoint_path = './ckpt/{}_opt_parameter_checkpoint.pth'
    # 1) Loading all learnable parameters of the model and optimizer
    model.load_state_dict(torch.load(model_parameter_checkpoint_path))
    optimizer.load_state_dict(torch.load(opt_parameter_checkpoint_path))

    print('Resume completed for the model\n')
    return model, optimizer

def freeze_weights(sequential):
    for layer in sequential:
        for param in layer.parameters():
            param.requires_grad = False

if __name__ == '__main__':
    # # Featurized Cloud data path
    # PATH = "/home/featurize/data/data/image"
    # LABEL_PATH = "/home/featurize/data/data/label"

    # Local Data path
    PATH = "./data/image"
    LABEL_PATH = "./data/label"

    batch_size = 1
    epochs = 5
    base_lr = 5e-5
    lr_cos = lambda n: 0.5 * (1 + np.cos(n / epochs * np.pi)) * base_lr

    dataset = torch_data = VideoDataset(PATH, LABEL_PATH)
    trainset, testset = torch.utils.data.random_split(dataset, [500, 34])

    trainloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=False)
    testloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=False)

    dataloaders = (trainloader, testloader)

    model = model.bodypose_model().to(device)

    model_path = 'model/body_pose_model.pth'

    # pretrained_dict = torch.load(model_path)
    pretrained_dict = util.transfer(model, torch.load(model_path))
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    need2freeze = [model.model0, model.model1_1, model.model2_1, model.model3_1, model.model4_1,
                   model.model5_1, model.model6_1, model.model1_2, model.model2_2, model.model3_2,
                   model.model4_2, model.model5_2, model.model6_2]
    need2train = [model.convLSTM_1, model.convLSTM_2]

    params = [{'params': freeze_para.parameters(), 'lr': float(0)} for freeze_para in need2freeze]
    for para in need2train:
        params.append({'params': para.parameters(), 'lr': base_lr})
    # optimizer
    optimizer = torch.optim.Adam(params, lr=base_lr, betas=(0.9, 0.999))

    # # optimizer
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, betas=(0.9,0.999))

    train(model, optimizer, dataloaders, epochs=epochs)
    print('training finished')


    # # resume the trained model
    # if is_resume:
    #     model, optimizer = resume(model, optimizer)
    #
    # if is_test == 1: # test mode, resume the trained model and test
    #     testing_accuracy = evaluate(model, testloader)
    #     print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))
    # else: # train mode, train the network from scratch
    #     train(model, optimizer, dataloaders)
    #     print('training finished')
