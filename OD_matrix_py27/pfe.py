import torch
import numpy as np
import math

def get_gradient(A, y, x):
    (n, d) = A.size()
    # tmp = y - torch.mm(A, x.view(d, -1)) # !! His code
    tmp = y.view(n, -1) - torch.mm(A, x.view(d, -1)) # !! My code
    return - torch.mm(A.t(), tmp.view(n, -1)) / np.float(n)

def get_loss(A, y, x):
    (n, d) = A.size()
    return torch.sum(torch.pow(y.view(n, -1) - torch.mm(A, x.view(d, -1)), 2))


    # Question: min{q} ||x-Bq||     s.t. q>=0, rs \in Kq, 1<=h1<=N
    # y is the vector of ground truth for link flow
    # A is the matrix that is multiplied to vector q in the formulation
# (q_est, r_norm) = nnls(A, x_o, 300, 8192, 5, adagrad = True, use_GPU = True, D_vec = None, D_vec_weight = 0.01) # !!
def nnls(A, y, num_epoch, batch_size, learning_rate0, adagrad = False,
             use_GPU = False, D_vec = None, D_vec_weight = 0.1, verbose = True):
    learning_rate = learning_rate0
    list_loss = list() # !! My code
    (n, d) = A.shape
    scale = np.sqrt(6) / np.sqrt(np.float(d))
    x = (np.random.rand(d) * 2 - 1.0) * scale
    A_torch = torch.FloatTensor(A) # !! I uncommented this line
    y_torch = torch.FloatTensor(y)
    x_torch = torch.FloatTensor(x)

    if (float(n) / float(batch_size)) < 1:  # My added if line
        print "Number of batch sizes is larger than the number of instances."  # My added if line
        print "Number of samples: ", n
        print "Batch size: ", batch_size
        print "Number of batches: 1"
    else:
        print "Number of batches: ", math.ceil(float(n) / float(batch_size))

    if D_vec is not None:
        D_vec_torch = torch.FloatTensor(D_vec)
    if use_GPU:
        A_torch = A_torch.cuda() # !! I uncommented this line
        y_torch = y_torch.cuda()
        x_torch = x_torch.cuda()
        if D_vec is not None:
            D_vec_torch = D_vec_torch.cuda()

    counter = 0  # !! My code
    counter2 = 0  # !! My code
    lr_loss_check = [float('inf')]  # !! My code

    for epoch in range(num_epoch):
        # print "epoch: ", epoch
        if adagrad:
            sum_g_square = 0
        # if verbose:   # !! I uncommented this line
        #     print "Epoch:", epoch  # !! I uncommented this line
        #     loss = get_loss(A_torch, y_torch, x_torch)   # !! I uncommented this line
        #     print "Current loss is:", loss  # !! I uncommented this line
        # Permute the index of rows
        seq = np.random.permutation(n)

        if (len(seq) / batch_size) == 0: # My added if line
            # print "Number of batch sizes larger than the number of instances." # My added if line
            train_sample_list = np.array_split(seq, 1) # My added if line
        else:
            # print "Number of batches: ", len(seq) / batch_size
            # Divide the permuted indices in the number of batches # His code
            train_sample_list = np.array_split(seq, len(seq) / batch_size) # His code
        lr_loss_check_temp = []  # !! My code
        for sample_ind in train_sample_list:
            # if use_GPU:
            #     sample_ind_torch = torch.cuda.LongTensor(sample_ind)
            # else:
            #     sample_ind_torch = torch.LongTensor(sample_ind)
            # A_sub = torch.index_select(A_torch, 0, sample_ind_torch)
            # The portion of matrix for the batch
            A_sub = torch.FloatTensor(A[sample_ind, :])
            if use_GPU:
                A_sub = A_sub.cuda()
            if use_GPU:
                sample_ind_torch = torch.cuda.LongTensor(sample_ind)
            else:
                sample_ind_torch = torch.LongTensor(sample_ind)
            # The portion of y array for the batch
            y_sub = y_torch[sample_ind_torch]

            # get_gradient output:
            # (n, d) = A.size()
            # - torch.mm(A_sub.t(), (y_torch.view(n, -1) - torch.mm(A_sub, x_torch.view(d, -1))).view(n, -1)) / np.float(n)
            # g = - A_sub.T.dot(y_sub - A_sub.dot(x)) / batch_size # !! My code based on new_nnls function

            # g = get_gradient(A_sub, y_torch, x_torch) # !! His code
            g = get_gradient(A_sub, y_sub, x_torch) # !! My code
            # get_function: (n_, d_) = A.size() !! My code based on nnls and get_function function
            # get_function: - torch.mm(A.t(), (y - torch.mm(A, x.view(d_, -1))).view(n_, -1)) / np.float(n_)
            # new_nnls:
            # g = - A_sub.T.dot(y_sub - A_sub.dot(x)) / batch_size # !! My code based on new_nnls function
            # print "Gradient: ", g
            if D_vec is not None:
                g += D_vec_weight * D_vec_torch.view(d, -1)
            if adagrad:
                # sum_g_square = sum_g_square + torch.pow(g, 2) # !! His code
                sum_g_square = sum_g_square + torch.pow(torch.flatten(g), 2)  # !! My code
                # x_torch = x_torch - learning_rate * torch.flatten(g) / (torch.sqrt(sum_g_square)) # !! His code
                x_torch = x_torch - learning_rate * torch.flatten(g) / (torch.sqrt(sum_g_square)+1e-8)  # !! My code
                # print "torch.flatten(g): ", torch.flatten(g)  # !! My code
                # print "x_torch: ", x_torch  # !! My code
                # print "torch.sqrt(sum_g_square): ", torch.sqrt(sum_g_square)  # !! My code
                # print "lr: ", learning_rate * torch.flatten(g) / torch.sqrt(sum_g_square)  # !! My code
            else:
                x_torch = x_torch - learning_rate * torch.flatten(g) / np.float(epoch + 1)
                # print "lr: ", learning_rate * torch.flatten(g) / np.float(epoch + 1)  # !! My code

            # Change Nan to 0
            x_torch[x_torch != x_torch] = 0.0
            x_torch = torch.clamp(x_torch, min=0.0)
            # loss function: # (For decoding 1)
            # A_loss = A_sub # (For decoding 1)
            # y_loss = y_sub # (For decoding 1)
            # x_loss = x_torch # (For decoding 1)
            # (n_, d_) = A_loss.size() (For decoding 1)
            # loss = torch.sum(torch.pow(y_loss.view(n_, -1) - torch.mm(A_loss, x_loss.view(d_, -1)), 2)) # (For decoding 1)
            # loss = get_loss(A, y, x) # !! His code
            loss = get_loss(A_sub, y_sub, x_torch)  # !! My code
            lr_loss_check_temp.append(loss.cpu().item())
            if math.isnan(loss.cpu().item()):
                print "x_torch: ", x_torch  # !! My code
                print "torch.flatten(g): ", torch.flatten(g)  # !! My code
                print "torch.sqrt(sum_g_square): ", sum_g_square  # !! My code
                print "lr: ", learning_rate * torch.flatten(g) / torch.sqrt(sum_g_square)  # !! My code
            list_loss.append(loss.cpu().item())

        avg_loss_temp = sum(lr_loss_check_temp)/float(n)
        normalized_loss_temp = math.sqrt(avg_loss_temp)/np.linalg.norm(y)
        min_temp = min(lr_loss_check)

        # if avg_loss_temp + 0.001 < min_temp:  # !! My code
        #     counter = 0  # !! My code
        # else:  # !! My code
        #     counter += 1  # !! My code
        #  # !! My code
        # if counter > 100:  # !! My code
        #     counter = 0  # !! My code
        #     # if math.log(learning_rate0 / learning_rate, 10) == 1:  # !! My code
        #     if math.log(learning_rate0 / learning_rate, 10) == 3:  # !! My code
        #         # print "math.log(learning_rate0 / learning_rate, 10) == 1"  # !! My code
        #         break  # !! My code
        #     else:  # !! My code
        #         # print "learning_rate = learning_rate / 10"  # !! My code
        #         learning_rate = learning_rate / 10  # !! My code

        lr_loss_check.append(avg_loss_temp)

        print "\nEpoch number: %i, Avg loss: %.2f, SQRT of avg loss: %.2f" % (epoch, avg_loss_temp, math.sqrt(avg_loss_temp))
        print "L2 loss:  %.2f, L2 norm of true volume:  %.2f, Normalized L2 loss:  %.2f" % (math.sqrt(sum(lr_loss_check_temp)), math.sqrt(sum(np.power(y, 2))), (math.sqrt(sum(lr_loss_check_temp)))/math.sqrt(sum(np.power(y, 2))))

        # print "\n(Epoch number, Counter): ", (epoch, counter)
        # print "(Average loss, Min average loss so far):  (%.2f, %.2f) " % (avg_loss_temp, min(lr_loss_check))
        # print "SQRT>> (Average loss, Min average loss so far):  (%.2f, %.2f) " % (math.sqrt(avg_loss_temp), math.sqrt(min(lr_loss_check)))
        # print "Normalized L2 loss:  %.2f" % (math.sqrt(sum(lr_loss_check_temp))/torch.sqrt(torch.sum(torch.pow(torch.flatten(y_torch), 2))))
        # print "L2 norm of true volume:  %.2f" % (math.sqrt(sum(np.power(y, 2))))
        # print "L2 loss:  %.2f" % (math.sqrt(sum(lr_loss_check_temp)))
        # print "Normalized L2 loss:  %.2f" % ((math.sqrt(sum(lr_loss_check_temp)))/math.sqrt(sum(np.power(y, 2))))

        # loss = get_loss(A_torch, y_torch, x_torch) # !! My code
        # print "Current loss is:", loss

    if use_GPU:
        x_torch = x_torch.cpu()
    # return (x_torch.numpy(), loss) # !! His code
    return (x_torch.numpy(), loss, list_loss) # !! My code


def new_nnls(A, y, maxiter, batch_size, step_size, adagrad = False, verbose = False):
    # L.dot(A)
    (n,d) = A.shape
    # x = np.random.rand(d)
    scale = np.sqrt(6) / np.sqrt(np.float(d))
    x = (np.random.rand(d) * 2 - 1.0) * scale
    for epoch in range(maxiter):
        if adagrad:
                sum_g_square = 1e-6
        seq = np.random.permutation(n)
        train_sample_list = np.array_split(seq, len(seq) / batch_size)
        for sample_ind in train_sample_list:
            A_sub = A[sample_ind, :]
            y_sub = y[sample_ind]
    #         print y_sub.shape
            g = - A_sub.T.dot(y_sub - A_sub.dot(x)) / batch_size
    #         print g
            if adagrad:
                sum_g_square = sum_g_square + np.power(g, 2)
    #             print sum_g_square
                x = x - step_size * g / np.sqrt(sum_g_square)
            else:
                x = x - step_size / np.float(epoch + 1) * g
        x = np.maximum(0, x)
        loss = np.linalg.norm(y - A.dot(x))
        if verbose:
            print epoch, loss
    return (x, loss)


def Ali_nnls(A, y, num_epoch, batch_size, learning_rate0, adagrad = False,
             use_GPU = False, D_vec = None, D_vec_weight = 0.1, verbose = True):
    learning_rate = learning_rate0
    list_loss = list() # !! My code
    (n, d) = A.shape
    scale = np.sqrt(6) / np.sqrt(np.float(d))
    x = (np.random.rand(d) * 2 - 1.0) * scale
    A_torch = torch.FloatTensor(A) # !! I uncommented this line
    y_torch = torch.FloatTensor(y)
    x_torch = torch.FloatTensor(x)

    if (float(n) / float(batch_size)) < 1:  # My added if line
        print "Number of batch sizes larger than the number of instances."  # My added if line
        print "Number of samples: ", n
        print "Batch size: ",
        print "Number of batches: 1"
    else:
        print "Number of batches: ", math.ceil(float(n) / float(batch_size))

    if D_vec is not None:
        D_vec_torch = torch.FloatTensor(D_vec)
    if use_GPU:
        A_torch = A_torch.cuda() # !! I uncommented this line
        y_torch = y_torch.cuda()
        x_torch = x_torch.cuda()
        if D_vec is not None:
            D_vec_torch = D_vec_torch.cuda()

    counter = 0  # !! My code
    counter2 = 0  # !! My code
    lr_loss_check = [float('inf')]  # !! My code

    for epoch in range(num_epoch):
        # print "epoch: ", epoch
        if adagrad:
            sum_g_square = 0
        # if verbose:   # !! I uncommented this line
        #     print "Epoch:", epoch  # !! I uncommented this line
        #     loss = get_loss(A_torch, y_torch, x_torch)   # !! I uncommented this line
        #     print "Current loss is:", loss  # !! I uncommented this line
        # Permute the index of rows
        seq = np.random.permutation(n)

        if (len(seq) / batch_size) == 0: # My added if line
            # print "Number of batch sizes larger than the number of instances." # My added if line
            train_sample_list = np.array_split(seq, 1) # My added if line
        else:
            # print "Number of batches: ", len(seq) / batch_size
            # Divide the permuted indices in the number of batches # His code
            train_sample_list = np.array_split(seq, len(seq) / batch_size) # His code
        lr_loss_check_temp = []  # !! My code
        for sample_ind in train_sample_list:
            # if use_GPU:
            #     sample_ind_torch = torch.cuda.LongTensor(sample_ind)
            # else:
            #     sample_ind_torch = torch.LongTensor(sample_ind)
            # A_sub = torch.index_select(A_torch, 0, sample_ind_torch)
            # The portion of matrix for the batch
            A_sub = torch.FloatTensor(A[sample_ind, :])
            if use_GPU:
                A_sub = A_sub.cuda()
            if use_GPU:
                sample_ind_torch = torch.cuda.LongTensor(sample_ind)
            else:
                sample_ind_torch = torch.LongTensor(sample_ind)
            # The portion of y array for the batch
            y_sub = y_torch[sample_ind_torch]

            # get_gradient output:
            # (n, d) = A.size()
            # - torch.mm(A_sub.t(), (y_torch.view(n, -1) - torch.mm(A_sub, x_torch.view(d, -1))).view(n, -1)) / np.float(n)
            # g = - A_sub.T.dot(y_sub - A_sub.dot(x)) / batch_size # !! My code based on new_nnls function

            # g = get_gradient(A_sub, y_torch, x_torch) # !! His code
            g = get_gradient(A_sub, y_sub, x_torch) # !! My code
            # get_function: (n_, d_) = A.size() !! My code based on nnls and get_function function
            # get_function: - torch.mm(A.t(), (y - torch.mm(A, x.view(d_, -1))).view(n_, -1)) / np.float(n_)
            # new_nnls:
            # g = - A_sub.T.dot(y_sub - A_sub.dot(x)) / batch_size # !! My code based on new_nnls function
            # print "Gradient: ", g
            if D_vec is not None:
                g += D_vec_weight * D_vec_torch.view(d, -1)
            if adagrad:
                # sum_g_square = sum_g_square + torch.pow(g, 2) # !! His code
                sum_g_square = sum_g_square + torch.pow(torch.flatten(g), 2)  # !! My code
                # x_torch = x_torch - learning_rate * torch.flatten(g) / (torch.sqrt(sum_g_square)) # !! His code
                x_torch = x_torch - learning_rate * torch.flatten(g) / (torch.sqrt(sum_g_square)+1e-8)  # !! My code
                # print "torch.flatten(g): ", torch.flatten(g)  # !! My code
                # print "x_torch: ", x_torch  # !! My code
                # print "torch.sqrt(sum_g_square): ", torch.sqrt(sum_g_square)  # !! My code
                # print "lr: ", learning_rate * torch.flatten(g) / torch.sqrt(sum_g_square)  # !! My code

            else:
                x_torch = x_torch - learning_rate * torch.flatten(g) / np.float(epoch + 1)
                # print "lr: ", learning_rate * torch.flatten(g) / np.float(epoch + 1)  # !! My code

            # Change Nan to 0
            x_torch[x_torch != x_torch] = 0.0
            x_torch = torch.clamp(x_torch, min=0.0)
            # loss function: # (For decoding 1)
            # A_loss = A_sub # (For decoding 1)
            # y_loss = y_sub # (For decoding 1)
            # x_loss = x_torch # (For decoding 1)
            # (n_, d_) = A_loss.size() (For decoding 1)
            # loss = torch.sum(torch.pow(y_loss.view(n_, -1) - torch.mm(A_loss, x_loss.view(d_, -1)), 2)) # (For decoding 1)
            # loss = get_loss(A, y, x) # !! His code
            loss = get_loss(A_sub, y_sub, x_torch)  # !! My code
            lr_loss_check_temp.append(loss.cpu().item())
            if math.isnan(loss.cpu().item()):
                print "x_torch: ", x_torch  # !! My code
                print "torch.flatten(g): ", torch.flatten(g)  # !! My code
                print "torch.sqrt(sum_g_square): ", sum_g_square  # !! My code
                print "lr: ", learning_rate * torch.flatten(g) / torch.sqrt(sum_g_square)  # !! My code
            list_loss.append(loss.cpu().item())

        avg_loss_temp = sum(lr_loss_check_temp)/float(n)
        min_temp = min(lr_loss_check)

        # if avg_loss_temp + 0.001 < min_temp:  # !! My code
        #     counter = 0  # !! My code
        # else:  # !! My code
        #     counter += 1  # !! My code
        #  # !! My code
        # if counter > 100:  # !! My code
        #     counter = 0  # !! My code
        #     # if math.log(learning_rate0 / learning_rate, 10) == 1:  # !! My code
        #     if math.log(learning_rate0 / learning_rate, 10) == 3:  # !! My code
        #         # print "math.log(learning_rate0 / learning_rate, 10) == 1"  # !! My code
        #         break  # !! My code
        #     else:  # !! My code
        #         # print "learning_rate = learning_rate / 10"  # !! My code
        #         learning_rate = learning_rate / 10  # !! My code

        lr_loss_check.append(avg_loss_temp)

        print "\n(Epoch number, Counter): ", (epoch, counter)
        print "(Average loss, Min average loss so far):  (%.2f, %.2f) " % (avg_loss_temp, min(lr_loss_check))
        print "SQRT>> (Average loss, Min average loss so far):  (%.2f, %.2f) " % (math.sqrt(avg_loss_temp), math.sqrt(min(lr_loss_check)))

        # loss = get_loss(A_torch, y_torch, x_torch) # !! My code
        # print "Current loss is:", loss

    if use_GPU:
        x_torch = x_torch.cpu()
    # return (x_torch.numpy(), loss) # !! His code
    return (x_torch.numpy(), loss, list_loss) # !! My code


# My function
def Ali_nn(A, y, num_epoch, batch_size, learning_rate0, opt_,
             use_GPU=False, D_vec=None, D_vec_weight=0.1, verbose=True):
    opt_dict_ = {'adagrad': 1, 'adam': 2, 'SGD': 3}  # !! My code
    n = y.shape[0]
    d = y.shape[1]
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = n, d, 2*d, A[0, 0].shape[1]

    # Create random Tensors to hold inputs and outputs
    x_torch = torch.randn(N, 1)
    A_torch = torch.FloatTensor(A)
    y_torch = torch.FloatTensor(y)
    if use_GPU:
        A_torch = A_torch.cuda()
        y_torch = y_torch.cuda()
        x_torch = x_torch.cuda()

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algorithms. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(500):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()













    list_loss = list() # !! My code
    (n, d) = A.shape
    scale = np.sqrt(6) / np.sqrt(np.float(d))
    x = (np.random.rand(d) * 2 - 1.0) * scale
    A_torch = torch.FloatTensor(A) # !! I uncommented this line
    y_torch = torch.FloatTensor(y)
    x_torch = torch.FloatTensor(x)

    if (float(n) / float(batch_size)) < 1:  # My added if line
        print "Number of batch sizes larger than the number of instances."  # My added if line
        print "Number of samples: ", n
        print "Batch size: ",
        print "Number of batches: 1"
    else:
        print "Number of batches: ", math.ceil(float(n) / float(batch_size))

    if D_vec is not None:
        D_vec_torch = torch.FloatTensor(D_vec)
    if use_GPU:
        A_torch = A_torch.cuda() # !! I uncommented this line
        y_torch = y_torch.cuda()
        x_torch = x_torch.cuda()
        if D_vec is not None:
            D_vec_torch = D_vec_torch.cuda()

    counter = 0  # !! My code
    counter2 = 0  # !! My code
    lr_loss_check = [float('inf')]  # !! My code

    for epoch in range(num_epoch):
        # print "epoch: ", epoch
        if opt_ == 1:
            sum_g_square = 0
        # if verbose:   # !! I uncommented this line
        #     print "Epoch:", epoch  # !! I uncommented this line
        #     loss = get_loss(A_torch, y_torch, x_torch)   # !! I uncommented this line
        #     print "Current loss is:", loss  # !! I uncommented this line
        # Permute the index of rows
        seq = np.random.permutation(n)

        if (len(seq) / batch_size) == 0: # My added if line
            # print "Number of batch sizes larger than the number of instances." # My added if line
            train_sample_list = np.array_split(seq, 1) # My added if line
        else:
            # print "Number of batches: ", len(seq) / batch_size
            # Divide the permuted indices in the number of batches # His code
            train_sample_list = np.array_split(seq, len(seq) / batch_size) # His code
        lr_loss_check_temp = []  # !! My code
        for sample_ind in train_sample_list:
            # if use_GPU:
            #     sample_ind_torch = torch.cuda.LongTensor(sample_ind)
            # else:
            #     sample_ind_torch = torch.LongTensor(sample_ind)
            # A_sub = torch.index_select(A_torch, 0, sample_ind_torch)
            # The portion of matrix for the batch
            A_sub = torch.FloatTensor(A[sample_ind, :])
            if use_GPU:
                A_sub = A_sub.cuda()
            if use_GPU:
                sample_ind_torch = torch.cuda.LongTensor(sample_ind)
            else:
                sample_ind_torch = torch.LongTensor(sample_ind)
            # The portion of y array for the batch
            y_sub = y_torch[sample_ind_torch]

            # get_gradient output:
            # (n, d) = A.size()
            # - torch.mm(A_sub.t(), (y_torch.view(n, -1) - torch.mm(A_sub, x_torch.view(d, -1))).view(n, -1)) / np.float(n)
            # g = - A_sub.T.dot(y_sub - A_sub.dot(x)) / batch_size # !! My code based on new_nnls function

            # g = get_gradient(A_sub, y_torch, x_torch) # !! His code
            g = get_gradient(A_sub, y_sub, x_torch) # !! My code
            # get_function: (n_, d_) = A.size() !! My code based on nnls and get_function function
            # get_function: - torch.mm(A.t(), (y - torch.mm(A, x.view(d_, -1))).view(n_, -1)) / np.float(n_)
            # new_nnls:
            # g = - A_sub.T.dot(y_sub - A_sub.dot(x)) / batch_size # !! My code based on new_nnls function
            # print "Gradient: ", g
            if D_vec is not None:
                g += D_vec_weight * D_vec_torch.view(d, -1)
            if opt_ == 1:
                # sum_g_square = sum_g_square + torch.pow(g, 2) # !! His code
                sum_g_square = sum_g_square + torch.pow(torch.flatten(g), 2)  # !! My code
                # x_torch = x_torch - learning_rate * torch.flatten(g) / (torch.sqrt(sum_g_square)) # !! His code
                x_torch = x_torch - learning_rate * torch.flatten(g) / (torch.sqrt(sum_g_square)+1e-8)  # !! My code
                # print "torch.flatten(g): ", torch.flatten(g)  # !! My code
                # print "x_torch: ", x_torch  # !! My code
                # print "torch.sqrt(sum_g_square): ", torch.sqrt(sum_g_square)  # !! My code
                # print "lr: ", learning_rate * torch.flatten(g) / torch.sqrt(sum_g_square)  # !! My code

            else:
                x_torch = x_torch - learning_rate * torch.flatten(g) / np.float(epoch + 1)
                # print "lr: ", learning_rate * torch.flatten(g) / np.float(epoch + 1)  # !! My code

            # Change Nan to 0
            x_torch[x_torch != x_torch] = 0.0
            x_torch = torch.clamp(x_torch, min=0.0)
            # loss function: # (For decoding 1)
            # A_loss = A_sub # (For decoding 1)
            # y_loss = y_sub # (For decoding 1)
            # x_loss = x_torch # (For decoding 1)
            # (n_, d_) = A_loss.size() (For decoding 1)
            # loss = torch.sum(torch.pow(y_loss.view(n_, -1) - torch.mm(A_loss, x_loss.view(d_, -1)), 2)) # (For decoding 1)
            # loss = get_loss(A, y, x) # !! His code
            loss = get_loss(A_sub, y_sub, x_torch)  # !! My code
            lr_loss_check_temp.append(loss.cpu().item())
            if math.isnan(loss.cpu().item()):
                print "x_torch: ", x_torch  # !! My code
                print "torch.flatten(g): ", torch.flatten(g)  # !! My code
                print "torch.sqrt(sum_g_square): ", sum_g_square  # !! My code
                print "lr: ", learning_rate * torch.flatten(g) / torch.sqrt(sum_g_square)  # !! My code
            list_loss.append(loss.cpu().item())

        avg_loss_temp = sum(lr_loss_check_temp)/float(n)
        min_temp = min(lr_loss_check)

        # if avg_loss_temp + 0.001 < min_temp:  # !! My code
        #     counter = 0  # !! My code
        # else:  # !! My code
        #     counter += 1  # !! My code
        #  # !! My code
        # if counter > 100:  # !! My code
        #     counter = 0  # !! My code
        #     # if math.log(learning_rate0 / learning_rate, 10) == 1:  # !! My code
        #     if math.log(learning_rate0 / learning_rate, 10) == 3:  # !! My code
        #         # print "math.log(learning_rate0 / learning_rate, 10) == 1"  # !! My code
        #         break  # !! My code
        #     else:  # !! My code
        #         # print "learning_rate = learning_rate / 10"  # !! My code
        #         learning_rate = learning_rate / 10  # !! My code

        lr_loss_check.append(avg_loss_temp)

        print "\n(Epoch number, Counter): ", (epoch, counter)
        print "(Average loss, Min average loss so far):  (%.2f, %.2f) " % (avg_loss_temp, min(lr_loss_check))
        print "SQRT>> (Average loss, Min average loss so far):  (%.2f, %.2f) " % (math.sqrt(avg_loss_temp), math.sqrt(min(lr_loss_check)))

        # loss = get_loss(A_torch, y_torch, x_torch) # !! My code
        # print "Current loss is:", loss

    if use_GPU:
        x_torch = x_torch.cpu()
    # return (x_torch.numpy(), loss) # !! His code
    return (x_torch.numpy(), loss, list_loss) # !! My code
