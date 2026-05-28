# coding=utf-8
import numpy as np
from scipy import linalg
from scipy import spatial


def load_dataset(file_name):
    dataset = []
    f = open(file_name)
    for line in f.readlines():
        curline = line.strip().split('\t')
        fltline = map(float, curline)
        dataset.append(fltline)
    return np.mat(dataset)


def hyperconvert2d(data3d):
    """
        A2: Hyperspectral cube (H×W×B) converted to B×N matrix — see dic_constr.py call site.
        Hyperconvert2d Convets an HSI cube to a 2D matrix
        Converts a 3D HSI cube to a 2D matrix of points
        Usage
            data3d = hyperconvert2d(data2d)
        Inputs
            inputdata - 3D HSI cube
        Outputs
            outputdata - 2D data matrix
        
        Methodology Reference: Section 3.2.1 - Reshaping
        The hyperspectral cube H×W×B is reshaped into a matrix B×N,
        where N=H×W is the total number of pixels reshaped into column vectors.
    """
    rows, cols, channels = data3d.shape
    # A2: Reshape cube to B×N (bands × pixels)
    data2d = data3d.reshape(rows * cols, channels, order='F')
    return data2d.transpose()


def hyperconvert3d(data2d, rows, cols, channels):
    """
    HyperConvert2D Converts an 2D matrix to a 3D data cube
    Usage
        data3d=hyeprconvert3d(data2d)
    Inputs
        data2d - 2d data matrix
    Outputs
        data3d - 3d data cube
    """
    channels, pixnum = data2d.shape
    data3d = data2d.transpose().reshape(rows, cols, channels, order='F')
    return data3d


def hypercorr(data2d):
    """
    compute the sample autocorrelation matrix of a 2D matrix
    Usage
        R = hyperCorr(data2d)
    Inputs
        data2d - 2D matrix
    Outputs
        R - Sample autocorreltation matrix
    """
    rows, cols = data2d.shape
    return np.dot(data2d.transpose(), data2d) / cols


def hypercov(data2d):
    """
    hypercorr compute the sample covariance matrix of a 2D matrix
    Usage
        C = hypercorr(data2d)
    Inputs
        data2d - 2D matrix
    Outputs
        C - Sample covariance matrix
    """
    rows, cols = data2d.shape
    mu = np.mean(data2d, 1)
    for i in range(cols):
        data2d[:, i] = data2d[:, i] - mu

    return np.dot(data2d.transpose(), data2d) / (cols - 1)


def hypernorm(data2d, flag):
    """
    hpernormalize Normalize data to be in range [0,1]
    Usage
        normdata = hypernorm(data2d)
    Inputs
        data2d - 2D data matrix
    Outputs
        normdata - Normalize 2D data matrix
    
    Methodology Reference: Section 3.2.1 - Normalization
    Each spectral vector is L2-normalized to compensate for illumination variations.
    When flag == "L2_norm", each pixel x_i is normalized: x_i / ||x_i||_2
    """
    normdata = np.zeros(data2d.shape)
    if flag == "minmax":
        minval = np.min(data2d)
        maxval = np.max(data2d)
        normdata = data2d - minval
        if maxval == minval:
            normdata = np.zeros(data2d.shape)
        else:
            normdata /= maxval - minval
    elif flag == "L2_norm":
        # Section 3.2.1: L2-normalization of spectral vectors
        for i in range(data2d.shape[1]):
            col_norm = linalg.norm(data2d[:, i])
            normdata[:, i] = data2d[:, i] / col_norm

    return normdata


def buildtraintestsamples(data2d, groundtruth, classnum, trainpercent):
    """
    Usage
        build train data and test data randomly
    Inputs
        data2d - 2D data matrix after normalize
        groundtruth - 2D data matrix of labels
        classnum - the number of the class
        trainpercent - the percent of training samples
    Outputs
        train data
        test data
        train label
        test label
    """
    rows, cols = groundtruth.shape
    label = groundtruth.reshape(rows * cols)
    trainSampletuple = []
    testSampletuple = []
    trainLabeltuple = []
    testLabeltuple = []
    for i in range(classnum):
        tem = np.where(label == i + 1)
        temIndex = tem[0]
        sampleNum = temIndex.size
        trainNum = int(round(sampleNum * trainpercent))
        testNum = sampleNum - trainNum
        print ("the number of train subspace and test space {0}".format((trainNum, testNum)))
        np.random.shuffle(temIndex)
        # print "the size of temIndex{0}".format(temIndex.shape)
        # print "the number of select trainspace{0}".format(data2d[:,temIndex[0:trainNum]].shape)
        # print "the number of seeect testspace{0}".format(data2d[:, temIndex[trainNum:testNum+trainNum]].shape)
        trainSampletuple.append(data2d[:, temIndex[0:trainNum]])
        testSampletuple.append(data2d[:, temIndex[trainNum:testNum + trainNum]])
        trainLabeltuple.append(label[temIndex[0:trainNum]])
        testLabeltuple.append(label[temIndex[trainNum:testNum + trainNum]])

    traindata = np.column_stack(trainSampletuple)
    testdata = np.column_stack(testSampletuple)
    trainlabel = np.hstack(trainLabeltuple)
    testlabel = np.hstack(testLabeltuple)
    return traindata, testdata, trainlabel, testlabel


def convetimage(data2d):
    """
    Usage imshow the input matrix
    input
        data2d - 2d data matrix
    output
        2D data after converting
    """
    maxO = data2d.max()
    minO = data2d.min()
    odata = (256/(maxO-minO))*(data2d - minO)
    return odata


def hyperwincreat(data3d, winsize):
    """
    A4: Spatial window creation — build w×w neighborhood matrix W_i ∈ R^(B×w²) per pixel.
    hyperwincreat group the data as locally window block
    Usage
        winMatrix = hyperwincreat(data3d, winsize)
    Input
        data3d - 3D data matrix
        winsize - the size of current window block
    Output
        winMatrix - each element of the tuple is a block matrix
    
    Methodology Reference: Section 3.3 - Spatial Window Extraction
    To incorporate spatial context, overlapping windows of size w×w are extracted 
    around each pixel. Each window forms W_i ∈ R^(B×w²) as in equation (3).
    """
    rows, cols, bands = data3d.shape
    outputuple = []
    if winsize == 1:
        for i in range(rows):
            for j in range(cols):
                outputuple.append(data3d[i, j, :])

    elif winsize / 2 == 0:
        print ("The size of the window must be odd !")
        exit()
    elif winsize >= cols or winsize >= rows:
        print ("The size of the window is too large!")
        exit()

    else:
        r = int((winsize - 1) / 2)
        extern_data = np.zeros((rows + 2 * r, cols + 2 * r, bands))
        for k in range(r):
            for i in range(cols):
                extern_data[k, i + r, :] = data3d[r - k - 1, i, :]
                extern_data[k + rows + r, i + r, :] = data3d[rows - 1 - k, i, :]

            for i in range(rows):
                extern_data[i + r, k, :] = data3d[i, r - k - 1, :]
                extern_data[i + r, k + cols + r, :] = data3d[i, cols - 1 - k, :]

        for i in range(rows):
            for j in range(cols):
                extern_data[i + r, j + r, :] = data3d[i, j, :]

        tmp = np.zeros((bands, winsize * winsize))
        win_matrix = np.zeros((bands, winsize * winsize, rows*cols))
        for i in range(rows):
            for j in range(cols):
                for m in range(winsize):
                    for n in range(winsize):
                        tmp[:, m * winsize + n] = extern_data[i + m, j + n, :]
                win_matrix[:, :, i*cols+j] = tmp
                # outputuple.append(tmp)

        # n = len(outputuple)
        # win_matrix = np.zeros((bands, winsize * winsize, n))
        # for i in range(n):
        #     win_matrix[:, :, i] = outputuple[i]

        return win_matrix


def hyper_IPD(data, center):
    """
       calculate image patch distance
       Usage：
           distance = hyperIPD(data, center, win_size)
       Input:
           data - 2D win_data
           center - target win_data
       output:
           distance - the distance between two block
    """
    rows, cols = data.shape
    c_rows, c_cols = center.shape
    dist_matrix = spatial.distance.cdist(center.transpose(), data.transpose())
    n = int(cols/c_cols)
    dist_matrix_3d = dist_matrix.reshape(c_cols, c_cols, n, order='F')
    tmp1 = np.min(dist_matrix_3d, axis=1)
    tmp2 = np.min(dist_matrix_3d, axis=0)
    dist_com = np.column_stack((tmp1, tmp2)).reshape(c_cols, n, 2, order='F')
    distance = np.sum(np.max(dist_com, axis=2), axis=0)
    return distance


def rand_cent(data_matrix, k):
    rows, cols, n = data_matrix.shape
    # centrdist = np.zeros((rows, cols, n))
    centroids = np.zeros((rows, cols, k))
    # for i in range(n):
    #     bottom = data_matrix[:, :, i].min()
    #     gap = (data_matrix[:, :, i] - bottom).max()
    #     centrdist[:, :, i] = bottom + gap * np.random.rand(rows, cols)

    index = np.arange(n-k*10)
    np.random.shuffle(index)
    for i in range(k):
        centroids[:, :, i] = data_matrix[:, :, index[i]:index[i]+20*10].mean(2)
    #     centroids[:, :, i] = centrdist[:, :, index[i]]

    return centroids


def Kmeans_win(data_matrix, k):
    """
    A5: K-means clustering over window vectors — called from dic_constr.py; prints iteration count and error E.
    Methodology Reference: Section 3.3 - Clustering
    K-means clustering is performed over all window vectors to partition them 
    into C clusters as in equation (4): {W_1, W_2, …, W_C}.
    Clustering ensures that dictionary atoms represent diverse background 
    materials and spatial structures.
    """
    rows, cols, n = data_matrix.shape
    data_matrix_2d = data_matrix.reshape(rows, cols*n, order='F')
    cluster_assment = np.zeros((n, 2))
    centroids = rand_cent(data_matrix, k)
    distance = np.zeros((k, n))
    label_old = np.zeros(n)
    E = np.inf
    E_old = np.inf
    count = 1
    cluster_changed = True
    # A5: K-means iterate until cluster labels stabilize (printed count and error E)
    while cluster_changed:
        cluster_changed = False
        print (count)
        count += 1
        for j in range(k):
            distance[j, :] = hyper_IPD(data_matrix_2d, centroids[:, :, j])

        label_tmp = distance.argmin(0)
        value_tmp = distance.min(0)

        if (label_tmp == label_old).all():
            break
        else:
            label_old = label_tmp
            cluster_changed = True

        for cent in range(k):
            tmp = np.where(label_tmp == cent)
            cluster_index = tmp[0]
            if cluster_index.size == 0:
                centroids[:, :, cent] = centroids[:, :, cent]
            else:
                centroids[:, :, cent] = data_matrix[:, :, cluster_index].mean(2)

        E = np.linalg.norm(value_tmp)
        print (E)
        if abs(E_old-E) < 0.1:
            break
        if count > 50:
            break


    centroids_2d = centroids.reshape(rows, cols*k)
    for i in range(k):
        p_num = np.where(label_tmp == i)
        per_num = p_num[0]
        if per_num.size < 2:
            c_dist = hyper_IPD(centroids_2d, centroids[:, :, i])
            new_class = c_dist.argmax()
            label_tmp[per_num] = new_class
        else:
            continue

    return label_tmp


def somp(dict, X, K):
    """
        A7: SOMP solver used inside A8 (JSR) — selects K joint sparse atoms for each window.
        calculate the joint sparse
        representation based on somp
        Usage:
            alpha, alpha_index, residual, chosen_atom \\
            = somp(dict, data, k)
        Input:
            dict : the complete dictionary
            data : the data need to be representation
            k    : the level of sparse
        Output:
            alpha : the sparse coefficients of the data
            alpha_index : the index coresponding to the column of the data
            resdiual : the final residual of the data
            chosen_atom : the finally chosen K atoms in the dictionary
        
        Methodology Reference: Section 3.4 - Joint Sparse Representation (JSR)
        For each cluster W_c, Simultaneous Orthogonal Matching Pursuit (SOMP) 
        is used to obtain sparse representations: W_c = D_c A_c + R_c (equation 5).
        The SOMP procedure selects atoms that jointly minimize the residual 
        across all pixels in the window: arg min_(A_c) ||W_c - D_c A_c||_F (equation 6).
        """
    #print "somp start ......."
    x_rows, x_cols = X.shape
    index = np.zeros(K)
    resdiual = X
    chosen_atom = np.zeros((x_rows, K))
    chosen_tuple = []
    err = -np.inf
    alpha = []
    for i in range(K):
        proj = np.dot(resdiual.transpose(), dict)
        tmp = np.linalg.norm(proj, axis=0)
        max_index = np.argmax(tmp)
        chosen_atom[:,i] = dict[:, max_index]
        index[i] = max_index
        tmp2 = np.dot(chosen_atom[:,0:i+1].transpose(), chosen_atom[:,0:i+1])
        alpha = np.dot(np.dot(np.linalg.pinv(tmp2), chosen_atom[:,0:i+1].transpose()), X)
        resdiual = X - np.dot(chosen_atom[:,0:i+1], alpha)
        # err = np.linalg.norm(resdiual)
        # if err < 0.1:
        #     break

    # print "Success! reconstruct error is:{0}".format(err)
    return alpha, index, chosen_atom, resdiual

















