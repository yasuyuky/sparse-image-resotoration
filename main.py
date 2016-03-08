import numpy as np
import math,random
from sklearn import linear_model
import pprint
from PIL import Image

debug_params = {
        'A' : ('A', False),
        'projections' : ('projections', False),
        'n_iter' : ('iteration num', True),
        'poe' : ('projection of estimated image', False),
        'orig_img' : ('original image', False),
        'restored_img' : ('restored image', False),
        'diff' : ('diff image', False),
        'diff_v' : ('sum of diff', True),
        }

def debug_message(id, item):
    if id in debug_params:
        title, flag = debug_params[id]
    else:
        return
    if flag:
        print title+":"
        pprint.pprint(item)

def generate_projection_matrix(m, n):
    A = np.ndarray(shape=(m+n+(m+n-1)+(m+n-1),m*n), dtype=np.float64, order='F')
    # horizontal
    for i in range(m):
        for p in range(m*n):
            if n*i <= p and p < n*(i+1):
                A[i,p] = 1.
            else:
                A[i,p] = 0.
    # vertical
    for i in range(n):
        for p in range(m*n):
            if p % n == i:
                A[m+i,p] = 1.
            else:
                A[m+i,p] = 0.
    # diagonal
    c = m+n
    for k in range(m+n-1):
        cc = c+k
        for p in range(m*n):
            i,j = p/n, p%n
            if i+j == k:
                A[cc, p] = 1.
            else:
                A[cc, p] = 0.
    # anti-diagonal
    c = m+n+m+n-1
    for k in range(m+n-1):
        cc = c+k
        for p in range(m*n):
            i,j = p/n, p%n
            if i-j+n-1 == k:
                A[cc, p] = 1.
            else:
                A[cc, p] = 0.
    return A

def generate_random_matrix(m, n, n_samples):
    return np.random.randint(2, size=(n_samples, m*n)).astype(dtype=np.float64, order='F')

def restoration(img):
    m,n = len(img), len(img[0])
    #A = generate_projection_matrix(m, n)
    A = generate_random_matrix(m, n, 10000)
    debug_message('A', A)

    img_mat = np.array(img, dtype=np.float64, order='F')
    img_vec = img_mat.flatten()
    #projections = np.dot(A, img_vec).reshape((m+n,1), order='F')
    projections = np.dot(A, img_vec).ravel()
    debug_message('projections', projections)
    penalty_order = -3
    #penalty = (m+n) * math.pow(10, penalty_order)
    #w = spams.lasso(X=projections, D=A, return_reg_path=False, lambda1=penalty)
    estimator = linear_model.Lasso(
            alpha=math.pow(10, penalty_order)
            ,positive=True
            #,tol=math.pow(10,-5)
            ,max_iter=math.pow(10,6)
            )
    estimator.fit(A,projections)
    debug_message('n_iter', estimator.n_iter_)
    debug_message('poe', estimator.coef_)
    r = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(estimator.coef_[i*n+j])
        r.append(row)

    return r

def make_random_image(m, n, sample_num):
    img = []
    for i in range(m):
        img.append([0.]*n)
    for j in range(m):
        for i in range(sample_num):
            x = random.randint(0,n-1)
            img[j][x] = 1.
    return img

def make_diagonal_image(m, n):
    img = [[0. for i in range(n)] for j in range(m)]
    k = min(m, n)
    for i in range(k):
        img[i][i] = 1.
    return img

def make_diff(orig, target):
    m = len(orig)
    n = len(orig[0])
    img = [[0. for j in range(n)] for i in range(m)]
    for i in range(m):
        for j in range(n):
            img[i][j] = target[i][j] - orig[i][j]
    return img

def abs_sum(l):
    return reduce(lambda acc,x: acc + abs(x), l, 0.)

def save_image(src_img):
    m,n = len(img), len(img[0])
    dst_img = Image.new('L', (n, m))
    scaler = lambda x: min(255, max(0, int(x*255)))
    for i in range(m):
        for j in range(n):
            dst_img.putpixel((j,i), scaler(src_img[i][j]))
    dst_img.save('output.png')

def read_image():
    img = Image.open('input.png', 'r')
    data = list(img.getdata())
    #data = map(lambda x: x / 255., data)
    r = [data[i: i+img.width] for i in xrange(0, len(data), img.width)]
    return r

#random.seed(0)
#img = make_random_image(10, 10, 2)
#img = make_diagonal_image(3, 3)
img = read_image()
debug_message('orig_img', img)

restored_img = restoration(img)
debug_message('restored_img', restored_img)

diff = make_diff(img,restored_img)
debug_message('diff', diff)

diff_v = reduce(lambda acc,x: acc + abs_sum(x), diff, 0.)
debug_message('diff_v', diff_v)

save_image(restored_img)

