import numpy
import math

# LSH signature generation using random projection
def get_signature(user_vector, rand_proj): 
    res = 0
    for p in (rand_proj):
        res = res << 1
        val = numpy.dot(p, user_vector)
        if val >= 0:
            res |= 1
    return res

# get number of '1's in binary
# running time: O(# of '1's)
def nnz(num):
    if num == 0:
        return 0
    res = 1
    num = num & (num-1)
    while num:
        res += 1
        num = num & (num-1)
    return res     

# angular similarity using definitions
# http://en.wikipedia.org/wiki/Cosine_similarity
def angular_similarity(a,b):
    dot_prod = numpy.dot(a,b)
    sum_a = sum(a**2) **.5
    sum_b = sum(b**2) **.5
    cosine = dot_prod/sum_a/sum_b # cosine similarity
    theta = math.acos(cosine)
    return 1.0-(theta/math.pi)

if __name__ == '__main__':
    dim = 3 # number of dimensions per data
    d = 2**4 # number of bits per signature
    
    nruns = 24 # repeat times
    
    avg = 0
    for run in xrange(nruns):
        user1 = numpy.random.randn(dim)
        user2 = numpy.random.randn(dim)
        randv = numpy.random.randn(d, dim)    
        r1 = get_signature(user1, randv)
        r2 = get_signature(user2, randv)
        xor = r1^r2
        true_sim, hash_sim = (angular_similarity(user1, user2), (d-nnz(xor))/float(d))
        diff = abs(hash_sim-true_sim)/true_sim
        avg += diff
        print 'true %.4f, hash %.4f, diff %.4f' % (true_sim, hash_sim, diff) 
    print 'avg diff' , avg / nruns
