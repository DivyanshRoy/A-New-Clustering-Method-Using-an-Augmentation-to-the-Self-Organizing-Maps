import tensorflow as tf
import numpy as np
from tqdm import tqdm
from random import  randint
from numpy import genfromtxt

class Modified_SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """

    #To check if the SOM has been trained
    _trained = False

    def __init__(self, m, n, dim, indata,n_iterations=100, alpha=None, sigma=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.

        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """

        #Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))

        ##INITIALIZE GRAPH
        self._graph = tf.Graph()

        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE

            #Randomly initialized weightage vectors for all neurons,
            #stored together as a matrix Variable of size [m*n, dim]
            indata=np.array(indata)
            dist_array=np.zeros((indata.shape[0],1))
            for i in range(indata.shape[0]):
                for j in range(indata.shape[0]):
                    dist_array[i] = dist_array[i] + np.sqrt( (indata[i,0]-indata[j,0])*(indata[i,0]-indata[j,0]) + (indata[i,1]-indata[j,1])*(indata[i,1]-indata[j,1]) )
            sum_dist=0.0
            mn_dist=100000.0
            mx_dist=0.0

            for i in range(indata.shape[0]):
                sum_dist = sum_dist + dist_array[i]
                if dist_array[i]>mx_dist:
                    mx_dist=dist_array[i]
                if dist_array[i]<mn_dist:
                    mn_dist=dist_array[i]

            avg_dist=sum_dist/indata.shape[0]

            neuron_dist=np.zeros((m*n,1))
            diff_neuron = (mx_dist-mn_dist) / (m * n)

            uxy=np.zeros((m*n,2))
            for i in range(m*n):
                neuron_dist[i] = mn_dist + diff_neuron*(i-0.5)
                mn_2 = 1000000.0
                mn_index = -1
                for j in range(indata.shape[0]):
                    tmp_dist = np.fabs( neuron_dist[i] - dist_array[j] )
                    if tmp_dist<mn_2:
                        mn_2=tmp_dist
                        mn_index=j
                uxy[i,0] = indata[mn_index,0]
                uxy[i,1] = indata[mn_index,1]

            print(uxy)
            uxy = np.array(uxy)
            print(uxy.dtype)



            #self._weightage_vects = tf.Variable(tf.random_normal([m*n, dim]))
            self._weightage_vects = tf.Variable(tf.convert_to_tensor(uxy,tf.float64,name=None,preferred_dtype=tf.float64))
            print('1: ',np.array(self._weightage_vects))
            for i in range(self._weightage_vects.shape[0]):
                print(np.array(self._weightage_vects.read_value()[i][0])," , ",np.array(self._weightage_vects[i][1]))
            self._weightage_vects=tf.Variable(tf.constant(uxy))
            self._weightage_vects=tf.Variable(tf.cast(self._weightage_vects,tf.float32))
            print('2: ',self._weightage_vects.value())

            print(self._weightage_vects.dtype)

            #Matrix of size [m*n, 2] for SOM grid locations
            #of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))

            ##PLACEHOLDERS FOR TRAINING INPUTS
            #We need to assign them as attributes to self, since they
            #will be fed in during training

            #The training vector
            self._vect_input = tf.placeholder("float", [dim])
            #Iteration number
            self._iter_input = tf.placeholder("float")

            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            #Only the final, 'root' training op needs to be assigned as
            #an attribute to self, since all the rest will be executed
            #automatically during training

            #To compute the Best Matching Unit given a vector
            #Basically calculates the Euclidean distance between every
            #neuron's weightage vector and the input, and returns the
            #index of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self._weightage_vects, tf.stack([self._vect_input for i in range(m*n)])), 2), 1)),                                  0)

            #This will extract the location of the BMU based on the BMU's
            #index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]),dtype=tf.int64)),
                                 [2])

            #To compute the alpha and sigma values based on iteration
            #number
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)

            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                               for i in range(m*n)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),
                       self._weightage_vects))
            new_weightages_op = tf.add(self._weightage_vects,weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,new_weightages_op)


            ##INITIALIZE SESSION
            self._sess = tf.Session()

            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        #Nested iterations over both dimensions
        #to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
        #fig2 = plt.figure()

        #Training iterations
        for iter_no in tqdm(range(self._n_iterations)):
            #Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,self._iter_input: iter_no})

        #Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        #im_ani = animation.ArtistAnimation(fig2, centroid_grid, interval=50, repeat_delay=3000, blit=True)
        self._centroid_grid = centroid_grid
        #print(centroid_grid)


        self._trained = True
        #plt.show()

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def map_vects(self, input_vects,other):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")
        to_return=np.zeros((0,4))
        input_vects=np.array(input_vects)
        for vect,i in zip(input_vects,range(0,input_vects.shape[0])):
            to_return=np.insert(to_return,i,np.array((vect[0],vect[1],self._locations[min([i for i in range(len(self._weightages))],key=lambda x: np.linalg.norm(vect-self._weightages[x]))][0] , self._locations[min([i for i in range(len(self._weightages))],key=lambda x: np.linalg.norm(vect-self._weightages[x]))][1] )),0)
        return to_return