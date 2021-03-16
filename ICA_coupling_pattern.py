import numpy as np


class astro_pp_pattern_generator():
    def __init__(self, space_dims, sampling_scale=20):

        self.space_shp = space_dims
        self.sampling_scale = sampling_scale

        x = np.arange(space_dims[0])
        y = np.arange(space_dims[1])
        self.X, self.Y = np.meshgrid(x, y)


    def generate_pattern_landscape_parameters_normal_dist(self, num_of_clusters, amp_min_max=(1, 7), diam_min_max=(1,30)):

        '''
        
        :param num_of_clusters: NUMBER OF RADIAL BASIS FUNCTIONS
        :param amp_min_max: MIN AND MAX AMPLITUDE OF RADIAL BASIS FUNCTION
        :param diam_min_max: WIDTH OF RADIAL BASIS POINT
        :param thresh: 
        :return: 
        '''

        x_coor = np.random.randint(0, self.space_shp[0], num_of_clusters)
        y_coor = np.random.randint(0, self.space_shp[1], num_of_clusters)
        amp = np.random.normal(amp_min_max[0], amp_min_max[1], num_of_clusters)
        diam = np.random.normal(diam_min_max[0], diam_min_max[1], num_of_clusters)
        diam_max = np.amax(diam)
        diam = np.clip(diam,1,diam_max)

        return [x_coor, y_coor, amp, diam, num_of_clusters]



    def space_func_2d(self, new_param_list0, new_param_list1, arr_p_list0_in, arr_p_list1_in, arr_w_in, radius_vec_in):

        '''
        GENERATES CONTINUOUS SURFACE BASED ON THE RADIAL BASIS FUNCTION PARAMETERS GIVEN
        
        :param new_param_list0: X COORDINATES
        :param new_param_list1: Y COORDINATES
        :param arr_p_list0_in: x_coor OUTPUT FROM generate_pattern_landscape_parameters_normal_dist
        :param arr_p_list1_in: y_coor OUTPUT FROM generate_pattern_landscape_parameters_normal_dist
        :param arr_w_in: amp OUTPUT FROM generate_pattern_landscape_parameters_normal_dist
        :param radius_vec_in: diam OUTPUT FROM generate_pattern_landscape_parameters_normal_dist
        
        :return: 
        '''

        arr_p_list0 = np.expand_dims(arr_p_list0_in, len(np.shape(arr_p_list0_in)))
        arr_p_list0 = np.expand_dims(arr_p_list0, len(np.shape(arr_p_list0)))

        arr_p_list1 = np.expand_dims(arr_p_list1_in, len(np.shape(arr_p_list1_in)))
        arr_p_list1 = np.expand_dims(arr_p_list1, len(np.shape(arr_p_list1)))

        arr_w = np.expand_dims(arr_w_in, len(np.shape(arr_w_in)))
        arr_w = np.expand_dims(arr_w, len(np.shape(arr_w)))

        radius_vec = np.expand_dims(radius_vec_in, len(np.shape(radius_vec_in)))
        radius_vec = np.expand_dims(radius_vec, len(np.shape(radius_vec)))

        eval_vector = np.divide(
            np.sum(
                np.multiply(
                    np.exp(
                        np.multiply(
                            np.power(
                                np.divide(
                                    np.sqrt(
                                        np.add(
                                            np.power(
                                                np.subtract(
                                                    np.broadcast_to(np.array(new_param_list0),
                                                                    (len(arr_p_list0), np.shape(new_param_list0)[0],
                                                                     np.shape(new_param_list0)[1])),
                                                    np.broadcast_to(arr_p_list0, (
                                                        len(arr_p_list0), np.shape(new_param_list0)[0],
                                                        np.shape(new_param_list0)[1]))),2),
                                                np.power(
                                                    np.subtract(
                                                        np.broadcast_to(np.array(new_param_list1),
                                                                        (len(arr_p_list1), np.shape(new_param_list1)[0],
                                                                         np.shape(new_param_list1)[1])),
                                                        np.broadcast_to(arr_p_list1, (
                                                            len(arr_p_list1), np.shape(new_param_list1)[0],
                                                            np.shape(new_param_list1)[1]))),2)
                                        )
                                    ),
                                    radius_vec), 2), -1)),
                    np.broadcast_to(arr_w,
                                    (len(arr_p_list0), np.shape(new_param_list0)[0], np.shape(new_param_list0)[1]))),
                0),
            np.sum(np.absolute(arr_w)))


        return eval_vector

