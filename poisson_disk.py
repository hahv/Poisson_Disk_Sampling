import math

import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import pairwise_distances

from auto_tag_photos.sampling.heap import Heap


class PoissonDiskSampling:
    def __init__(self, points):

        self.points = points

        self.alpha = 8
        self.beta = 0.65
        self.gamma = 1.5
        self.domainSize = None

        self.dimensions = len(self.points.shape)
        self.boundsMin = np.zeros(self.dimensions)
        self.boundsMax = np.ones(self.dimensions)

    def do_sample_with_size(self, output_size):
        d_max = 2 * self.get_max_poisson_disk_radius(output_size)
        output_indices = self.do_eliminate(output_size, d_max)
        return output_indices

    def get_weight_limit_fraction(self, input_size, output_size):
        ratio = float(output_size) / float(input_size)
        fraction = (1.0 - math.pow(ratio, self.gamma)) * self.beta
        return fraction

    @staticmethod
    def weight_function(d2, d_min, d_max, alpha):
        d = d2
        if d < d_min:
            d = d_min
        return math.pow((1.0 - d / d_max), alpha)

    def do_eliminate(self, output_size, d_max):

        # Build a k-d tree for samples
        kd_tree = spatial.KDTree(self.points)

        # Assign weights to each sample
        w = np.zeros(self.points.shape[0])
        d_min = d_max * self.get_weight_limit_fraction(len(w), output_size)

        for index in range(len(w)):

            indices = kd_tree.query_ball_point(self.points[index], r=d_max)

            for i in range(len(indices)):
                neighbor_index = indices[i]

                if neighbor_index != index:
                    d_matrix = pairwise_distances([self.points[index]],
                                                  [self.points[neighbor_index]])
                    d2 = d_matrix.flatten()[0]

                    w[index] += PoissonDiskSampling.weight_function(d2=d2, d_min=d_min, d_max=d_max, alpha=self.alpha)

        # Build a heap for the samples using their weights
        heap = Heap()
        heap.SetDataPointer(w)
        heap.Build()

        output_indices = []

        sample_size = self.points.shape[0]

        while sample_size > output_size:
            index = int(heap.GetTopItemID())
            heap.Pop()

            # For each sample around it, remove its weight contribution and update the heap
            indices = kd_tree.query_ball_point(self.points[index], r=d_max)

            for i in range(len(indices)):
                neighbor_index = indices[i]

                if neighbor_index != index:
                    d_matrix = pairwise_distances([self.points[index]],
                                                  [self.points[neighbor_index]])
                    d2 = d_matrix.flatten()[0]
                    w[neighbor_index] -= PoissonDiskSampling.weight_function(d2=d2, d_min=d_min, d_max=d_max,
                                                                             alpha=self.alpha)
                    heap.MoveItemDown(neighbor_index)

            sample_size -= 1

        # Get the output result
        target_size = output_size
        for i in range(target_size):
            output_indices.append(heap.GetIDFromHeap(i))

        output_indices = np.array(output_indices).astype(int)

        return output_indices

    def get_max_poisson_disk_radius(self, sample_count):
        self.domainSize = self.boundsMax[0] - self.boundsMin[0]
        for d in range(1, self.dimensions):
            self.domainSize *= (self.boundsMax[d] - self.boundsMin[d])

        sample_area = self.domainSize / sample_count

        if self.dimensions == 2:
            r_max = math.sqrt(sample_area / (2 * math.sqrt(3)))
        elif self.dimensions == 3:
            r_max = math.pow(sample_area / (4 * math.sqrt(2)), 1.0 / 3)
        else:
            if self.dimensions & 1:
                c = 2
                d_start = 3
            else:
                c = math.pi
                d_start = 4

            for d in range(d_start, self.dimensions + 1, 2):
                c *= 2 * math.pi / d
            r_max = math.pow(sample_area / c, 1.0 / self.dimensions)

        return r_max
