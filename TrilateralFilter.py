import numpy
import math
import cv2
import FrangiVesselDetection
from numba import jit
import time

sigmaI = 15
sigmaS = 6
kernel = 21
sigmaF = 0.0005

intensity_vector = numpy.zeros(4096, dtype=float)
gaussian_kernel_matrix = numpy.zeros((kernel,kernel), dtype=float)
frangi_kernel_matrix = numpy.zeros((kernel,kernel), dtype=float)


def create_intensity_matrix():
    i =0
    for i in range(len(intensity_vector)):
        val = (i ** 2) / (2 * sigmaI ** 2)
        intensity_vector[i] = (1.0 / (numpy.sqrt(2 * math.pi )* (sigmaI ** 2))) * math.exp(-val)
    i += 1

def create_gaussian_matrix():
    half_kernel = math.floor(kernel/2)
    i = -half_kernel
    while i < (half_kernel + 1):
        j = -half_kernel
        while j <(half_kernel + 1):
            gaussian_kernel_matrix[i + half_kernel][j+half_kernel] = gaussian2(sigmaS, i, j)
            j += 1
        i += 1


def gaussian2(sigma, x, y):
    gauss2 = (1.0 / (numpy.sqrt(2.0 * math.pi) * sigma * sigma)) * math.exp(-(((x * x ) + (y * y)) / (2 * sigma * sigma)))
    return gauss2

def distance(x, y, i, j):
    return numpy.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    val = (x ** 2) / (2 * sigma ** 2)
    gauss = (1.0 / (numpy.sqrt(2 * math.pi )* (sigma ** 2))) * math.exp(-val)
    return gauss

def frangiGaussian(source, i, j):

    frangiVesselnessValue = source[i][j]
    val = (frangiVesselnessValue ** 2) / (2 * sigmaF ** 2)
    gf = (1.0 / (numpy.sqrt(2 * math.pi )* (sigmaF ** 2))) * math.exp(-val)
    return gf

@jit(nopython=True)
def apply_trilateral_filter(source, frangiVesselness, filtered_image, x, y, kernel, sigma_i, sigma_s):
    half_kernel = math.floor(kernel/2)
    numerator_sum = 0
    denominator_sum = 0
    i = (x - half_kernel)
    while i < (x + half_kernel):
        j = (y - half_kernel)
        while j < (y + half_kernel):
            if ( i >= 0 and j >= 0 and i < len(source) and j < len(source[0])):
                intensity_difference = numpy.abs(int(source[i][j]) - int(source[x][y]))
                frangiDifference = frangiVesselness[i][j] - frangiVesselness[x][y]

                gf =  math.exp(-((frangiDifference ** 2) / (2 * sigmaF ** 2)))
                gi = (1.0 / (numpy.sqrt(2 * math.pi )* (sigmaI ** 2))) * math.exp(-((intensity_difference ** 2) / (2 * sigmaI ** 2)))
                gs = gaussian_kernel_matrix[x - i + half_kernel][y - j + half_kernel]

                w = gs * gi * gf
                numerator_sum += w * source[i][j]
                denominator_sum += w
            j += 1
        i += 1
    i_filtered = numerator_sum / denominator_sum
    filtered_image[x][y] = int(round(i_filtered))

def trilateral_filter(source, filter_diameter, frangiVesselness, sigma_i, sigma_s):
    filtered_image = numpy.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_trilateral_filter(source, frangiVesselness, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    return filtered_image


if __name__ == "__main__":

    imagename = numpy.fromfile("DSA_x464x464x1xushort.raw", dtype='uint16')
    image = imagename.reshape(464, 464)


    imageInverted = numpy.invert(image) #Invert the DSA image so that vessels are light and non vessel region is dark

    blood = cv2.normalize(imageInverted.astype('double'), None, 0.0, 1.0,
                          cv2.NORM_MINMAX)  # Convert to normalized floating point

    start = time.time()

    FrangiOutput = FrangiVesselDetection.FrangiFilter2D(blood)
    outIm = FrangiOutput[0] * (10000)

    outIm = cv2.normalize(outIm.astype('double'), None, 0.0, 1.0,
                  cv2.NORM_MINMAX)


    create_gaussian_matrix()
    filtered_image_own = trilateral_filter(image, kernel, outIm, sigmaI, sigmaS)

    filtered_image_own.astype('uint16').tofile("TF_x464x464x1xushort.raw")

