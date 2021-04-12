import numpy
import sys
import cv2
import math
from scipy import signal
import matplotlib.pyplot as plt


def Hessian2D(I, Sigma):

    if Sigma < 1:
        print("error: Sigma<1")
        return -1
    I = numpy.array(I, dtype=float)
    Sigma = numpy.array(Sigma, dtype=float)
    S_round = numpy.round(3 * Sigma)

    [X, Y] = numpy.mgrid[-S_round:S_round + 1, -S_round:S_round + 1]


    DGaussxx = 1 / (2 * math.pi * pow(Sigma, 4)) * (X ** 2 / pow(Sigma, 2) - 1) * numpy.exp(
        -(X ** 2 + Y ** 2) / (2 * pow(Sigma, 2)))
    DGaussxy = 1 / (2 * math.pi * pow(Sigma, 6)) * (X * Y) * numpy.exp(-(X ** 2 + Y ** 2) / (2 * pow(Sigma, 2)))
    DGaussyy = 1 / (2 * math.pi * pow(Sigma, 4)) * (Y ** 2 / pow(Sigma, 2) - 1) * numpy.exp(
        -(X ** 2 + Y ** 2) / (2 * pow(Sigma, 2)))

    Dxx = signal.convolve2d(I, DGaussxx, boundary='fill', mode='same', fillvalue=0)
    Dxy = signal.convolve2d(I, DGaussxy, boundary='fill', mode='same', fillvalue=0)
    Dyy = signal.convolve2d(I, DGaussyy, boundary='fill', mode='same', fillvalue=0)

    return Dxx, Dxy, Dyy


def eig2image(Dxx, Dxy, Dyy):
    # This function eig2image calculates the eigen values from the
    # hessian matrix, sorted by abs value. And gives the direction
    # of the ridge (eigenvector smallest eigenvalue) .
    # inumpyut:Dxx,Dxy,Dyy
    # output:Lambda1,Lambda2,Ix,Iy
    # Compute the eigenvectors of J, v1 and v2
    Dxx = numpy.array(Dxx, dtype=float)
    Dyy = numpy.array(Dyy, dtype=float)
    Dxy = numpy.array(Dxy, dtype=float)


    if (len(Dxx.shape) != 2):
        print("len(Dxx.shape)!=2！")
        return 0

    tmp = numpy.sqrt((Dxx - Dyy) ** 2 + 4 * Dxy ** 2)

    v2x = 2 * Dxy
    v2y = Dyy - Dxx + tmp

    mag = numpy.sqrt(v2x ** 2 + v2y ** 2)
    i = numpy.array(mag != 0)  # only divides values zero, not close to zero

    v2x[i == True] = v2x[i == True] / mag[i == True]
    v2y[i == True] = v2y[i == True] / mag[i == True]


    v1x = -v2y.copy()
    v1y = v2x.copy()

    mu1 = 0.5 * (Dxx + Dyy + tmp)
    mu2 = 0.5 * (Dxx + Dyy - tmp)


    #plt.gray()
    #plt.imshow(mu1)
    #plt.imshow(mu2)

    check = abs(mu1) > abs(mu2)

    Lambda1 = mu1.copy()
    #plt.imshow(Lambda1)
    Lambda1[check == True] = mu2[check == True]
    #plt.imshow(Lambda1)
    Lambda2 = mu2.copy()
    #plt.imshow(Lambda2)
    Lambda2[check == True] = mu1[check == True]
    #plt.imshow(Lambda2)

    Ix = v1x
    Ix[check == True] = v2x[check == True]
    Iy = v1y
    Iy[check == True] = v2y[check == True]

    return Lambda1, Lambda2, Ix, Iy


def FrangiFilter2D(I):
    I = numpy.array(I, dtype=float)
    defaultoptions = {'FrangiScaleRange': (3, 9), 'FrangiScaleRatio': 2, 'FrangiBetaOne': 0.5, 'FrangiBetaTwo': 15,
                      'verbose': True, 'BlackWhite': True};
    options = defaultoptions

    sigmas = numpy.arange(options['FrangiScaleRange'][0], options['FrangiScaleRange'][1], options['FrangiScaleRatio'])
    sigmas.sort()

    beta = 2 * pow(options['FrangiBetaOne'], 2)
    c = 2 * pow(options['FrangiBetaTwo'], 2)


    shape = (I.shape[0], I.shape[1], len(sigmas))
    ALLfiltered = numpy.zeros(shape)
    ALLangles = numpy.zeros(shape)

    # Frangi filter for all sigmas
    Rb = 0
    S2 = 0
    for i in range(len(sigmas)):
        # Show progress
        if (options['verbose']):
            print('Current Frangi Filter Sigma: ', sigmas[i])

        # Make 2D hessian
        [Dxx, Dxy, Dyy] = Hessian2D(I, sigmas[i])

        # Correct for scale
        Dxx = pow(sigmas[i], 2) * Dxx
        Dxy = pow(sigmas[i], 2) * Dxy
        Dyy = pow(sigmas[i], 2) * Dyy

        # Calculate (abs sorted) eigenvalues and vectors
        [Lambda1, Lambda2, Ix, Iy] = eig2image(Dxx, Dxy, Dyy) #changed to lambda2

        # Compute the direction of the minor eigenvector
        angles = numpy.arctan2(Ix, Iy)
        #plt.gray()
        #plt.imshow(Lambda2)

        # Compute some similarity measures

        #Lambda2[Lambda2 == 0] = numpy.spacing(1) #change to lambda2
        near_zeros = numpy.isclose(Lambda2, numpy.zeros(Lambda2.shape))
        Lambda2[near_zeros] = 2 ** (-52)

        #plt.imshow(Lambda2)

        Rb = (Lambda1 / Lambda2) ** 2 #lambda1/2
        S2 = Lambda1 ** 2 + Lambda2 ** 2

        # Compute the output image

        Ifiltered = numpy.exp(-Rb / beta) * (numpy.ones(I.shape) - numpy.exp(-S2 / c))

        if (options['BlackWhite']):
            Ifiltered[Lambda2 > 0] = 0
        else:
            Ifiltered[Lambda2 < 0] = 0


        # store the results in 3D matrices
        ALLfiltered[:, :, i] = Ifiltered
        ALLangles[:, :, i] = angles


        if len(sigmas) > 1:
            outIm = numpy.amax(ALLfiltered, axis=2)
            outIm = outIm.reshape(I.shape[0], I.shape[1], order='F')
            whatScale = numpy.argmax(ALLfiltered, axis=2)
            whatScale = numpy.reshape(whatScale, I.shape, order='F')

            indices = range(I.size) + (whatScale.flatten(order='F') - 1) * I.size
            values = numpy.take(ALLangles.flatten(order='F'), indices)
            direction = numpy.reshape(values, I.shape, order='F')
        else:
            outIm = ALLfiltered.reshape(I.shape[0], I.shape[1], order='F')
            whatScale = numpy.ones(I.shape)
            direction = numpy.reshape(ALLangles, I.shape, order='F')

    return outIm, whatScale, direction



if __name__ == "__main__":
    imagename = numpy.fromfile("C:\Siemens\FrangiTesting\FlipSubnoiseGaussian_x592x592x1xushort.010", dtype='uint16')
    image = imagename.reshape(592, 592)

    imageInverted = numpy.invert(image) #Invert the DSA image so that vessels are light and non vessel region is dark


    blood = cv2.normalize(imageInverted.astype('double'), None, 0.0, 1.0,
                          cv2.NORM_MINMAX)  # Convert to normalized floating point
    frangiOutput = FrangiFilter2D(blood)

    outIm = frangiOutput[0] * (10000)


    outIm = cv2.normalize(outIm.astype('float32'), None, 0.0, 1.0,
                      cv2.NORM_MINMAX)


    outIm.astype('float32').tofile("C:\Siemens\FrangiTesting\FrangitestingNoiseGaussian_x592x592x1xsingle.010")
