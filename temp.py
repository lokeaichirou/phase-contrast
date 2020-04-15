# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import pylab
import numpy as np
import os
import scipy.ndimage
import scipy.fftpack

# Change directory to data file to fetch data
path_1 = "f:"  # Disk F

os.chdir(path_1)
retval_1 = os.getcwd()
print("当前工作目录为 %s" % retval_1)

path_2 = "\internship\DEEP LEARNING BASED PHASE RETRIEVAL FOR X-RAY PHASE CONTRAST IMAGING\data"
os.chdir(path_2)  # change to data file
retval_2 = os.getcwd()
print("当前工作目录为 %s" % retval_2)

oversampling = 4 # oversampling is already existing when we load the acr format data; we have oversampling then for 4
# times; Before CTF, we have to recover it to be 512

# Fetch the data of attenuation coefficients
fname_a = './a/acqui_0.acr'
a_0000 = np.fromfile(fname_a, dtype=np.float32, offset=182)  # Remove header of 182 bytes
a_0000 = a_0000.reshape(2048, 2048)  # Reshape it to be a 4*4*2048 tensor (2048 because of 512*4)
print(a_0000.shape)
fig0 = plt.figure(0)
plt.imshow(a_0000, cmap='gray')
fig0.suptitle(fname_a, fontsize=16)
plt.colorbar()
pylab.show()

# Fetch the data of phase shift
fname_ph = './ph/acqui_0.acr'
ph_0000 = np.fromfile(fname_ph, dtype=np.float32, offset=182)
ph_0000 = ph_0000.reshape(2048, 2048)
print(ph_0000.shape)
#fig1 = plt.figure(1)
#plt.imshow(ph_0000, cmap='gray')
#fig1.suptitle(fname_ph, fontsize=16)
#plt.colorbar()
#pylab.show()

# Image part
length_scale = 1e-6  # downsize it to the real range of distance
padding = 2  # padding is for the purpose of avoiding circular convolution
pixel_size = 1  # pixel size
nx_final = 512  # final picture length is 512.
ny_final = 512  # final picture width is 512.
nx = nx_final * oversampling  # nx = 2048  number of pixels horizontally
ny = ny_final * oversampling  # ny = 2048  number of pixels vertically

# Spatial domain
# ps = (pixel_size / oversampling) * 1e-6  # real range pixel size (resolution;real
ps = (7.5/oversampling)*1e-6
# range distance between two adjunct points)
x = np.linspace((-ps * nx / 2), ps * (nx / 2 - 1), nx)  # spatial domain
y = np.linspace((-ps * ny / 2), ps * (ny / 2 - 1), ny)
xx, yy = np.meshgrid(x, y)

# Optics part
energy = 24  # Energy of photon, unit is keV
#Lambda = 12.4e-10 / energy  # X-ray wavelength
Lambda = 0.5166e-10
distance = [0, 0.18, 0.318, 0.918]  # distance from exit plane of object to diffraction plane. Should be a set of 4 different
# distances for CTF method according to Dr Max's thesis?

# Wave part
wave = np.exp(-a_0000 + 1j * ph_0000)  # Refer to Formula (2.40), it is the transmittance function, because we assume
# the incident wave as uniform flux ; wave at the exit plane of object = Multiplication of original incident wave
# with transmittance function (decided by complex refractive index including attenuation coefficients and phase shift)
wave = np.pad(wave, ((ny // 2, ny // 2), (nx // 2, nx // 2)), 'edge') # edge padding is done to extend the boundary of 2D
# wave function representation.Since we want to keep this
wave = np.fft.fft2(wave)  # Fourier domain representation of wave at exit plane of object, same as u_0,(u node) as in
# Dr Max's report
# In numpy,by default, Fourier transform will do with last two axes automatically, don't worry about it
#wave = np.fft.fftshift(wave)

# Sampling part and frequency domain for PROPAGATOR
# Here refer to some knowledge of digital signal processing

fs = 1/ps  # corresponds to shannon-nyquist sampling frequency
f = np.linspace(-fs / 2, fs / 2 - fs / (nx * padding), nx * padding) # because fs is shannon-nyquist sampling frequency,
# the maximum detected frequency shoule be half of fs;
# horizontal representation of frequency domain; 4096 at this moment
g = np.linspace(-fs / 2, fs / 2 - fs / (ny * padding), ny * padding)  # vertical representation of frequency domain;
# 4096 at this moment
ff, gg = np.meshgrid(f, g)  # 4096*4096 at this moment

# PROPAGATOR part
# P=np.fft.ifftshift(np.exp(-1j*np.pi*Lambda*(distance)*(ff**2+gg**2)))
P = [0 for x in range(len(distance))]
for x in range(len(distance)):
    P[x] = np.fft.ifftshift(np.exp(-1j * np.pi * Lambda * distance[x] * (ff ** 2 + gg ** 2))) # P(f) PROPAGATOR in Fourier domain expression

# We can consider the free space propagation as a linear space invariant system, knowledge of signal and model could be
# implemented here for computation purpose

# Fresnel diffraction intensity part
Id = [0 for x in range(len(distance))]
for x in range(len(distance)):
    Id[x] = np.fft.ifft2(wave*P[x])  # complex form wave at certain distance (D) from the object (sample) or
# let's say wave on diffraction plane can be computed by computed as multiplication of wave and PROPAGATOR in Fourier
# domain, we recover it to be back to spatial domain, i.e. spatial representation using inverse Fourier transform
# Id size: 4096*4096 at this moment
# Because numpy has broadcasting, we don't have to mind the fact that wave is a 3 dimensional tensor while P is a 2
# dimensional matrix, numpy will copy P for 4 times for the multiplication processing

#Id = np.abs(Id) ** 2  # intensity of wave on diffraction plane is squared modulus of its representation
Id = np.abs(Id)  # intensity of wave on diffraction plane is squared modulus of its representation
Id = Id[::, ny // 2:-ny // 2, nx // 2:-nx // 2]  # We keep only the core part after the convolution (in Fourier domain,it is
# multiplication done above, at the same moment in spatial domain, it is convolution). This is in line with the wave
# before the convolution with PROPAGATOR
# Id size: 2048*2048 at this moment
Id = scipy.ndimage.zoom(Id, (1, 1/oversampling, 1/oversampling))  # Since the data obtained is oversampling data with oversampling rate = 4,
# we recover it to be back to 512 as very original
# Id size: 512*512 at this moment
print("Id shape", Id.shape)

FId = np.fft.fft2(np.pad(Id, ((0, 0), ((Id.shape[1]) // 2, (Id.shape[1]) // 2), ((Id.shape[2]) // 2, (Id.shape[2]) // 2)), 'edge'))
# compute the 2D Fourier transform of intensity of diffraction pattern, this would be the involved in ctf
for x in range(len(distance)):
    fig3 = plt.figure(x+2)
    plt.imshow(Id[x], cmap='gray')
    fig3.suptitle("Distance = %f for acqui_0"%distance[x], fontsize=16)
    plt.colorbar()
    pylab.show()

# show the image of intensity of diffraction pattern
print("Propagator Congratulation! finished")

Fresnel_number = [0.0*x for x in range(len(distance))]
for x in range(len(distance)):
    Fresnel_number[x] = Lambda * distance[x]/(length_scale**2)


class PhaseRetrievalAlgorithm2D:
    """Reconstruction algorithms should be given as an object I guess. Should
    have a function reconstruct that eats dataSet?"""

    def __init__(self):
        print("we have entered the class PhaseRetrievalAlgorithm2D ")

        self.lengthscale = 1e-6   # downsize it to the real range of distance

        self.nx = 512  # width of Fourier transform of diffraction pattern intensity image
        self.ny = 512  # length of Fourier transform of diffraction pattern intensity image
        self.padding = 2  # padding for 2 times
        self.nfx = self.nx * self.padding  # TODO: Double with DS/flex w pad size
        # width of Fourier transform of diffraction pattern intensity image after padding
        self.nfy = self.ny * self.padding
        # length of Fourier transform of diffraction pattern intensity image after padding

        self.pixel_size = 7.5*1e-6
        # self.sample_frequency = 1 / (self.pixel_size * self.lengthscale)
        self.sample_frequency = self.lengthscale/self.pixel_size
        # Corresponds to Shannon-nyquist sampling frequency
        self.fx, self.fy = self.FrequencyVariable(self.nfx, self.nfy, self.sample_frequency)
        # 1024*1024 at this moment

        self.ND = len(distance)  # number of distances in experiment set-up

        self.coschirp = [0 for xxx in range(self.ND)] # corresponds to cos "chirp" term for different distances in formula 4.14
        self.sinchirp = [0 for xxx in range(self.ND)] # corresponds to sin "chirp" term for different distances in formula 4.14

        for distances in range(self.ND):
            self.coschirp[distances] = np.cos((np.pi * Fresnel_number[distances]) * (self.fx ** 2) + (np.pi * Fresnel_number[distances]) * (self.fy ** 2))
            # corresponds to cos "chirp" term in formula 4.14
            self.sinchirp[distances] = np.sin((np.pi * Fresnel_number[distances]) * (self.fx ** 2) + (np.pi * Fresnel_number[distances]) * (self.fy ** 2))
            # corresponds to sin "chirp" term in formula 4.14

        self.alpha = 1e-8  # should be given as the exponential only?
        self.alpha_cutoff = 0.5
        self.alpha_cutoff_frequency = self.alpha_cutoff * self.sample_frequency
        # TODO: should be a property (dynamically calculated from alpha_cutoff)
        self.alpha_slope = .5e3

    def FrequencyVariable(self, nfx, nfy, sample_frequency):
        # creation of frequency variable
        # According to numpy fft convention;
        # a[0] should contain the zero frequency term,
        # a[1:n//2] should contain the positive-frequency terms,
        # a[n//2 + 1:] should contain the negative-frequency terms, in increasing 
        # order starting from the most negative frequency.        
        #f = np.linspace(-sample_frequency / 2, sample_frequency / 2 - sample_frequency / (nx * self.padding), nx * self.padding)  # horizontal representation of frequency domain
        #g = np.linspace(-sample_frequency / 2, sample_frequency / 2 - sample_frequency / (ny * self.padding), ny * self.padding)  # vertical representation of frequency domain

        #f = scipy.fftpack.fftfreq(nfx, d=1/sample_frequency)
        #f_shift = scipy.fftpack.fftshift(f)
        #g = scipy.fftpack.fftfreq(nfy, d=1/sample_frequency)
        #g_shift = scipy.fftpack.fftshift(g)

        x = 0
        x = np.append(x, np.linspace(sample_frequency / nfx, sample_frequency / 2, nfx // 2))
        x = np.append(x, np.linspace(-sample_frequency / 2 + sample_frequency / nfx, -sample_frequency / nfx,
                                     nfx // 2 - 1 ))  # + (np.ceil(nfx / 2) - nfx // 2)

        y = 0
        y = np.append(y, np.linspace(sample_frequency / nfy, sample_frequency / 2, nfy // 2))
        y = np.append(y, np.linspace(-sample_frequency / 2 + sample_frequency / nfy, -sample_frequency / nfy,
                                     nfy // 2 - 1 ))  # + (np.ceil(nfy / 2) - nfy // 2)

        #return np.meshgrid(f_shift, g_shift)

        return np.meshgrid(x, y)


class CTF(PhaseRetrievalAlgorithm2D):
    def __init__(self):
        super().__init__()
        print("we have entered the class CTF")
        self.A = np.zeros((self.nfy, self.nfx))
        # corresponds to term A in formula 4.17, A has same shape as fx and fx
        self.B = self.A.copy()
        # corresponds to term B in formula 4.17, A has same shape as fx and fx
        self.C = self.A.copy()
        # corresponds to term C in formula 4.17, A has same shape as fx and fx

        for distances in range(self.ND):
            self.A += self.sinchirp[distances] * self.coschirp[distances]
            # A is summation of product of sin "chirp" and cos "chirp" for different distances according to (4.14)
            self.B += self.sinchirp[distances] * self.sinchirp[distances]
            # B is summation of product of sin "chirp" and sin "chirp" for different distances according to (4.14)
            self.C += self.coschirp[distances] * self.coschirp[distances]
            # C is summation of product of cos "chirp" and cos "chirp" for different distances according to (4.14)

        self.Delta = self.B * self.C - self.A ** 2
        # corresponds to delta in (4.14)

        # FID = [0 for x in range(ND)] #TODO: test variable

        # TODO: normalisation of CTF factors is not explicit in Zabler 
        # but should probably be done?
        # firecp = firecp/length(planes);
        # fireca = fireca/length(planes);
        # fig = pyplot.imshow(self.sinchirp[3])
        # pyplot.colorbar()

    def reconstruct_projection(self, *argv):
        # TODO The interface is not nice. Should probably be with all kwargs?
        fID = [0 for x in range(self.ND)]  # different distances have different Fresenel diffraction pattern, and definitely
        # different corresponding Fourier transform result
        for distances in range(0, self.ND):
            fID[distances] = FId[distances]
        # Generate CTF factors
        # TODO: should possibly be done in constructor
        sin_ctf_factor = np.zeros((self.nfy, self.nfx))
        cos_ctf_factor = sin_ctf_factor.copy()

        for distances in range(self.ND):
            sin_ctf_factor = sin_ctf_factor + self.sinchirp[distances] * fID[distances] # first summation term in (4.17)
            cos_ctf_factor = cos_ctf_factor + self.coschirp[distances] * fID[distances] # second summation term in (4.17)
            # TODO: The removal of the delta is not explicit in the paper 
            # but should probably be done
            # s{k}(1,1) -= nf*mf; # remove 1 in real space
            # TODO: verify correct padding

        phase = (self.C * sin_ctf_factor - self.A * cos_ctf_factor) / (2 * self.Delta + self.alpha)
        # formula (4.17)
        attenuation = (self.A * sin_ctf_factor - self.B * cos_ctf_factor) / (2 * self.Delta + self.alpha)

        phase = np.real(np.fft.ifft2(phase))
        # take its real part of complex form phase
        attenuation = np.real(np.fft.ifft2(attenuation))

        # truncate to nx-ny
        phase = phase[self.ny // 2:-self.ny // 2, self.nx // 2:-self.nx // 2]
        # We keep only the core part after the convolution (in Fourier domain, it is
        # multiplication done above, at the same moment in spatial domain, it is convolution).
        attenuation = attenuation[self.ny // 2:-self.ny // 2, self.nx // 2:-self.nx // 2]

        return phase, attenuation


ctf = CTF()  # define a ctf object
phase_retrieval, attenuation_retrieval = ctf.reconstruct_projection()  # apply reconstruct_projection method, obtain the phase retrieval result
fig11 = plt.figure(11)
plt.imshow(phase_retrieval, cmap='gray')  # plot result image
fig11.suptitle("phase_retrieval for acqui_0 based on 4 distances", fontsize=16)
plt.colorbar()
pylab.show()

fig12 = plt.figure(12)
plt.imshow(attenuation_retrieval, cmap='gray')  # plot result image
fig12.suptitle("attenuation_retrieval for acqui_0 based on 4 distances", fontsize=16)
plt.colorbar()
pylab.show()
print("all finished!")

print("see you next time")
