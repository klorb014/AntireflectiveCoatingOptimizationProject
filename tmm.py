import numpy as np
import cmath
import math
import matplotlib.pyplot as plt 
import scipy.integrate as integrate


class TransferMatrixMethod:

	CENTRAL_WAVELENGTH_AIR = 650 * math.pow(10,-9)

	def __init__(layers: int, materials=None, thickness=None):
		self.layers = layers
		self.refractive_indices, self.center_wavelength = self.init_refractive_indices()
		self.interfaces = self.init_interface(self.layers, self.refractive_indices)
		self.dynamical_matrix = self.init_dynamical_matrix(self.layers, self.interfaces)
		self.layer_thicknesses = self.init_layer_thicknesses(self.layers, self.central_wavelength)
		


	def init_refractive_indices(self, layers: int, substrate: float) -> dict:
		""" This method initializes the refractive indices of the coating
			layers. Layer0 is the represents the air, so the refractive index 
			is assumed to be 1.000293. The last layer represents the solar panel
			substrate and has an assumed refractive index of 3.5. The """

		central_wavelength = {}
		refractive_indices = {}
		refractive_indices['n0'] = 1.000293
		central_wavelength['CW0'] = CENTRAL_WAVELENGTH_AIR/refractive_indices['n0']
		
		for l in range(layers):
			refractive_indices['n' + str(l+1)] = np.random.uniform(1.000293,3)
			central_wavelength['CW' + str(l+1)] = CENTRAL_WAVELENGTH_AIR/refractive_indices['n' + str(l+1)]
		refractive_indices['n' + str(layers+1)] = 3.5
		central_wavelength['CW' + str(layers+1)] = CENTRAL_WAVELENGTH_AIR/refractive_indices['n' + str(layers+1)]
		return refractive_indices, central_wavelength

	def init_interface(self, layers: int, refractive_indices: dict) -> dict:
		""" This method computes the reflectance and transmittance
			 at each interface. This calculation is based on Fresnel 
			 equation for incident light """

		interfaces = {}

		for l in range(layers):
			interfaces['R' + str(l) + str(l+1)] = (refractive_indices['n' + str(l+1)] - refractive_indices['n' + str(l)])/(refractive_indices['n' + str(l+1)] + refractive_indices['n' + str(l)])
			interfaces['T' + str(l) + str(l+1)] = 1 + interfaces['R' + str(l) + str(l+1)
		interfaces['R' + str(layers) + str(layers+1)] = (refractive_indices['n' + str(layers+1)] - refractive_indices['n' + str(layers)])/(refractive_indices['n' + str(layers+1)] + refractive_indices['n' + str(layers)])
		interfaces['T' + str(layers) + str(layers+1)] = 1 + interfaces['R' + str(layers) + str(layers+1)]

		return interfaces
		
	def init_dynamical_matrix(self, layers: int, interfaces: dict) -> dict :
		"""This method initializes the dynamical matrix used
			in the TMM. """

		dynamical_matrix = {}

		for l in range(layers):
			t = interfaces["T" + str(l) + str(l+1)]
			r = interfaces["T" + str(l) + str(l+1)]
			dynamical_matrix["Q" + str(l) + str(l+1)] = (1/t) * np.array([[1,r],[r,1]])

		t = interfaces["T" + str(layers) + str(layers+1)]
		r = interfaces["T" + str(layers) + str(layers+1)]
		dynamical_matrix["Q" + str(layers) + str(layers+1)] = (1/t) * np.array([[1,r],[r,1]])

		return dynamical_matrix

	def init_layer_thicknesses(self, layers: int, central_wavelengths: dict) -> dict:
		"""This method initializes the thickness for each layer"""
		layer_thicknesses = {}

		for l in range(layers):
			layer_thicknesses["d" + str(l+1)] = central_wavelengths["CW" + str(l+1)]
		
		return layer_thicknesses

	def calculate_transfer_matrix(self, wavelength: np.array) -> dict:
		"""This method computes the transfer matrix which characterizes the trasmission and reflection
		thought the antireflective coating. The transfer matrix is the matrix product of all the 
		dynamical and propagation matrices"""
		bounds = wavelength.shape[0]
		transfer_matrix = self.dynamical_matrix["Q01"]
		propagation_matrix = {}
		for l in range(self.layers):
			n = self.refractive_indices["n"+str(l+1)]
			d = self.layer_thicknesses["d"+str(l+1)]
			delta = ((2*cmath.pi)/wavelength)*n*d
			exp_plus = np.exp(1j*delta)
			exp_minus = np.exp(-1j*delta)

			zero = np.zeros(wavelength.shape)
			propagation_matrix = np.ravel(np.array([exp_plus,zero,exp_minus,zero]), order="F").reshape(bounds,2,2)
			transfer_matrix = transfer_matrix.dot(propagation_matrix)
			transfer_matrix = transfer_matrix.dot( self.dynamical_matrix["Q"+str(l)+str(l+1)])
		
		return transfer_matrix

	def calculate_reflectance(self, transfer_matrix: numpy.array) -> numpy.array:
		"""This method coating's reflectance for a particular wavelength"""
		T_1_1 = transfer_matrix.ravel()[::4]
		T_2_1 = transfer_matrix.ravel()[2::4]
		return np.divide(T_2_1,T_1_1)

	def getReflectivitySpectrum3Layers(dMatrix,layer1,layer2,layer3,lowerBound,upperBound):

	step = 1
	w = np.arange(lowerBound, upperBound , step)
	r = np.zeros((upperBound - lowerBound)/step)
	count = 0

	for wavelength in range(lowerBound,upperBound,step):
		
		wavelength = wavelength * math.pow(10,-9)

		tMatrix = calcTransferMatrix3Layers(dMatrix,layer1,layer2,layer3, wavelength)
		reflectivity = calcReflectivity(tMatrix) * 100
		r[count] = reflectivity
		count = count+1

	return w,r

def getTransmissivitySpectrum3Layers(dMatrix,layer1,layer2,layer3,lowerBound,upperBound):

	w = np.arange(lowerBound, upperBound ,1)
	t = np.zeros((upperBound - lowerBound))
	count = 0

	for wavelength in range(lowerBound,upperBound,1):
		
		wavelength = wavelength * math.pow(10,-9)
		tMatrix = calcTransferMatrix3Layers(dMatrix,layer1,layer2,layer3, wavelength)
		transmissivity = calcTransmissivity(tMatrix) * 100
		#print(wavelength* math.pow(10,9), transmissivity)
		t[count] = transmissivity
		count = count+1

	return w,t



def TripleLayerMain(n1,n2,n3,scale1,scale2,scale3):
	lowerBound = 400
	upperBound = 1400

	reflectCoefs, transCoefs, dynamicalMatrices = setMaterialParams3Layers(n1, n2,n3)
	layer1, layer2, layer3 = setDesignParams3Layers(n1,n2,n3,scale1,scale2,scale3)

	wavelength, reflectivity = getReflectivitySpectrum3Layers(dynamicalMatrices,layer1,layer2,layer3, lowerBound,upperBound)
	wavelength, transmissivity = getTransmissivitySpectrum3Layers(dynamicalMatrices,layer1,layer2, layer3, lowerBound,upperBound)


	power = calculatePower3Layers(dynamicalMatrices,layer1,layer2,layer3,200,2200)

	return wavelength,reflectivity,transmissivity, power

	def findOptimalTripleLayer():

	maxPower = 0
	maxPowerParams = [0]

	n1 = 1.4
	n2 = 1.6
	n3 = 3.15

	scale1 = 0.25
	scale2 = 0.25
	scale3 = 0.25

	while(n2<=3):  #test n1 and n2 for range of values (1.33 - 4)


		wavelength,reflectivity,transmissivity, power = TripleLayerMain(n1,n2,n3,scale1,scale2,scale3)

		PowerParams = [power,round(n2,2)]

		if (power > maxPower):
			maxPower = power
			maxPowerParams[0] = PowerParams
		print(round(n2,2), "Power = " + str(round(power,3)),  "Current Max: " + str(maxPowerParams[0]))

		n2 += 0.001

	return maxPowerParams

	def calculatePower3Layers(dMatrix,layer1,layer2,layer3,lowerBound,upperBound):
	power = integrate.quad(lambda x: getIntegrand3Layers(dMatrix,layer1,layer2,layer3,x),lowerBound,upperBound)[0]
	return power

def getIntegrand3Layers(dMatrix,layer1,layer2,layer3, wavelength):
	tMatrix = calcTransferMatrix3Layers(dMatrix,layer1,layer2,layer3, wavelength* math.pow(10,-9))
	transmissivity = calcTransmissivity(tMatrix)
	#transmissivity = 1
	irradiance = (6.16*math.pow(10,15))/(math.pow(wavelength,5) * ((math.exp((2484/wavelength))) - 1))
	return transmissivity*irradiance