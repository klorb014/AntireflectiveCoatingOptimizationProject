import numpy as np
import cmath
import math
import scipy.integrate as integrate

# Optimizer Imports
from deap import base
from deap import benchmarks
from deap import creator


class TransferMatrixMethod:

    CENTRAL_WAVELENGTH_AIR = 650 * math.pow(10, -9)

    def __init__(self, layers: int, materials=None, thickness=None, spectrum_lower=None, spectrum_upper=None):
        self.layers = layers
        self.refractive_indices = materials if materials != None else self.init_refractive_indices()
        self.center_wavelength = self.init_center_wavelength(
            self.layers, self.refractive_indices)
        self.interfaces = self.init_interface(
            self.layers, self.refractive_indices)
        self.dynamical_matrix = self.init_dynamical_matrix(
            self.layers, self.interfaces)
        self.layer_thicknesses = thickness if thickness != None else self.init_layer_thicknesses(
            self.layers, self.center_wavelength)
        self.ideal_power = TransferMatrixMethod.get_ideal_power(spectrum_lower, spectrum_upper) if (
            spectrum_lower != None) and (spectrum_upper != None) else None

    @staticmethod
    def evaluate_solution(individual) -> float:
        """This static method accepts a candidate and computes the
            theoretical power and then normalizes it with the ideal
            power"""

        layers = len(individual)
        refractive_indices = {"n0": 1.0}
        thickness = {}
        for l in range(layers):
            refractive_indices["n"+str(l+1)] = individual[l]
            thickness["d"+str(l+1)] = 0.25 * \
                (TransferMatrixMethod.CENTRAL_WAVELENGTH_AIR/individual[l])
        refractive_indices["n"+str(layers+1)] = 3.5
        tmm = TransferMatrixMethod(layers, materials=refractive_indices,
                                   thickness=thickness, spectrum_lower=400, spectrum_upper=1400)
        candidate_power = tmm.calculate_power(400, 1400)
        return candidate_power/tmm.ideal_power,

    @ staticmethod
    def get_irradiance(wavelength: float) -> float:
        """This static method computes the idealized 'blackbody' irradiance.
            This quantity describes the amount of radiation absorbed by the solar
            cell at a particular wavelength. """
        return (6.16*math.pow(10, 15))/(math.pow(wavelength, 5) * ((math.exp((2484/wavelength))) - 1))

    @ staticmethod
    def get_ideal_power(lower_bound: int, upper_bound: int) -> float:
        """This static method computes the ideal power generated
            by the solar cell. This assumes transmissivity is 100%.
            """
        return integrate.quad(lambda x: TransferMatrixMethod.get_irradiance(x), lower_bound, upper_bound)[0]

    def init_refractive_indices(self, layers: int, materials: float) -> dict:
        """ This method initializes the refractive indices of the coating
                                        layers. Layer0 is the represents the air, so the refractive index
                                        is assumed to be 1.000293. The last layer represents the solar panel
                                        substrate and has an assumed refractive index of 3.5. The """

        refractive_indices = {}
        refractive_indices['n0'] = 1.000293
        for l in range(layers):
            refractive_indices['n' + str(l+1)] = np.random.uniform(1.000293, 3)
        refractive_indices['n' + str(layers+1)] = 3.5
        return refractive_indices

    def init_interface(self, layers: int, refractive_indices: dict) -> dict:
        """This method computes the reflectance and transmittance at each interface.
         This calculation is based on Fresnel equation for incident light"""

        interfaces = {}

        for l in range(layers):
            rf_ind_numerator = (
                refractive_indices['n'+str(l)] - refractive_indices['n'+str(l+1)])
            fr_ind_denominator = (
                refractive_indices['n'+str(l+1)] + refractive_indices['n'+str(l)])
            interfaces['R' + str(l) + str(l+1)
                       ] = rf_ind_numerator / fr_ind_denominator
            interfaces['T' + str(l) + str(l+1)] = 1 + \
                interfaces['R' + str(l) + str(l+1)]

        rf_ind_numerator = (
            refractive_indices['n' + str(layers)] - refractive_indices['n'+str(layers+1)])
        fr_ind_denominator = (
            refractive_indices['n' + str(layers+1)] + refractive_indices['n' + str(layers)])
        interfaces['R' + str(layers) + str(layers+1)
                   ] = rf_ind_numerator/fr_ind_denominator
        interfaces['T' + str(layers) + str(layers+1)] = 1 + \
            interfaces['R'+str(layers)+str(layers+1)]

        return interfaces

    def init_center_wavelength(self, layers: int, refractive_indices: dict) -> dict:
        """ This method initializes the central wavelength for each coating layer"""
        CENTRAL_WAVELENGTH_AIR = 650 * math.pow(10, -9)
        central_wavelength = {}
        central_wavelength['CW0'] = CENTRAL_WAVELENGTH_AIR / \
            refractive_indices['n0']

        for l in range(layers):
            central_wavelength['CW' + str(l+1)] = CENTRAL_WAVELENGTH_AIR / \
                refractive_indices['n' + str(l+1)]

        central_wavelength['CW' + str(layers+1)] = CENTRAL_WAVELENGTH_AIR / \
            refractive_indices['n' + str(layers+1)]
        return central_wavelength

    def init_dynamical_matrix(self, layers: int, interfaces: dict) -> dict:
        """This method initializes the dynamical matrix used
                                        in the TMM. """

        dynamical_matrix = {}

        for l in range(layers):
            t = interfaces["T" + str(l) + str(l+1)]
            r = interfaces["R" + str(l) + str(l+1)]
            dynamical_matrix["Q" + str(l) + str(l+1)
                             ] = (1/t) * np.array([[1, r], [r, 1]])

        t = interfaces["T" + str(layers) + str(layers+1)]
        r = interfaces["R" + str(layers) + str(layers+1)]
        dynamical_matrix["Q" + str(layers) + str(layers+1)
                         ] = (1/t) * np.array([[1, r], [r, 1]])
        return dynamical_matrix

    def init_layer_thicknesses(self, layers: int, central_wavelengths: dict) -> dict:
        """This method initializes the thickness for each layer"""
        layer_thicknesses = {}

        for l in range(layers):
            layer_thicknesses["d" +
                              str(l+1)] = central_wavelengths["CW" + str(l+1)]

        return layer_thicknesses

    def calculate_transfer_matrix(self, wavelength: np.array) -> np.array:
        """This method computes the transfer matrix which characterizes
        the trasmission and reflection thought the antireflective coating.
        The transfer matrix is the matrix product of all the dynamical and propagation matrices"""

        transfer_matrix = self.dynamical_matrix["Q01"]
        for l in range(self.layers):
            n = self.refractive_indices["n"+str(l+1)]
            d = self.layer_thicknesses["d"+str(l+1)]
            delta = ((2*cmath.pi)/wavelength)*n*d
            exp_plus = np.exp(1j*delta)
            exp_minus = np.exp(-1j*delta)

            propagation_matrix = np.array([[exp_plus, 0], [0, exp_minus]])

            transfer_matrix = transfer_matrix.dot(propagation_matrix)
            transfer_matrix = transfer_matrix.dot(
                self.dynamical_matrix["Q"+str(l+1)+str(l+2)])

        return transfer_matrix

    def calculate_reflectance(self, transfer_matrix: np.array) -> np.array:
        """This method coating's reflectance for a particular wavelength"""

        T_1_1 = cmath.polar(transfer_matrix[1][0])[0]
        T_2_1 = cmath.polar(transfer_matrix[0][0])[0]
        return (T_2_1/T_1_1)**2

    def calculate_transmittance(self, transfer_matrix: np.array) -> np.array:
        """This method coating's reflectance for a particular wavelength"""
        T_1_1 = cmath.polar(transfer_matrix[0][0])[0]
        n0, n_substrate = self.refractive_indices["n0"], self.refractive_indices["n"+str(
            self.layers+1)]
        return (n_substrate/n0)*((1/T_1_1)**2)

    def calculate_power(self, wavelength: np.array) -> float:
        """This method computes the theoretical power
                                        produced using this solar cell and antireflective
                                        coating."""

        transfer_matrix = self.calculate_transfer_matrix(wavelength)
        transmittance = self.calculate_transmittance(transfer_matrix)

        calc_irradiance = np.vectorize(lambda x: (
            6.16*(10**15))/((x**5)*(math.exp(2484/x)-1)))
        irradiance = calc_irradiance(wavelength)
        return np.sum(np.multiply(transmittance, irradiance))

    def calculate_power(self, lower_bound: int, upper_bound) -> float:
        """This method computes the theoretical power
                                        produced using this solar cell and antireflective
                                        coating."""
        power = integrate.quad(lambda x: self.get_integrand(x),
                               lower_bound, upper_bound)[0]
        return power

    def get_integrand(self, wavelength: float) -> float:
        transfer_matrix = self.calculate_transfer_matrix(
            wavelength * math.pow(10, -9))
        transmissivity = self.calculate_transmittance(transfer_matrix)

        irradiance = TransferMatrixMethod.get_irradiance(wavelength)
        return transmissivity*irradiance

    def calculate_spectrum(self, lower_bound: int, upper_bound) -> dict:
        length = int((upper_bound-lower_bound)/10)
        spectrum = {
            "S": np.arange(lower_bound, upper_bound, 10),
            "T": np.zeros(length),
            "R": np.zeros(length),
        }
        count = 0
        for wavelength in range(lower_bound, upper_bound, 10):
            wavelength = wavelength * math.pow(10, -9)

            transfer_matrix = self.calculate_transfer_matrix(wavelength)
            transmissivity = self.calculate_transmittance(
                transfer_matrix)*100
            reflectivity = 100-transmissivity

            spectrum["T"][count] = transmissivity
            spectrum["R"][count] = reflectivity
            count = count+1

        return spectrum
