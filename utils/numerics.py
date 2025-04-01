#Sparse system for quickly calculating physics informed losses from a differential equation
import torch
import numpy as np
from config.load_configurations import load_exp_configs

class calculus():

    def __init__(self):

        super(calculus, self).__init__()

    #function to take the derivative of a polynomial
    def derivative(self, coeffs, powers, deriv_dim):
        
        powers_der = powers.clone()
        coeffs_der = coeffs * powers[deriv_dim, :, 0, :]
        powers_der[deriv_dim, :, :, :] = powers[deriv_dim, :, :, :] - 1

        return coeffs_der, powers_der
    
    #function to multiply two polynomials with each other
    def poly_multiply(self, coeffs_1, powers_1, coeffs_2, powers_2):
        # Compute the product of coefficients
        # coeffs_1 shape: [batch_size, num_terms_1]
        # coeffs_2 shape: [batch_size, num_terms_2]
        batch_size = coeffs_1.size(0)
        coeffs_product = torch.einsum('bi,bj->bij', coeffs_1, coeffs_2)  # Outer product
        coeffs_product = coeffs_product.view(batch_size, -1)  # Flatten to [batch_size, num_terms_1 * num_terms_2]
        
        # Compute the sum of powers for each dimension
        # powers_1 shape: [N, batch_size, 1, num_terms_1]
        # powers_2 shape: [N, batch_size, 1, num_terms_2]
        powers_1_expanded = powers_1.unsqueeze(-1)  # [N, batch_size, 1, num_terms_1, 1]
        powers_2_expanded = powers_2.unsqueeze(-2)  # [N, batch_size, 1, 1, num_terms_2]
        powers_sum = powers_1_expanded + powers_2_expanded  # Sum exponents
        # Reshape to [N, batch_size, 1, num_terms_1 * num_terms_2]
        powers_product = powers_sum.view(powers_1.size(0), batch_size, 1, -1)
        
        return coeffs_product, powers_product

class PadeNumerics(calculus):

    def __init__(self):
        super(PadeNumerics, self).__init__()

        self.exp_configs = load_exp_configs()

        self.num_order = self.exp_configs["model"]["pade_layer"]["numerator_order"]                                #pade approximant numerator coefficients
        self.denom_order = self.exp_configs["model"]["pade_layer"]["denominator_order"]                            #pade approximant denominator coefficients
        self.calc = calculus()

    def eval_polynomial(self, X, coeffs, powers, order, dim):
        
        res = torch.zeros_like(X)
        for ord_index in range(ord):
            for dim_index in range(dim):
                

            


    def pade_approximant(self, X_vals_num, X_vals_denom, num_coeffs, denom_coeffs, num_powers, denom_powers, dim):
        
        for dim_index in range(dim):

            #raise to the power
            X_vals_num[dim_index, :, :, :] = (X_vals_num[dim_index, :, :, :].clone()**num_powers[dim_index, :, :, :])
            X_vals_denom[dim_index, :, :, :] = (X_vals_denom[dim_index, :, :, :].clone()**denom_powers[dim_index, :, :, :])

        #calculate multiplication of values along all dimensions
        precoefficient_num = X_vals_num[0, :, :, :]
        precoefficient_denom = X_vals_denom[0, :, :, :]

        if self.dim > 1:
            for dim_index in range(dim - 1):

                precoefficient_num = precoefficient_num*X_vals_num[dim_index + 1, :, :, :].clone()
                precoefficient_denom = precoefficient_denom*X_vals_denom[dim_index + 1, :, :, :].clone()

        #unsqueeze the coefficient array
        num_coeffs = num_coeffs.unsqueeze(0).unsqueeze(2)        
        denom_coeffs = denom_coeffs.unsqueeze(0).unsqueeze(2)    

        #calculate each term multiplied by the coefficient
        postcoefficient_num = num_coeffs * precoefficient_num           #size = [M, N, num_order]
        postcoefficient_denom = denom_coeffs * precoefficient_denom     #size = [M, N, denom_order]

        #sum along relevant dimensions
        sum_num = torch.sum(postcoefficient_num, (3))
        sum_denom = torch.sum(postcoefficient_denom, (3))

        pade = torch.div(sum_num, sum_denom + self.e)

        return pade
    
    def first_derivative(self, X, num_coeffs, num_powers, denom_coeffs, denom_powers, dim, deriv_dim):

        num_coeffs_der, num_powers_der = self.calc.derivative(num_coeffs, num_powers, deriv_dim)
        denom_coeffs_der, denom_powers_der = self.calc.derivative(denom_coeffs, denom_powers)





            
    


def test():

    print("Testing auxiliary numerics...")

    print("[TEST]: 1D derivative")

    clcs = calculus()
    
    #test 1 - derivatives

    coeffs_1 = torch.ones(100, 2)
    powers_1 = torch.ones(1, 100, 1, 2)

    coeffs_1[:, 1] = 0.5
    powers_1[:, :, :, 1] = 2

    coeffs_der, powers_der = clcs.derivative(coeffs_1, powers_1, 0)

    coeffs_der_result = np.array([1, 1])
    powers_der_result = np.array([0, 1])
    assert (coeffs_der[0, :].detach().numpy() == coeffs_der_result).all(), "Coefficients are incorrect for 1D polynomial derivative"
    assert (powers_der[0, 0, 0, :].detach().numpy() == powers_der_result).all(), "Powers are incorrect for 1D polynomial derivative"

    print("[TEST]: test passed")

    #test 2 - 1D polynomial multiplication
    print("[TEST]: 1D polynomial multiplication")

    coeffs_2 = torch.ones(100, 3)
    powers_2 = torch.ones(1, 100, 1, 3)

    powers_2[:, :, :, 1] = 3
    powers_2[:, :, :, 2] = 4

    coeffs_2[:, 1] = 2
    coeffs_2[:, 2] = 10

    coeffs_product, powers_product = clcs.poly_multiply(coeffs_1, powers_1, coeffs_2, powers_2)

    coeffs_result = np.array([1 , 2, 10, 0.5, 1, 5])
    powers_result = np.array([2, 4, 5, 3, 5, 6])
    assert (coeffs_product[0, :].detach().numpy() == coeffs_result).all(), "Coefficients are incorrect for 1D polynomial multiplication"
    assert (powers_product[0, 0, 0, :].detach().numpy() == powers_result).all(), "Powers are incorrect for 1D polynomial multiplication"

    print("[TEST]: test passed")

    # Test 3 - 3D polynomial multiplication
    print("[TEST]: 3D polynomial multiplication")

    # Create first polynomial: 1*x^1y^2z^3 + 0.5*x^2y^0z^1
    coeffs_1 = torch.tensor([[1.0, 0.5]])  # shape [1, 2]
    powers_1 = torch.tensor([
        [[[1.0, 2.0]]],  # x exponents (dimension 0)
        [[[2.0, 0.0]]],  # y exponents (dimension 1)
        [[[3.0, 1.0]]],  # z exponents (dimension 2)
    ])  # shape [3, 1, 1, 2]

    # Create second polynomial: 2*x^0y^1z^0 + 3*x^1y^0z^2 + 10*x^3y^0z^1
    coeffs_2 = torch.tensor([[2.0, 3.0, 10.0]])  # shape [1, 3]
    powers_2 = torch.tensor([
        [[[0.0, 1.0, 3.0]]],  # x exponents
        [[[1.0, 0.0, 0.0]]],  # y exponents
        [[[0.0, 2.0, 1.0]]],  # z exponents
    ])  # shape [3, 1, 1, 3]

    coeffs_product, powers_product = clcs.poly_multiply(coeffs_1, powers_1, coeffs_2, powers_2)

    # Expected results
    expected_coeffs = np.array([2.0, 3.0, 10.0, 1.0, 1.5, 5.0])
    expected_x_powers = np.array([1.0, 2.0, 4.0, 2.0, 3.0, 5.0])  # x-dimension
    expected_y_powers = np.array([3.0, 2.0, 2.0, 1.0, 0.0, 0.0])  # y-dimension
    expected_z_powers = np.array([3.0, 5.0, 4.0, 1.0, 3.0, 2.0])  # z-dimension

    # Verify coefficients
    assert np.allclose(coeffs_product[0].detach().numpy(), expected_coeffs), \
        "Coefficients are incorrect for 3D polynomial multiplication"

    # Verify exponents for each dimension
    assert np.allclose(powers_product[0, 0, 0, :].detach().numpy(), expected_x_powers), \
        "X-dimension powers are incorrect"
    assert np.allclose(powers_product[1, 0, 0, :].detach().numpy(), expected_y_powers), \
        "Y-dimension powers are incorrect"
    assert np.allclose(powers_product[2, 0, 0, :].detach().numpy(), expected_z_powers), \
        "Z-dimension powers are incorrect"

    print("[TEST]: test passed")

    print("All tests passed.")
    

