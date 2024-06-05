import torch
import numpy as np
import unittest
import sys
import os
sys.path.append(os.path.abspath(os.pardir))
from nan_ops import NaNPool2d

class TestNaNPool2d(unittest.TestCase):

    def unit_test_indices(self, input_tensor, expected, test_torch=True, test_expected=True):
        nanPoolPy = NaNPool2d(0.25) 
        torchpool = torch.nn.MaxPool2d(2, 2, return_indices=True)

        default = torchpool(input_tensor)[1]
        testing = nanPoolPy(input_array=input_tensor, pool_size=(2, 2), strides=(2, 2))[1]

        if test_torch:
            self.assertTrue(torch.equal(default, testing), "Torch and NaNPool indices do not match.")
            print("Test passed: Torch and NaNPool indices match.")
        else: print('NaNPool expected behaviour differs from Torch -- comparison to Torch is skipped')
        # print(default , testing)

        print(expected , testing)
        if test_expected:
            self.assertTrue(torch.equal(expected, testing), "Expected and NaNPool indices do not match.")
            print("Test passed: Expected and NaNPool indices match.")
        else: print('NanPool expected behaviour matches Torch -- no need to run additional test')
        #print(expected , testing)

    def unit_test_maxvalues(self, input_tensor, expected, test_torch=True, test_expected=True):
        nanPoolPy = NaNPool2d(0.25) 
        torchpool = torch.nn.MaxPool2d(2, 2, return_indices=True)

        default = torchpool(input_tensor)[0]
        default = default.masked_fill(torch.isnan(default), 0.5)
        testing = nanPoolPy(input_array=input_tensor, pool_size=(2, 2), strides=(2, 2))[0]
        testing = testing.masked_fill(torch.isnan(testing), 0.5)
        # print(default , testing)

        if test_torch:
            self.assertTrue(torch.equal(default, testing), "Torch and NaNPool max values do not match.")
            print("Test passed: Torch and NaNPool max values match.")
        else: print('NaNPool expected behaviour differs from Torch -- comparison to Torch is skipped')
        # print(default , testing)

        if test_expected:
            self.assertTrue(torch.equal(expected, testing), "Expected and NaNPool max values do not match.")
            print("Test passed: Expected and NaNPool max values match.")
        else: print('NanPool expected behaviour matches Torch -- no need to run additional test')
        # print(expected , testing)

    def test_all_nans(self):
        """ 
        Example Input:
        [[[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]]]
        """
        input = torch.randint(0, 100, (2, 2, 2, 2)).float()
        input[:] = np.nan
        # print('INPUT:\n', input, input.shape)
        expected_idx = torch.tensor([[[[0]], [[0]]], [[[0]], [[0]]]])
        expected_val = torch.tensor([[[[0.5]], [[0.5]]], [[[0.5]], [[0.5]]]])
        self.unit_test_indices(input, expected_idx, test_torch=False, test_expected=True)
        self.unit_test_maxvalues(input, expected_val, test_torch=True, test_expected=True)

    def test_no_nans(self):
        """ 
        Example Input:
        [[[[90., 78., 10., 12.],
          [ 6., 64.,  8., 62.]],

         [[75., 39., 51., 52.],
          [44., 79., 66., 34.]]],


        [[[14., 29., 44., 36.],
          [60., 29., 70., 29.]],

         [[12., 93., 76., 94.],
          [22., 44., 20., 58.]]]]
        """
        input = torch.randperm(100)[:2*2*2*4].float()
        input = input.reshape(2, 2, 2, 4)
        # print('INPUT:\n', input, input.shape)
        self.unit_test_indices(input, None, test_torch=True, test_expected=False)
        self.unit_test_maxvalues(input, None, test_torch=True, test_expected=False)

    def test_mixed_nans_multimaxval(self):
        """ 
        Example Input:
        [[[[nan,  1.],
          [ 1.,  1.]],

         [[ 1.,  1.],
          [ 1.,  1.]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, 10.]]]]
        """
        input = torch.randint(0, 100, (2, 2, 2, 2)).float()
        input[:] = np.nan
        input[0] = 1
        input[0, 0, 0, 0] = np.nan
        input[1,1,1,1] = 10
        # print('INPUT:\n', input, input.shape)
        expected_idx = torch.tensor([[[[0]], [[0]]], [[[0]], [[3]]]])
        expected_val = torch.tensor([[[[0.5]], [[0.5]]], [[[0.5]], [[10]]]])
        self.unit_test_indices(input, expected_idx, test_torch=False, test_expected=True)
        self.unit_test_maxvalues(input, expected_val, test_torch=False, test_expected=True)

    def test_mixed_nans_no_multimaxval(self):
        """ 
        Example Input:
        [[[[nan, nan],
          [33.,  3.]],

         [[75., 59.],
          [53., 24.]]],


        [[[61., 64.],
          [26., 27.]],

         [[67., 68.],
          [11., nan]]]]
        """
        input = torch.tensor([[[[np.nan, np.nan], [33.,  3.]],
                         [[75., 59.], [53., 24.]]],
                         [[[61., 64.], [26., 27.]],
                          [[67., 68.], [11., np.nan]]]])

        # print('INPUT:\n', input, input.shape)
        expected_idx = torch.tensor([[[[2]], [[0]]], [[[1]], [[1]]]])
        expected_val = torch.tensor([[[[33]], [[75]]], [[[64]], [[68]]]])
        self.unit_test_indices(input, expected_idx, test_torch=False, test_expected=True)
        self.unit_test_maxvalues(input, expected_val, test_torch=False, test_expected=True)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNaNPool2d)
    unittest.TextTestRunner().run(suite)