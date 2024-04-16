import unittest
import numpy as np
from funs import acosd

class TestAngleCalculations(unittest.TestCase):
    
    def test_phix_phiy_values(self):
        # Vectors and cosine values from your calculation
        cos_angle_nx = 0.20841928356798572
        cos_angle_ny = 0.15345586315107312

        # Expected results
        expected_phix = 12.029734293393275
        expected_phiy = 8.827252352585319

        # Calculate angles using acosd
        phix = 90 - acosd(cos_angle_nx)
        phiy = 90 - acosd(cos_angle_ny)

        # Asserting that the calculated values are almost equal to the expected values
        self.assertAlmostEqual(phix, expected_phix, places=4, msg="phix does not match expected value")
        self.assertAlmostEqual(phiy, expected_phiy, places=4, msg="phiy does not match expected value")

if __name__ == '__main__':
    unittest.main()
