from unittest import TestCase
from id3 import read_data
from id3 import infogain
from id3 import entropy


class TestId3(TestCase):
    def test_entropy(self):
        p1 = 0.5
        p2 = 1
        p3 = 3/5
        p4 = 2

        self.assertEqual(entropy(p1), 1)
        self.assertEqual(entropy(p2), 0)
        self.assertEqual(round(entropy(p3), 3), 0.971)
        with self.assertRaises(ValueError):
            entropy(p4)

    def test_InfoGain(self):
        # using toy data from tennis set
        humidity_high_py_pxi = 3
        humidity_high_pxi = 7
        humidity_high_py = 9
        total = 14

        humidity_high_expected = 0.152
        humidity_high_result = infogain(humidity_high_py_pxi, humidity_high_pxi, humidity_high_py, total)
        self.assertEquals(humidity_high_expected, round(humidity_high_result, 3))

        outlook_overcast_py_pxi = 4
        outlook_overcast_pxi = 4
        outlook_overcast_py = 9

        outlook_overcast_expected = 0.226
        outlook_overcast_result = infogain(outlook_overcast_py_pxi, outlook_overcast_pxi, outlook_overcast_py, total)
        self.assertEquals(outlook_overcast_expected, round(outlook_overcast_result,3))
