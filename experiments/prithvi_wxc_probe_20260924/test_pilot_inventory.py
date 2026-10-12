import unittest
from experiments.prithvi_wxc_probe_20260924.prepare_pilot_inventory import selected_paths, validate_entry


class InventoryTests(unittest.TestCase):
    def test_real_date(self):
        paths = selected_paths('2020-01-01T14:00:00+08:00')
        self.assertEqual(paths[0], 'merra-2/MERRA2_sfc_20200101.nc')
        self.assertEqual(paths[3], 'climatology/climate_vertical_doy001_hour12.nc')

    def test_leap_mapping_matches_official_no_shift(self):
        self.assertIn('doy061', selected_paths('2020-03-01T14:00:00+08:00')[2])
        self.assertIn('doy365', selected_paths('2020-12-31T14:00:00+08:00')[2])

    def test_naive_or_other_hour_rejected(self):
        for value in ['2020-01-01T14:00:00', '2020-01-01T15:00:00+08:00']:
            with self.assertRaises(ValueError):
                selected_paths(value)

    def test_missing_hash_and_size_mismatch_rejected(self):
        for entry in [{'size': 2}, {'size': 2, 'lfs': {'oid': 'a'*64, 'size': 3}}]:
            with self.assertRaises(ValueError):
                validate_entry(entry)


if __name__ == '__main__':
    unittest.main()
