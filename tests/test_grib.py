import pytest

from haze_forecast.grib import parse_wgrib2_point_output


def test_parse_wgrib2_point_output_reads_nearest_grid_value():
    point = parse_wgrib2_point_output(
        "1:0:d=2021032318:UGRD:10 m above ground:6 hour fcst::"
        "lon=104.000000,lat=30.750000,val=-1.19978\n"
    )

    assert point.longitude == 104.0
    assert point.latitude == 30.75
    assert point.value == -1.19978


@pytest.mark.parametrize("output", ["", "val=missing", "lon=104,lat=30,val=undef"])
def test_parse_wgrib2_point_output_rejects_missing_values(output):
    with pytest.raises(ValueError):
        parse_wgrib2_point_output(output)
