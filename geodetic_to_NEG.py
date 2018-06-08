# First import the utm and numpy packages
import utm
import numpy

# To convert a GPS position (longitude, latitude, altitude) to a local position (north, east, down) you need to define
# a global home position as the origin of your NED coordinate frame. In general this might be the position your vehicle
# is in when the motors are armed, or some other home base position. You first task is to define a function to convert
# from global position to a local position using the utm.


def global_to_local(global_position, global_home):
    # TODO: Get easting and northing of global_home
    # TODO: Get easting and northing of global_position
    # TODO: Create local_position from global and home positions
    easting, northing, zone_number, zone_letter = utm.from_latlon(global_position[1], global_position[0])
    altitude = global_position[2]
    easting0, northing0, zone_number0, zone_letter0 = utm.from_latlon(global_home[1], global_home[0])
    altitude0 = global_home[2]
    local_position = numpy.array([northing-northing0, easting-easting0,  -(altitude-altitude0)])
    return local_position

# Now try converting a local position (north, east, down) relative to the home position to a global
# position (long, lat, up).


def local_to_global(local_position, global_home):
    # TODO: get easting, northing, zone letter and number of global_home
    # TODO: get (lat, lon) from local_position and converted global_home
    # TODO: Create global_position of (lat, lon, alt)

    eastHome, northHome, zoneHome, zoneLetter = utm.from_latlon(global_home[1], global_home[0])
    northing, easting,  altitude = local_position
    lat, lon = utm.to_latlon(easting+eastHome, northing+northHome, zoneHome, zoneLetter)
    global_position = numpy.array([lon, lat, -altitude])
    return global_position


# As an example, we will use two sets of global coordinates. One geodetic_home_coordinates serving as a local origin
# for NED frame and the second geodetic_current_coordinates which we will be expressed in NED coordinates relative to
# the first one.

if __name__ == "__main__":
    numpy.set_printoptions(precision=2)

    geodetic_current_coordinates = [-122.079465, 37.393037,  30]
    geodetic_home_coordinates = [-122.108432, 37.400154,  0]

    geodetic_current_coordinates = [-122.397435, 37.790358, 7]
    geodetic_home_coordinates = [-122.397450, 37.792480, 0]

    local_coordinates_NED = global_to_local(geodetic_current_coordinates, geodetic_home_coordinates)

    print('on roof:', local_coordinates_NED)
    # Should print [ -764.96  2571.59   -30.  ]

    numpy.set_printoptions(precision=6)
    NED_coordinates = [25.21, 128.07, -30.]

    # convert back to global coordinates
    geodetic_current_coordinates = local_to_global(NED_coordinates, geodetic_home_coordinates)

    print(geodetic_current_coordinates)
    # Should print [-122.106982   37.40037    30.      ]
