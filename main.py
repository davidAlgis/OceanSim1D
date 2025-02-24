import argparse
from ocean.oceanwaves import OceanWaves
from ocean.drawer import OceanDrawer


def main():
    parser = argparse.ArgumentParser(
        description="1D Tessendorf Ocean Simulation")
    parser.add_argument('-n',
                        '--n',
                        type=int,
                        default=256,
                        help="Number of grid points (default: 256)")
    parser.add_argument('-l',
                        '--length',
                        type=float,
                        default=100.0,
                        help="Domain length in meters (default: 100.0)")
    parser.add_argument('-w',
                        '--wind_speed',
                        type=float,
                        default=10.0,
                        help="Wind speed in m/s (default: 10.0)")
    parser.add_argument('-f',
                        '--fetch',
                        type=float,
                        default=1000.0,
                        help="Fetch in meters (default: 1000.0)")
    parser.add_argument(
        '-d',
        '--water_depth',
        type=float,
        default=1e6,
        help="Water depth (default: 1e6, deep water assumption)")
    parser.add_argument(
        '-t',
        '--time_scale',
        type=float,
        default=10.0,
        help="Time scaling factor for visualization (default: 10.0)")
    args = parser.parse_args()

    # Create the physics simulation instance.
    ocean_instance = OceanWaves(args.n, args.length, args.wind_speed,
                                args.fetch, args.water_depth)

    # Create the drawer instance and run the simulation.
    drawer = OceanDrawer(ocean_instance, args.length, 10, 100 * args.n,
                         args.time_scale)
    drawer.run()


if __name__ == '__main__':
    main()
