#!/usr/bin/env bash
# INATURALIST dataset - Amazon network
# >>> Baseline
python generate_networks.py amazon_us --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py amazon_us --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# INATURALIST dataset - Ebone network
# >>> Baseline
python generate_networks.py ebone --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py ebone --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# INATURALIST dataset - Gaia network
# >>> Baseline
python generate_networks.py gaia --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py gaia --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# INATURALIST dataset - Geant network
# >>> Baseline
python generate_networks.py geantdistance --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py geantdistance --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# INATURALIST dataset - Exodus network
# >>> Baseline
python generate_networks.py exodus --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py exodus --experiment inaturalist --upload_capacity 1e10 --download_capacity 1e10 --arch ring