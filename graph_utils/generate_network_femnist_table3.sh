#!/usr/bin/env bash
# FEMNIST dataset - Amazon network
# >>> Baseline
python generate_networks.py amazon_us --experiment femnist --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py amazon_us --experiment femnist --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# FEMNIST dataset - Ebone network
# >>> Baseline
python generate_networks.py ebone --experiment femnist --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py ebone --experiment femnist --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# FEMNIST dataset - Gaia network
# >>> Baseline
python generate_networks.py gaia --experiment femnist --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py gaia --experiment femnist --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# FEMNIST dataset - Geant network
# >>> Baseline
python generate_networks.py geantdistance --experiment femnist --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py geantdistance --experiment femnist --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# FEMNIST dataset - Exodus network
# >>> Baseline
python generate_networks.py exodus --experiment femnist --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py exodus --experiment femnist --upload_capacity 1e10 --download_capacity 1e10 --arch ring