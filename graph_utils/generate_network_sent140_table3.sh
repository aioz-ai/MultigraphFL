#!/usr/bin/env bash
# SENTIMENT-140 dataset - Amazon network
# >>> Baseline
python generate_networks.py amazon_us --experiment sent140 --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py amazon_us --experiment sent140 --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# SENTIMENT-140 dataset - Ebone network
# >>> Baseline
python generate_networks.py ebone --experiment sent140 --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py ebone --experiment sent140 --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# SENTIMENT-140 dataset - Gaia network
# >>> Baseline
python generate_networks.py gaia --experiment sent140 --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py gaia --experiment sent140 --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# SENTIMENT-140 dataset - Geant network
# >>> Baseline
python generate_networks.py geantdistance --experiment sent140 --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py geantdistance --experiment sent140 --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# SENTIMENT-140 dataset - Exodus network
# >>> Baseline
python generate_networks.py exodus --experiment sent140 --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
python generate_network_multigraph.py exodus --experiment sent140 --upload_capacity 1e10 --download_capacity 1e10 --arch ring