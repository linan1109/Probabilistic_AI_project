#!/bin/bash
# -*- coding: utf-8 -*-

#SBATCH --ntasks=6
#SBATCH --time=3:58:58
#SBATCH --job-name=solution
#SBATCH --mem-per-cpu=1024

python -u checker_client.py --base-dir ./ --results-dir ./
