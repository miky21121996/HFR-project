<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>

<h3 align="center"> CLASS2-HCUR-MO</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
</div>

---

<p align="center"> This tool provides statistics comparing model data and high-frequency radar surface current velocities.
    <br> 
</p>

## Table of Contents

* [General Info](#General-Info)
* [Setup](#Setup)
* [Usage/Examples](#Usage/Examples)
* [Project Status](#Project-Status)
* [Contacts](#Contacts)

## General Info
The tool can manage the comparison of multiple experiments with the same observation dataset.


## Setup

Method 1: pip
```bash
pip install HFR-project
```

Method 2: Install From Source
```bash
git clone git@github.com:miky21121996/CLASS2-HCUR-HFR.git name_your_repo
cd name_your_repo
pip install .
```
To install the required packages and modules you can create a conda environment from the requirements.txt file:
```bash
conda create --name <new-environment-name> --file requirements.txt
conda activate <new-environment-name>
```
 
## Usage/Examples

```bash
usage: HFR_project [-h] [--link] [--destaggering] [--concat] [--plot_stats]

options:
  -h, --help      show this help message and exit
  --link          Link Model Files
  --destaggering  Destag Model Files
  --concat   Concatenate Model files along desired time period
  --plot_stats    Plot Statistics
```
Open ```configuration.ini``` to set all the necessary variables

Run using for example:
```bash
HFR_project --plot_stats 
```

## Project Status
Project is: _in progress_ 

## Contact
Created by Michele Giurato (michele.giurato@cmcc.it) - feel free to contact me! 
