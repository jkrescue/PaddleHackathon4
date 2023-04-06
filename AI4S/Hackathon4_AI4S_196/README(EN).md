# User Guide
<u>_wanghanweibnds2015@gmail.com_</u>

---

English | [简体中文](./README(ZH).md)

## Introduction

---

Supervised data is commonly used when PINN method adopted.
Design of this suite is to extract pre-calculated CFD results
and transform such data into specific form which could be 
easily read under the framework of PaddleScience. OpenFOAM's data is now supported.

## Getting Started

---

### Download dependencies

---

`pip install -r dependencies/requirements.txt`

### Run examples

---

Simple examples could be found in directory `demo`.
An OpenFOAM case of cavity calculated by icoFoam solver is used to demonstrate the usage.
CFD results could be found under directory `demo/OpenFOAMCavity`.

Run the demo code in Python commandline.

Learn basic usage and data format:
```commandline
cd demo
python read_cavity.py
```

Implant CFD data into PINN program:
```commandline
cd demo
python insert_from_cavity.py
```

## API reference

---

[![Documentation Status](https://img.shields.io/badge/API_reference-blue.svg)](./doc/API-reference.md)