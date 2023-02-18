# Kinematic dynamos for Roberts flows

Scripts in this repository solve kinematic dynamo problems from Roberts (1972).
Here we use the pseudospectral [Dedalus](https://github.com/DedalusProject/dedalus) framework and search for peak growth rates via solutions to sparse eigenvalue problems (evps).

The scripts in this repo are:
- `roberts_flow_peak.py`: solve for peak growth rate at fixed `lambda`.  Reproduces growth rates in Roberts (1972)
- `roberts_flow_scan_lambda.py`: solve for growth curves, scanning in `lambda` and `kx`.  Reproduces Figure 7 of Roberts (1972)

Also included are:
- `roberts_flow.py`: solve for fastest growing mode at fixed `kx` and `lambda`
- `roberts_flow_scan_kx.py`: solve for growth curves at fixed `lambda`, scanning in `kx`
- `roberts_flow_3D_evp.py`: solve full 3-D evp; very slow

## Usage

### Growth curves

To compute growth curves at a variety of `lambda`, run
```bash
python roberts_flow_scan_lambda.py
```
which produces the following figure:
![Repoduction of Roberts (1972), Figure 7](https://github.com/bpbrown/kinematic_dynamos/blob/main/Roberts_1972_flow1_fig7_N32.png?raw=true)

This figure agrees very well, by eye, with the results in Roberts (1972), Figure 7.


### Peak growth rates

To compute peak growth rates at fixed `lambda`, but searching over `kx`, run:
```bash
python roberts_flow_peak.py
```
To reproduce all reported values in Roberts (1972), run these:
```bash
python roberts_flow_peak.py --flow=1
python roberts_flow_peak.py --flow=2
python roberts_flow_peak.py --flow=3
python roberts_flow_peak.py --flow=4 --lambda=1/64 --kx=1.5 --N=32
```
These yield the folowing values.

- flow 1, lambda=1/8
  - `ω_R = 0.17153310520117904, ω_I = 2.445162511133008e-16, at kx = 0.5384618674254258`
- flow 2, lambda=1/8
  - `ω_R = 0.02489507608928554, ω_I = 0.5090183158721694, at kx = 0.279865912211303`
- flow 3, lambda=1/8
  - `ω_R = 0.08972691904475676, ω_I = 0.4467167319782522, at kx = 0.27248575594328633`
- flow 4, lambda=1/64, N=32
  - `ω_R = 0.10714809611885308, ω_I = 3.062963590975353e-17, at kx = 1.6476979951898494`


The literature values from Roberts (1972) are (their `p => ω`, `j => kx`):
- flow 1, lambda=1/8 
  - `p = 0.173, j = 0.55`
- flow 2, lambda=1/8
  - `Re p = 0.025, j = 0.28`
- flow 3, lambda=1/8
  - `Re p = 0.09, j = 0.27`
- flow 4, lambda=1/64, N=32
  - `p = 0.10, j = 1.6`

All of these agree well, except for flow 1.  

There are reasons to suspect that the Roberts (1972) flow 1 peak growth rate is not fully converged in `kx`, and indeed another source (David Hughes, private correspondence) finds
- flow 1
  - `p = 0.17154, kx = 0.54`

which agrees well with the values produced by `python roberts_flow_peak.py --flow=1`.

## References
> Roberts, G.O., 1972,
> ``Dynamo action of fluid motions with two-dimensional periodicity'',
> Philosophical Transactions of the Royal Society of London
