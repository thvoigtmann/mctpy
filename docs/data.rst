.. include:: <isolat1.txt>

HDF5 Data Format
================

`mctpy` offers to save and retrieve correlator data in HDF5 files.
A typical structure of these files is that they provide a `model`
group storing information about the parameters of the model etc.,
and a `correlator` group with the time-dependent or frequency-dependent
data.

The version of `mctpy` used to create a given file is stored in the
`version` attribute of a top-level `mctpy` group in the file.
This is meant to verify that in case of bugfixes one can check the
integrity of the data.
