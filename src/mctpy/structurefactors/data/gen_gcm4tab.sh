#!/bin/bash
# generate the interpolation table used by the gcm4MSA structure factor
# requires Wolfram Mathematica

math=`type -p math`
if test -z "$math"; then
  echo "cannot find mathematica cmdline interpreter"
  exit 9
fi

tmpfile=`mktemp --suffix .h5`
if test -z "$tmpfile"; then
  echo "tmpfile creation failed"
  exit 9
fi

( cat << EOF
vq = Integrate[(4Pi/k) r Sin[k r] Exp[-r^4], {r,0,Infinity}, Assumptions->k>0];
vqd = D[vq, k];
numtab = Block[{\$MaxExtraPrecision=100}, Table[{N[k],N[vq,30],N[vqd,30]},{k,0,51.19,1/20}]];
Export["$tmpfile",numtab];
EOF
) | $math

python3 -c "
import h5py
import numpy as np
with h5py.File('$tmpfile','r') as file:
  np.savez_compressed('gcm4MSA.npz',gcm4vq=file['Dataset1'])
"

rm $tmpfile
