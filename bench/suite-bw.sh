IOSIM=../utils/bigfile-iosim

PREFIX=$1
NP=$2
NFILE=$3
SIZE=$4

STR=$NP-$NFILE-$SIZE

aprun -n $NP $IOSIM -f $NFILE -n $NP -s 1024 -w 1 create $PREFIX/bench-$STR
aprun -n $NP $IOSIM -f $NFILE -n $NP -s 1024 -w 1 read $PREFIX/bench-$STR
aprun -n $NP $IOSIM -f $NFILE -n $NP -s 1024 -w 1 -p update $PREFIX/bench-$STR
