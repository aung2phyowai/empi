#!/bin/sh
if [ $# -lt 1 ] ; then
  echo "USAGE: $0 type"
  exit 1
fi

gnuplot <<PLT
set terminal postscript eps color size 10,10
set output "$1.eps"
set multiplot layout 3,3 title "MJ 02-04-2016 ($1)"

set style fill transparent solid .5 noborder
set xlabel "scale [s]"
set ylabel "frequency [Hz]"
set xrange [0.15:2.5]
plot '1.log' u 7:8:(0.00002*\$4) w circles title "F3"
plot '2.log' u 7:8:(0.00002*\$4) w circles title "Fz"
plot '3.log' u 7:8:(0.00002*\$4) w circles title "F4"
plot '4.log' u 7:8:(0.00002*\$4) w circles title "C3"
plot '5.log' u 7:8:(0.00002*\$4) w circles title "Cz"
plot '6.log' u 7:8:(0.00002*\$4) w circles title "C4"
plot '7.log' u 7:8:(0.00002*\$4) w circles title "P3"
plot '8.log' u 7:8:(0.00002*\$4) w circles title "Pz"
plot '9.log' u 7:8:(0.00002*\$4) w circles title "P4"
PLT
