#!/usr/bin/gnuplot
set terminal postscript eps color size 6,3
set output "new.eps"
set style fill transparent solid .5 noborder
set xlabel "scale [s]"
set ylabel "frequency [Hz]"
set xrange [0.15:2.5]
plot 'new.log' u 7:8:(0.00002*$4) w circles title "MJ 02-04-2016 (Cz)"
