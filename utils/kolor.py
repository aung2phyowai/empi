#!/usr/bin/env python3
import sys
import numpy

def float2int(x):
	return int(255.0*(x**(1.0/2.2)))

kolory = {
	'Cz': [	1,	1,	1],
	'C3': [	1,	0,	0],
	'F3': [	1,	.5,	0],
	'Fz': [	1,	1,	0],
	'F4': [	.5,	1,	0],
	'C4': [	0,	1,	0],
	'P4': [	0,	1,	1],
	'Pz': [	0,	0,	1],
	'P3': [	1,	0,	1],
}

kolejnosc = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
wagi = numpy.array([0.2126, 0.7152, 0.0722])

Y0 = 0.5
for kanal in kolejnosc:
	kolor = numpy.array(kolory[kanal], dtype='float')
	Y = numpy.dot(wagi, kolor)
	if Y > Y0:
		kolor /= (Y / Y0)
	if Y < 0.5:
		korekcja = (Y0 - Y) / (1.0 - Y)
		kolor = (1-korekcja)*kolor + korekcja
	print(kanal, "\"#%02x%02x%02x\"" % (*map(float2int, kolor), ))
