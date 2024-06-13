cuda_cavity: 10_cavity.cu
	nvcc -O3 10_cavity.cu

cpp_cavity: 10_cavity.cu
	g++ -O3 10_cavity.cpp

plot: 10_plot.py
	python3 10_plot.py

py_cavity: 10_cavity.py
	python3 10_cavity.py

clean:
	rm a.out p.dat u.dat v.dat
