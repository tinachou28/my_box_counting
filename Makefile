all: Py_Box_Count_Stats

Py_Box_Count_Stats: Py_Box_Count_Stats.cpp
    g++-13 -O3 -shared Py_Box_Count_Stats.cpp -o Py_Box_Count_Stats.so -std=c++14 -undefined dynamic_lookup -fPIC -I/usr/local/include/eigen3 -I/usr/local/include -L/usr/local/lib -lfftw3 `python3.10 -m pybind11 --includes`
clean:
    rm *.so