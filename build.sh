g++ -I ../benchmark/include/ -O3 ./bench.cc  --std=c++17 -msse4.2 ../benchmark/build/src/libbenchmark*.a -g
