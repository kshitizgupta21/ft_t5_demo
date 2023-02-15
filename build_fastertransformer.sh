git clone https://github.com/NVIDIA/FasterTransformer.git
cd FasterTransformer
git submodule init && git submodule update
mkdir build
cd build 

cmake -DSM=86 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
make -j${nproc}
