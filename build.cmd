set path=C:\Program Files\CMake\bin;%PATH%
set CAFFE_DEPENDENCIES=D:/users/v-guoxie/caffe2/libraries
call "%VS120COMNTOOLS%..\..\VC\vcvarsall.bat" amd64
set CMAKE_GENERATOR=Ninja
set CMAKE_CONFIGURATION=Release
mkdir build2
cd build2
cmake -G%CMAKE_GENERATOR% -DBLAS=Open -DCMAKE_BUILD_TYPE=%CMAKE_CONFIGURATION% -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=install -DCUDNN_ROOT=D:/users/v-guoxie/caffe2/3rdparty/cuda -C %CAFFE_DEPENDENCIES%\caffe-builder-config.cmake  ..\
cmake --build . --config %CMAKE_CONFIGURATION%
cmake --build . --config %CMAKE_CONFIGURATION% --target install
cd ../