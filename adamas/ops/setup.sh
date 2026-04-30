set -euo pipefail

mkdir -p build

if [ ! -f build/build.ninja ]; then
    cmake -S . -B build \
        -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
        -GNinja
fi

cmake --build build

echo "Compilation Finish"
for file in $(find "./build" -maxdepth 1 -name "*.so"); do
    abs_file=$(realpath $file)
    if [ -e $abs_file ]; then
        ln -sfn $abs_file ../$(basename $file)
        echo "Linked $abs_file..."
    fi
done
