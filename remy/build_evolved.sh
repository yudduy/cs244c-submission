#!/bin/bash
# Build evolved-runner without autotools/LibTorch — only needs protobuf.
set -e

export PATH="/opt/homebrew/bin:$PATH"
cd "$(dirname "$0")"

PROTO_CFLAGS=$(pkg-config --cflags protobuf)
PROTO_LIBS=$(pkg-config --libs protobuf)
BOOST_INC="-I/opt/homebrew/include"

# Step 1: Compile protobufs if needed
if [ ! -f protobufs/libremyprotos.a ]; then
    echo "=== Building protobuf library ==="
    cd protobufs
    protoc --cpp_out=. dna.proto answer.proto problem.proto simulationresults.proto
    clang++ -std=c++17 -O2 $PROTO_CFLAGS -c *.pb.cc
    ar rcs libremyprotos.a *.pb.o
    cd ..
fi

# Step 2: Compile core Remy sources
cd src
echo "=== Compiling core sources ==="
CORE_FILES=(memory.cc memoryrange.cc rat.cc whisker.cc whiskertree.cc receiver.cc random.cc
            configrange.cc simulationresults.cc aimd.cc fin.cc fintree.cc fish.cc
            evaluator.cc)

for f in "${CORE_FILES[@]}"; do
    echo "  $f"
    clang++ -std=c++17 -O2 -I../protobufs $PROTO_CFLAGS $BOOST_INC -c "$f" -o "${f%.cc}.o"
done

# Step 3: Compile evolved files
echo "=== Compiling evolved files ==="
EVOLVED_FILES=(evolvedrat.cc evolved-evaluator.cc evolved-runner.cc)

for f in "${EVOLVED_FILES[@]}"; do
    echo "  $f"
    clang++ -std=c++17 -O2 -I../protobufs $PROTO_CFLAGS $BOOST_INC -c "$f" -o "${f%.cc}.o"
done

# Step 4: Link
echo "=== Linking evolved-runner ==="
OBJ_FILES=(memory.o memoryrange.o rat.o whisker.o whiskertree.o receiver.o random.o
           configrange.o simulationresults.o aimd.o fin.o fintree.o fish.o
           evaluator.o evolvedrat.o evolved-evaluator.o evolved-runner.o)

clang++ -o evolved-runner "${OBJ_FILES[@]}" ../protobufs/libremyprotos.a $PROTO_LIBS -lm

echo "=== BUILD SUCCESS ==="
ls -la evolved-runner
echo ""
echo "Usage: ./src/evolved-runner link=0.946 rtt=100 on=1000 off=1000 nsrc=2"
