#include "Halide.h"


using Halide::Generator;
using Halide::RVar;
using Halide::ConciseCasts::u8;
using Halide::ConciseCasts::u32;
using Halide::ConciseCasts::u8_sat;

class MatrixMultiply256 : public Generator<MatrixMultiply256> {
public:
    // Two signed 16-bit input matrices, indexed by x, y.
    GeneratorParam<int> matrix_size{"size", 256};
    Input<Buffer<int8_t>> A{ "A", 2 };
    Input<Buffer<int8_t>> B{ "B", 3 };

    Output<Buffer<int32_t>> res{ "res", 2 };

    Func mm{"mm"};
    void generate() {
        Var x("x"), y("y");
        Var rxi("rxi"), ryi("ryi");
        RVar rri("rri"), rro("rro");
        RDom r(0, matrix_size);
        mm(x, y) = cast<int32_t>(0);
        mm(x, y) += cast<int32_t>(A(r, y)) * cast<int32_t>(B(r % 4, x, r / 4));
        res = mm.in();
        int tile_x = 16, tile_y = 16, tile_r = 64;

        mm.compute_at(mm.in(), x)
        .store_in(MemoryType::AMXTile)
        .update()
        .tile(x, y, rxi, ryi, tile_x, tile_y, TailStrategy::GuardWithIf)
        .split(r, rro, rri, tile_r)
        .reorder(rri, rxi, ryi, rro, x, y)
        .atomic()
        .vectorize(rri)
        .vectorize(rxi)
        .vectorize(ryi);

        Var ixi("ixi"), iyi("iyi");
        mm.compute_at(mm.in(), x)
            .tile(x, y, ixi, iyi, tile_x, tile_y)
            .vectorize(ixi)
            .vectorize(iyi);

        // schedule the consumer
        Var mmxi("mmxi"), mmyi("mmyi");
        mm.in()
            .tile(x, y, mmxi, mmyi, tile_x, tile_y)
            .vectorize(mmxi)
            .vectorize(mmyi);

        mm.print_loop_nest();
    }   

    void schedule() {}

};

HALIDE_REGISTER_GENERATOR(MatrixMultiply256, batched_matmul_256_32bit)
