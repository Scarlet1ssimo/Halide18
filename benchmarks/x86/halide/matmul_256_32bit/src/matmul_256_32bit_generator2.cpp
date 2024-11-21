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
    Input<Buffer<int16_t>> A{ "A", 2 };
    Input<Buffer<int16_t>> B{ "B", 2 };

    Output<Buffer<int32_t>> res{ "res", 2 };

    void generate() {

        RDom k(0, matrix_size);
        mm(x,y) = 0;
        mm(x, y) += (cast<int32_t>(A(k, y)) * cast<int32_t>(B(x,k)));
        res(x, y) = mm(x, y);

        RVar red_dim(mm.update(0).get_schedule().dims()[0].var);

        res
            .compute_root()
            .split(y, y, yi, 4, TailStrategy::ShiftInwards)
            .split(x, x, xi, 64, TailStrategy::ShiftInwards)
            .split(xi, xi, xii, 16, TailStrategy::ShiftInwards)
            .vectorize(xii, 16)
            .reorder({xii, xi, yi, x, y})
            .unroll(xi)
            .unroll(yi);
            //.parallel(y);
        mm.update(0)
            .split(x, x, xi, 16, TailStrategy::GuardWithIf)
            .vectorize(xi, 16)
            .reorder({xi, x, y, red_dim})
            .unroll(x)
            .unroll(y);
        mm
            .store_in(MemoryType::Stack)
            .compute_at(res, x)
            .split(x, x, xi, 16, TailStrategy::RoundUp)
            .vectorize(xi, 16)
            .unroll(x)
            .unroll(y);

        res.print_loop_nest();
    }   

    void schedule() {}

private:
    Func mm{"mm"};
    Var x{ "x" }, y{ "y" }, yi{"yi"}, xi{"xi"}, yii{"yii"}, xii{"xii"}, yiii{"yiii"}, xiii{"xiii"};
};

HALIDE_REGISTER_GENERATOR(MatrixMultiply256, matmul_256_32bit)
