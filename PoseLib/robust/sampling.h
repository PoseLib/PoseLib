#ifndef POSELIB_ROBUST_SAMPLING_H_
#define POSELIB_ROBUST_SAMPLING_H_

// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "PoseLib/types.h"

#include <cstdint>
#include <vector>

namespace poselib {

typedef uint64_t RNG_t;
int random_int(RNG_t &state);

// Draws a random sample (NOTE: This assumes sample_sz >= N but does not check it!)
void draw_sample(size_t sample_sz, size_t N, std::vector<size_t> *sample, RNG_t &rng);

// Sampling for multi-camera systems
void draw_sample(size_t sample_sz, const std::vector<size_t> &N, std::vector<std::pair<size_t, size_t>> *sample,
                 RNG_t &rng);

class RandomSampler {
  public:
    RandomSampler(size_t N, size_t K, RNG_t seed = 0, bool use_prosac_sampling = false, int prosac_iters = 100000)
        : num_data(N), sample_sz(K), state(seed), use_prosac(use_prosac_sampling), max_prosac_iterations(prosac_iters) {
        if (use_prosac_sampling) {
            initialize_prosac();
        }
    }

    void generate_sample(std::vector<size_t> *sample);

  private:
    void initialize_prosac();

  public:
    size_t num_data;
    size_t sample_sz;
    RNG_t state;

    // state for PROSAC sampling
    bool use_prosac;
    size_t max_prosac_iterations; // number of iterations before reverting to standard RANSAC
    size_t sample_k;
    size_t subset_sz;
    // pre-computed growth function used in PROSAC sampling
    std::vector<size_t> growth;
};

} // namespace poselib
#endif