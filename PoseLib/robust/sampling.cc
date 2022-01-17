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

#include "sampling.h"

#include <cmath>

namespace poselib {

// Splitmix64 PRNG
typedef uint64_t RNG_t;
int random_int(RNG_t &state) {
    state += 0x9e3779b97f4a7c15;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// Draws a random sample
void draw_sample(size_t sample_sz, size_t N, std::vector<size_t> *sample, RNG_t &rng) {
    for (size_t i = 0; i < sample_sz; ++i) {
        bool done = false;
        while (!done) {
            (*sample)[i] = random_int(rng) % N;

            done = true;
            for (size_t j = 0; j < i; ++j) {
                if ((*sample)[i] == (*sample)[j]) {
                    done = false;
                    break;
                }
            }
        }
    }
}
// Sampling for multi-camera systems
void draw_sample(size_t sample_sz, const std::vector<size_t> &N, std::vector<std::pair<size_t, size_t>> *sample,
                 RNG_t &rng) {
    for (size_t i = 0; i < sample_sz; ++i) {
        bool done = false;
        while (!done) {
            (*sample)[i].first = random_int(rng) % N.size();
            if (N[(*sample)[i].first] == 0) {
                continue;
            }
            (*sample)[i].second = random_int(rng) % N[(*sample)[i].first];

            done = true;
            for (size_t j = 0; j < i; ++j) {
                if ((*sample)[i] == (*sample)[j]) {
                    done = false;
                    break;
                }
            }
        }
    }
}

void RandomSampler::generate_sample(std::vector<size_t> *sample) {
    if (use_prosac && sample_k < max_prosac_iterations) {
        draw_sample(sample_sz - 1, subset_sz - 1, sample, state);
        (*sample)[sample_sz - 1] = subset_sz - 1;

        // update prosac state
        sample_k++;
        if (sample_k < max_prosac_iterations) {
            if (sample_k > growth[subset_sz - 1]) {
                if (++subset_sz > num_data) {
                    subset_sz = num_data;
                }
            }
        }
    } else {
        // uniform ransac sampling
        draw_sample(sample_sz, num_data, sample, state);
    }
}

void RandomSampler::initialize_prosac() {
    growth.resize(std::max(num_data, sample_sz), 0);

    // In the paper, T_N = max_prosac_iterations

    // Initialize T_n for n = sample_sz
    double T_n = max_prosac_iterations;
    for (size_t i = 0; i < sample_sz; ++i)
        T_n *= static_cast<double>(sample_sz - i) / (num_data - i);

    // Note that that growth[] stores T_n prime
    // The growth function is then defined as
    // g(t) = smallest n such that T_n prime > t
    for (size_t n = 0; n < sample_sz; ++n) {
        growth[n] = 1;
    }

    size_t T_np = 1;
    for (size_t n = sample_sz; n < num_data; ++n) {
        // Recursive relation from eq. 3
        double T_n_next = T_n * (n + 1.0) / (n + 1.0 - sample_sz);

        // Eq. 4
        T_np += std::ceil(T_n_next - T_n);
        growth[n] = T_np;
        T_n = T_n_next;
    }

    // counter keeping track of which sample we are at
    sample_k = 1;
    subset_sz = sample_sz;
}

} // namespace poselib