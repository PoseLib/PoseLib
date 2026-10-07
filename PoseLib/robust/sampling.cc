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

#include <algorithm>
#include <cmath>
#include <limits>

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

size_t group_camera_centers(const std::vector<Point3D> &camera_centers, std::vector<size_t> *center_group) {
    const size_t num_cams = camera_centers.size();
    center_group->resize(num_cams);
    if (num_cams == 0) {
        return 0;
    }

    // Centers which differ by less than this fraction of the extent of the rig, i.e. of the
    // largest distance between two of its centers, are treated as coinciding. A rig which only
    // rotates about one center has no extent, and its centers then agree only up to the
    // rounding of -R' * t, so the tolerance is floored relative to the magnitude of the centers:
    // well above that rounding, and far below any baseline the scale could be observed from.
    double extent = 0.0;
    double magnitude = 0.0;
    for (size_t k = 0; k < num_cams; ++k) {
        magnitude = std::max(magnitude, camera_centers[k].norm());
        for (size_t j = 0; j < k; ++j) {
            extent = std::max(extent, (camera_centers[k] - camera_centers[j]).norm());
        }
    }
    const double tol = std::max(1e-6 * extent, 1e3 * std::numeric_limits<double>::epsilon() * magnitude);
    const double sq_tol = tol * tol;

    size_t num_groups = 0;
    for (size_t k = 0; k < num_cams; ++k) {
        (*center_group)[k] = num_groups;
        for (size_t j = 0; j < k; ++j) {
            if ((camera_centers[k] - camera_centers[j]).squaredNorm() <= sq_tol) {
                (*center_group)[k] = (*center_group)[j];
                break;
            }
        }
        if ((*center_group)[k] == num_groups) {
            num_groups++;
        }
    }
    return num_groups;
}

// Sampling for multi-camera systems where the sample has to span at least two camera centers
void draw_sample_distinct_centers(size_t sample_sz, const std::vector<size_t> &N,
                                  const std::vector<size_t> &center_group,
                                  std::vector<std::pair<size_t, size_t>> *sample, RNG_t &rng) {
    draw_sample(sample_sz, N, sample, rng);

    const size_t group = center_group[(*sample)[0].first];
    for (size_t i = 1; i < sample_sz; ++i) {
        if (center_group[(*sample)[i].first] != group) {
            return;
        }
    }

    // The sample degenerated to a single center, so we redraw its last element among the
    // cameras which do not share that center, uniformly over those cameras and then over
    // their observations, as draw_sample does. This cannot collide with the other elements
    // of the sample since it comes from a different camera.
    size_t num_eligible = 0;
    for (size_t k = 0; k < N.size(); ++k) {
        if (N[k] > 0 && center_group[k] != group) {
            num_eligible++;
        }
    }
    if (num_eligible == 0) {
        // No second center holds observations, so no sample can constrain the scale. We
        // leave the degenerate sample in place and let the caller reject the models it
        // generates; GeneralizedAbsolutePoseScaleEstimator rules this case out up front.
        return;
    }

    size_t pick = random_int(rng) % num_eligible;
    for (size_t k = 0; k < N.size(); ++k) {
        if (N[k] == 0 || center_group[k] == group) {
            continue;
        }
        if (pick-- == 0) {
            (*sample)[sample_sz - 1] = {k, random_int(rng) % N[k]};
            return;
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

void HybridSampler::generate_sample(std::vector<size_t> *sample_p3p, std::vector<size_t> *pairs_5p1pt,
                                    std::vector<size_t> *sample_5p1pt) {
    // Sample for P3P
    if (num_data_p3p >= sample_sz_p3p) {
        draw_sample(sample_sz_p3p, num_data_p3p, sample_p3p, state);
    } else {
        sample_p3p->clear();
    }

    // Sample for 5p1pt
    pairs_5p1pt->clear();
    sample_5p1pt->clear();
    // num_data_5p1pt is cleared in hybrid_pose.h in case there is not enough data
    if (num_data_5p1pt.size()) {
        size_t pair0, pair1;
        bool done = false;
        while (!done) {
            pair0 = random_int(state) % num_data_5p1pt.size();
            if (num_data_5p1pt[pair0] < 5)
                continue;

            pair1 = random_int(state) % num_data_5p1pt.size();
            if (pair0 == pair1 || num_data_5p1pt[pair1] == 0)
                continue;

            done = true;
        }
        pairs_5p1pt->push_back(pair0);
        pairs_5p1pt->push_back(pair1);

        // 6 samples - first 5 from pair0, last from pair1
        sample_5p1pt->resize(5);
        draw_sample(5, num_data_5p1pt[pair0], sample_5p1pt, state);
        size_t ind = random_int(state) % num_data_5p1pt[pair1];
        sample_5p1pt->push_back(ind);
    }
}

} // namespace poselib