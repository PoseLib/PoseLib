#ifndef POSELIB_ROBUST_SAMPLING_H_
#define POSELIB_ROBUST_SAMPLING_H_

#include "types.h"

namespace pose_lib {

typedef uint64_t RNG_t;
int random_int(RNG_t &state);


// Draws a random sample (NOTE: This assumes sample_sz >= N but does not check it!)
void draw_sample(size_t sample_sz, size_t N, std::vector<size_t> *sample, RNG_t &rng);

// Sampling for multi-camera systems
void draw_sample(size_t sample_sz, const std::vector<size_t> &N, std::vector<std::pair<size_t, size_t>> *sample, RNG_t &rng);



class RandomSampler {
public:
    RandomSampler(size_t N, size_t K, RNG_t seed = 0, bool use_prosac_sampling = false, int prosac_iters = 100000)
         : num_data(N), sample_sz(K), state(seed), use_prosac(use_prosac_sampling), max_prosac_iterations(prosac_iters) {
        if(use_prosac_sampling) {
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
    int max_prosac_iterations; // number of iterations before reverting to standard RANSAC
    int sample_k;
    int subset_sz;
    // pre-computed growth function used in PROSAC sampling
    std::vector<int> growth;
};

}
#endif