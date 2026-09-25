#pragma once

// Multithreaded k-means (k-means++ seeding, Lloyd iterations) on dense float
// points under squared Euclidean distance.
//
// Points are fitted on a sample, then every class is assigned with nearest().
// Results are deterministic for a fixed seed and thread count.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "core/parallel.h"

namespace poker2::abstraction {

struct KMeansResult {
  int k = 0;
  int dim = 0;
  std::vector<float> centers;  // k * dim
  double inertia = 0.0;        // mean squared distance to the assigned center
  double total_variance = 0.0; // mean squared distance to the global mean
  int iterations = 0;
  int reseeded = 0;            // empty clusters refilled during fitting
};

inline float squared_distance(const float* a, const float* b, int dim) {
  float sum = 0.0f;
  for (int d = 0; d < dim; ++d) {
    const float diff = a[d] - b[d];
    sum += diff * diff;
  }
  return sum;
}

inline int nearest_center(const float* centers, int k, int dim, const float* point,
                          float* distance_out = nullptr) {
  int best = 0;
  float best_distance = std::numeric_limits<float>::max();
  for (int c = 0; c < k; ++c) {
    const float d = squared_distance(centers + static_cast<size_t>(c) * dim, point, dim);
    if (d < best_distance) {
      best_distance = d;
      best = c;
    }
  }
  if (distance_out) *distance_out = best_distance;
  return best;
}

struct KMeansOptions {
  int k = 8;
  int max_iterations = 50;
  double tolerance = 1e-4;  // stop when inertia improves by less than this fraction
  uint64_t seed = 1;
  int threads = 1;
};

// points: n * dim floats.
inline KMeansResult kmeans_fit(const std::vector<float>& points, int dim,
                               const KMeansOptions& opt) {
  const size_t n = points.size() / dim;
  const int k = static_cast<int>(std::min<size_t>(opt.k, n));
  KMeansResult result;
  result.k = k;
  result.dim = dim;
  result.centers.assign(static_cast<size_t>(k) * dim, 0.0f);
  if (n == 0 || k == 0) return result;
  const float* p = points.data();
  auto point = [&](size_t i) { return p + i * dim; };

  // Global mean and total variance, for reporting explained variance.
  {
    std::vector<double> mean(dim, 0.0);
    for (size_t i = 0; i < n; ++i)
      for (int d = 0; d < dim; ++d) mean[d] += point(i)[d];
    for (double& m : mean) m /= n;
    std::vector<float> meanf(mean.begin(), mean.end());
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) total += squared_distance(point(i), meanf.data(), dim);
    result.total_variance = total / n;
  }

  // k-means++ seeding.
  std::mt19937_64 rng(opt.seed);
  std::vector<float> dist(n);
  size_t first = rng() % n;
  std::copy(point(first), point(first) + dim, result.centers.begin());
  parallel_ranges(n, opt.threads, [&](uint64_t b, uint64_t e, int) {
    for (uint64_t i = b; i < e; ++i) dist[i] = squared_distance(point(i), result.centers.data(), dim);
  });
  for (int c = 1; c < k; ++c) {
    double total = 0.0;
    for (float d : dist) total += d;
    size_t chosen = rng() % n;
    if (total > 0.0) {
      double target = std::uniform_real_distribution<double>(0.0, total)(rng);
      for (size_t i = 0; i < n; ++i) {
        target -= dist[i];
        if (target <= 0.0) {
          chosen = i;
          break;
        }
      }
    }
    float* center = result.centers.data() + static_cast<size_t>(c) * dim;
    std::copy(point(chosen), point(chosen) + dim, center);
    parallel_ranges(n, opt.threads, [&](uint64_t b, uint64_t e, int) {
      for (uint64_t i = b; i < e; ++i) dist[i] = std::min(dist[i], squared_distance(point(i), center, dim));
    });
  }

  // Lloyd iterations with per-thread accumulators merged in thread order.
  const int threads = std::max(1, opt.threads);
  std::vector<int> assignment(n);
  std::vector<std::vector<double>> sums(threads, std::vector<double>(static_cast<size_t>(k) * dim));
  std::vector<std::vector<uint64_t>> counts(threads, std::vector<uint64_t>(k));
  std::vector<double> inertia_part(threads);
  double previous = std::numeric_limits<double>::max();
  for (int it = 0; it < opt.max_iterations; ++it) {
    parallel_ranges(n, threads, [&](uint64_t b, uint64_t e, int t) {
      std::fill(sums[t].begin(), sums[t].end(), 0.0);
      std::fill(counts[t].begin(), counts[t].end(), 0);
      double part = 0.0;
      for (uint64_t i = b; i < e; ++i) {
        float d;
        const int c = nearest_center(result.centers.data(), k, dim, point(i), &d);
        assignment[i] = c;
        dist[i] = d;
        part += d;
        ++counts[t][c];
        double* s = sums[t].data() + static_cast<size_t>(c) * dim;
        for (int j = 0; j < dim; ++j) s[j] += point(i)[j];
      }
      inertia_part[t] = part;
    });
    double inertia = 0.0;
    for (double part : inertia_part) inertia += part;
    inertia /= n;
    result.inertia = inertia;
    result.iterations = it + 1;

    std::vector<double> sum(static_cast<size_t>(k) * dim, 0.0);
    std::vector<uint64_t> count(k, 0);
    for (int t = 0; t < threads; ++t) {
      for (size_t j = 0; j < sum.size(); ++j) sum[j] += sums[t][j];
      for (int c = 0; c < k; ++c) count[c] += counts[t][c];
    }

    // Refill empty clusters with the points farthest from their centers.
    std::vector<size_t> far;
    for (int c = 0; c < k; ++c) {
      if (count[c] > 0) {
        for (int j = 0; j < dim; ++j) {
          result.centers[static_cast<size_t>(c) * dim + j] =
              static_cast<float>(sum[static_cast<size_t>(c) * dim + j] / count[c]);
        }
        continue;
      }
      if (far.empty()) {
        far.resize(n);
        for (size_t i = 0; i < n; ++i) far[i] = i;
        std::stable_sort(far.begin(), far.end(), [&](size_t a, size_t b) { return dist[a] > dist[b]; });
      }
      const size_t pick = far[result.reseeded % n];
      std::copy(point(pick), point(pick) + dim, result.centers.begin() + static_cast<size_t>(c) * dim);
      dist[pick] = 0.0f;
      ++result.reseeded;
    }

    if (previous - inertia <= opt.tolerance * previous) break;
    previous = inertia;
  }
  return result;
}

}  // namespace poker2::abstraction
