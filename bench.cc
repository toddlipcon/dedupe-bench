#include <benchmark/benchmark.h>
#include <string>
#include <iostream>
#include <vector>
#include <array>
#include <string_view>
#include <unordered_map>
#include <nmmintrin.h>
#include <xmmintrin.h>

using namespace std;

struct TestData {
  string buf;
  vector<string_view> vals;
};

#pragma clang attribute push (__attribute__((target("sse4.2,avx"))), apply_to=function)

TestData Setup(benchmark::State& state) {
  const int kMaxSize = 10;
  const int kCardinality = 1000;
  const int kUniqueVals = state.range(0);

  string buf(kMaxSize * kUniqueVals, 'x');
  srand(1);
  vector<string_view> unique_vals;
  int off = 0;
  for (int i = 0; i < kUniqueVals; i++) {
    int size = (rand() % kMaxSize) + 1;
    unique_vals.emplace_back(&buf[off], size);
    off += size;
  }

  vector<string_view> vals;
  for (int i = 0; i < kCardinality; i++) {
    vals.emplace_back(unique_vals[rand() % kUniqueVals]);
  }
  return { move(buf), move(vals) };
}

static void NoDedup(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  auto ptrs = Setup(state);
  for (auto _ : state) {
    string out_buf;
    vector<int> offsets;
    out_buf.reserve(1024 * 1024);
    offsets.resize(ptrs.vals.size());
    auto* off_out = &offsets[0];
    for (auto v : ptrs.vals) {
      *off_out++ = out_buf.size();
      out_buf.append(v);
    }
    benchmark::DoNotOptimize(offsets);
    benchmark::DoNotOptimize(out_buf);
    state.counters["size"] = out_buf.size();
  }
}
// Register the function as a benchmark
BENCHMARK(NoDedup)->Range(8, 8<<10);

template<class MapType>
static void DedupWithMap(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  auto ptrs = Setup(state);
  for (auto _ : state) {
    MapType cache;
    cache.reserve(16);

    string out_buf;
    vector<int> offsets;
    out_buf.reserve(1024 * 1024);
    offsets.resize(ptrs.vals.size());
    auto* off_out = &offsets[0];

    for (auto v : ptrs.vals) {
      auto ins = cache.emplace(v.data(), -1);
      if (ins.second) {
        ins.first->second = out_buf.size();
        out_buf.append(v);
      }        
      *off_out++ = ins.first->second;
    }
    benchmark::DoNotOptimize(offsets);
    benchmark::DoNotOptimize(out_buf);
    state.counters["size"] = out_buf.size();
  }
}

static void DedupUnorderedMap(benchmark::State& state) {
  DedupWithMap<unordered_map<const char*, int>>(state);
}
BENCHMARK(DedupUnorderedMap)->Range(8, 8<<10);

struct Identity {
  auto operator ()(uintptr_t t) {
    return t;
  }
};
struct Crc32 {
  auto operator ()(uintptr_t t) {
    return _mm_crc32_u64(0, t);
  }
};

struct Crc32Shift {
  auto operator ()(uintptr_t t) {
    return _mm_crc32_u64(0, t) >> 16;
  }
};

struct FNV1A_PY {
  auto operator ()(uintptr_t t) {
    constexpr int seed = 0;
    const uint32_t PRIME = 591798841;
    uint64_t hash64 = (uint64_t)seed ^ UINT64_C(14695981039346656037);
    hash64 = (hash64 ^ t) * PRIME;
    uint32_t hash32 = (uint32_t)(hash64 ^ (hash64 >> 32));
    return hash32 ^ (hash32 >> 16);
  }
};

template<class Hash, size_t kNumSlots>
struct LossyArrayMap {
  array<const char*, kNumSlots> keys_ = {};
  array<int, kNumSlots> vals_;

  LossyArrayMap() {
    char* x = (char*)this;
    char* end = x + sizeof(*this);
    while (x < end) {
      _mm_clflush(x);
      x += 64;
    }
  }

  void reserve(int n) {
  }
  auto emplace(const char* p, int off) {
    Hash h;
    size_t slot = h(reinterpret_cast<uintptr_t>(p)) % kNumSlots;
    auto it = iterator{keys_[slot], vals_[slot]};
    bool need_insert = it->first != p;
    if (need_insert) {
      it->first = p;
    }
    return make_pair(it, need_insert);
  }

  struct iterator {
    const char*& first;
    int& second;
    iterator* operator->() {
      return this;
    }
  };
};

#define BMS(hash_size) \
  BENCHMARK_TEMPLATE(DedupWithMap, LossyArrayMap<Identity, hash_size>)->Range(2, 8<<10); \
  BENCHMARK_TEMPLATE(DedupWithMap, LossyArrayMap<Crc32, hash_size>)->Range(2, 8<<10); \
  BENCHMARK_TEMPLATE(DedupWithMap, LossyArrayMap<Crc32Shift, hash_size>)->Range(2, 8<<10); \
  BENCHMARK_TEMPLATE(DedupWithMap, LossyArrayMap<FNV1A_PY, hash_size>)->Range(2, 8<<10);

BMS(4);
BMS(8);
BMS(16);
BMS(32);
BMS(64);
BMS(128);
BMS(256);
BMS(512);
BMS(1024);

#pragma clang attribute pop
