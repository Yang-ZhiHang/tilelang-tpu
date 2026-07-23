#ifdef __bm1690__
#include <tpuv7_rt.h>
#elif defined(__bm1684x__)
#include "bmlib_runtime.h"
#include "tpu_defs.h"
#endif
#include "host_test_utils.h"
#include "kernel.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>

#ifdef __bm1690__
tpuRtStream_t stream;
tpuRtKernelModule_t tpu_module;
#elif defined(__bm1684x__)
bm_handle_t handle;
tpu_kernel_module_t tpu_module;
#endif

int init(){{
#ifdef __bm1690__
  tpuRtStatus_t ret;
  ret = tpuRtInit();
  if (ret != tpuRtSuccess) {{
    return -1;
  }}
  tpuRtSetDevice(14); // Set TPU ID
  tpuRtStreamCreate(&stream);
#elif defined(__bm1684x__)
  bm_status_t ret = BM_SUCCESS;
  ret = bm_dev_request(&handle, 0);
  if (ret != BM_SUCCESS)
    throw("bm_dev_request_failed");
  printf("bm_dev_request success\n");
#endif
  auto kernel_dir = getenv("PPL_KERNEL_PATH");
  if (!kernel_dir) {{
    printf("[ERROR] tpu launch failed: PPL_KERNEL_PATH doesn't exist\n");
    return -2;
  }}
#ifdef __bm1690__
  tpu_module = tpuRtKernelLoadModuleFile(kernel_dir, stream);
#elif defined(__bm1684x__)
  tpu_module = tpu_kernel_load_module_file(handle, kernel_dir);
#endif
  if (NULL == tpu_module) {{
    printf("tpuRtKernelLoadModuleFile failed\n");
    return -2;
  }}
  return 0;
}}

void post(){{
#ifdef __bm1690__
  tpuRtKernelUnloadModule(tpu_module, stream);
  tpuRtStreamDestroy(stream);
#elif defined(__bm1684x__)
  tpu_kernel_free_module(handle, tpu_module);
  bm_dev_free(handle);
#endif
}}

extern "C" int tilelang_tpu_run(void** args) {{
{arg_declarations}

  int res = init();
  if(res != 0){{
    return res;
  }}

  // 设备指针声明
{device_declarations}

  // 分配设备内存
{malloc_statements}

  // 拷贝数据到设备
{memcpy_s2d_statements}

  // 调用内核函数

  auto start = std::chrono::high_resolution_clock::now();  // 开始计时

{kernel_call}

  auto end = std::chrono::high_resolution_clock::now();    // 结束计时

  if (rst) {{
    printf("kernel_launch failed\n");
    return 1;
  }}
  printf("kernel_launch success\n");

  // 计算执行时间
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double elapsed_time_ms = duration.count() / 1000.0;
  double elapsed_time_us = duration.count();

  printf("Single kernel execution time: %.3f ms (%.0f us)\n", elapsed_time_ms, elapsed_time_us);

  // 性能测试
  const int warmup_runs = 5;
  const int measure_runs = 10;

  printf("\n=== Performance Benchmark (after %d warmup runs) ===\n", warmup_runs);

  // 预热运行
  for (int i = 0; i < warmup_runs; i++) {{
    {pure_kernel_call}
  }}

  printf("Runs: %d\n", measure_runs);

  // 测量运行
  double total_time_us = 0.0;
  double min_time_us = std::numeric_limits<double>::max();
  double max_time_us = 0.0;

  for (int i = 0; i < measure_runs; i++) {{
    auto run_start = std::chrono::high_resolution_clock::now();
    {pure_kernel_call}
    auto run_end = std::chrono::high_resolution_clock::now();
    
    auto run_duration = std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_start);
    double run_time_us = run_duration.count();
    
    total_time_us += run_time_us;
    min_time_us = std::min(min_time_us, run_time_us);
    max_time_us = std::max(max_time_us, run_time_us);
  }}

  double avg_time_us = total_time_us / measure_runs;
  double avg_time_ms = avg_time_us / 1000.0;


  printf("Average execution time: %.3f ms (%.0f us)\n", avg_time_ms, avg_time_us);
  printf("Minimum execution time: %.3f ms (%.0f us)\n", min_time_us / 1000.0, min_time_us);
  printf("Maximum execution time: %.3f ms (%.0f us)\n", max_time_us / 1000.0, max_time_us);
  
  // 拷贝输出数据回主机
{memcpy_d2s_statements}
#ifdef __bm1690__
  tpuRtStreamSynchronize(stream);
#elif defined(__bm1684x__)
  bm_thread_sync(handle);
#endif

  // 释放设备内存
{free_statements}

  post();
  return 0;
}}
