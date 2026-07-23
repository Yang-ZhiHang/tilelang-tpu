#include "kernel.h"
#include <ppl_mem.h>
#include <cstdio>
#include <mutex>
#include <memory>
#include <sstream>
#include <vector>
#include <unistd.h>
#include <numeric>
#include <string>
#ifdef __bm1690__
#include <tpuv7_rt.h>
extern tpuRtStream_t stream;
extern tpuRtKernelModule_t tpu_module;
#elif defined(__bm1684x__)
#include <bmlib_runtime.h>
extern bm_handle_t handle;
extern tpu_kernel_module_t tpu_module;
#endif

#define MIN(x, y) (((x)) < ((y)) ? (x) : (y))
#define MAX(x, y) (((x)) > ((y)) ? (x) : (y))

int {function_name}_check_mem({func_params}) {{
  return 0;
}}

int {function_name}_check_mem_s(tpu_kernel_api_{function_name}_t *api) {{
  return 0;
}}

tpu_kernel_api_{function_name}_t fill_{function_name}_struct({func_params}) {{
  tpu_kernel_api_{function_name}_t api;
  {struct_assignments}
  return api;
}}

int {function_name}({func_params}) {{
  tpu_kernel_api_{function_name}_t api;
  int ret = 0;
  {struct_assignments}
#ifdef __bm1690__
  int core_num = 1;
  int group_num = 1;
  int block_num = 1;
  tpu_kernel_api_{function_name}_t apis[core_num];
  for (int i = 0; i < core_num; ++i) {{
    apis[i] = api;
  }}
  ret = tpuRtKernelLaunch(tpu_module, "{function_name}", &apis, sizeof(apis), group_num, block_num, stream);
  if (ret != 0) {{
      printf("tpu kernel launch failed!");
      return ret;
  }}
  ret = tpuRtStreamSynchronize(stream);
  if (ret != 0) {{
      printf("tpu stream synchronize failed!");
      return ret;
  }}
#elif defined(__bm1684x__)
  static int func_id = -1;
  if (func_id < 0) {{
    func_id = tpu_kernel_get_function(handle, tpu_module, "{function_name}");
    if (func_id < 0) {{
      printf("load kernel function failed!\n");
      return -2;
    }}
  }}
  ret = tpu_kernel_launch(handle, func_id, &api, sizeof(api));
  if (ret != 0) {{
      printf("tpu kernel launch failed!");
      return ret;
  }}
  ret = bm_thread_sync(handle);
  if (ret != 0) {{
      printf("tpu synchronize failed!");
      return ret;
  }}
#endif
  return 0;
}}
