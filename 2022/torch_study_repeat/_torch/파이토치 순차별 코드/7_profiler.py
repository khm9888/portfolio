# PYTORCH 프로파일러(PROFILER)

# 1. 필요한 라이브러리들 불러오기

import torch
import torchvision.models as models
import torch.autograd.profiler as profiler

# 2. 간단한 ResNet 모델 인스턴스화 하기
# ResNet 모델 인스턴스를 만들고 입력값을 준비합니다 :

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

# 3. 프로파일러를 사용하여 실행시간 분석하기

# with profiler.profile(record_shapes=True) as prof:
#     with profiler.record_function("model_inference"):
#         model(inputs)

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
'''
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls                                   Input Shapes  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------  
                  model_inference         5.42%       3.899ms        99.98%      71.902ms      71.902ms             1                                             []  
                 aten::max_pool2d         0.02%      15.604us        16.94%      12.186ms      12.186ms             1        [[5, 64, 112, 112], [], [], [], [], []]  
    aten::max_pool2d_with_indices        16.90%      12.150ms        16.92%      12.170ms      12.170ms             1        [[5, 64, 112, 112], [], [], [], [], []]  
                     aten::conv2d         0.02%      11.696us        11.75%       8.451ms       8.451ms             1  [[5, 3, 224, 224], [64, 3, 7, 7], [], [], [],  
                aten::convolution         0.01%       8.688us        11.73%       8.439ms       8.439ms             1  [[5, 3, 224, 224], [64, 3, 7, 7], [], [], [],  
               aten::_convolution         0.03%      23.671us        11.72%       8.430ms       8.430ms             1  [[5, 3, 224, 224], [64, 3, 7, 7], [], [], [],  
         aten::mkldnn_convolution        11.67%       8.392ms        11.69%       8.407ms       8.407ms             1  [[5, 3, 224, 224], [64, 3, 7, 7], [], [], [],  
                     aten::conv2d         0.03%      20.597us        10.88%       7.825ms       1.956ms             4  [[5, 64, 56, 56], [64, 64, 3, 3], [], [], [],  
                aten::convolution         0.03%      19.389us        10.85%       7.805ms       1.951ms             4  [[5, 64, 56, 56], [64, 64, 3, 3], [], [], [],  
               aten::_convolution         0.06%      40.600us        10.83%       7.785ms       1.946ms             4  [[5, 64, 56, 56], [64, 64, 3, 3], [], [], [],  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------  
Self CPU time total: 71.916ms
'''
#4. 프로파일러를 사용하여 메모리 소비 분석하기

with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    model(inputs)

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

'''
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                      aten::empty         0.74%     428.729us         0.74%     428.729us       4.122us      94.79 Mb      94.79 Mb           104
                    aten::resize_         0.01%       8.165us         0.01%       8.165us       4.083us      11.48 Mb      11.48 Mb             2
                      aten::addmm         0.14%      81.437us         0.16%      90.909us      90.909us      19.53 Kb      19.53 Kb             1
                        aten::add         0.35%     202.590us         0.35%     202.590us      10.130us         160 b         160 b            20
              aten::empty_strided         0.01%       7.490us         0.01%       7.490us       7.490us           4 b           4 b             1
                     aten::conv2d         0.16%      93.489us        43.92%      25.394ms       1.270ms      47.37 Mb           0 b            20
                aten::convolution         0.16%      95.162us        43.76%      25.301ms       1.265ms      47.37 Mb           0 b            20
               aten::_convolution         0.28%     162.353us        43.59%      25.206ms       1.260ms      47.37 Mb           0 b            20
         aten::mkldnn_convolution        43.07%      24.906ms        43.31%      25.043ms       1.252ms      47.37 Mb           0 b            20
                aten::as_strided_         0.05%      29.089us         0.05%      29.089us       1.454us           0 b           0 b            20
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 57.822ms
'''

print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
'''

---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                      aten::empty         0.62%     438.848us         0.62%     438.848us       4.220us      94.79 Mb      94.79 Mb           104  
                 aten::batch_norm         0.11%      75.958us        28.14%      19.919ms     995.957us      47.41 Mb           0 b            20  
     aten::_batch_norm_impl_index         0.20%     138.664us        28.03%      19.843ms     992.159us      47.41 Mb           0 b            20  
          aten::native_batch_norm        21.16%      14.981ms        27.63%      19.561ms     978.061us      47.41 Mb           0 b            20  
                     aten::conv2d         0.12%      83.897us        50.38%      35.660ms       1.783ms      47.37 Mb           0 b            20  
                aten::convolution         0.13%      93.866us        50.26%      35.576ms       1.779ms      47.37 Mb           0 b            20  
               aten::_convolution         0.25%     173.661us        50.13%      35.482ms       1.774ms      47.37 Mb           0 b            20  
         aten::mkldnn_convolution        49.65%      35.148ms        49.88%      35.308ms       1.765ms      47.37 Mb           0 b            20  
                 aten::empty_like         0.08%      59.407us         0.22%     154.545us       7.727us      47.37 Mb           0 b            20  
                 aten::max_pool2d         0.03%      19.745us        18.39%      13.020ms      13.020ms      11.48 Mb           0 b             1  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
'''


with profiler.profile() as prof:
    with profiler.record_function("model_inference"):
        model(inputs)

prof.export_chrome_trace("trace.json")