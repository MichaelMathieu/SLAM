require 'libmatching'
require 'math'
require 'nn'

function opticalflow(im1, im2, K)
   output = torch.Tensor(im1:size(1), im1:size(2)):zero()
   outputErr = torch.Tensor(im1:size(1), im1:size(2)):zero()
   outputDebug = torch.Tensor(3, im1:size(1), im1:size(2)):zero()
   im1 = im1:contiguous()
   im2 = im2:contiguous()
   libmatching.opticalflow(im1, im2, outputDebug, output, outputErr, K)
   return output, outputErr, outputDebug
end

function undistort(im, K, dist)
   output = torch.Tensor(im:size())
   libmatching.undistort(im, output, K, dist)
   return output
end