torch.setdefaulttensortype("torch.FloatTensor")

require 'matching'
require 'ardrone'
require 'image'

local K = torch.Tensor{{275, 0, 158},{0, 275, 89}, {0, 0, 1}}
local dist = torch.Tensor{-0.5215, 0.4426, 0.0012, 0.0071, -0.0826}

ardrone.initvideo()
function getframe()
   local fr = nil
   fr = ardrone.getframe(fr)
   --cfr = image.scale(fr, 640, 320)
   fr = fr:squeeze()
   fr = undistort(fr, K, dist)
   return fr/fr:max()
end

local im1 = getframe()
local i_img = 0
local depth = torch.Tensor(im1:size(1), im1:size(2)):zero()
local conf = torch.Tensor(im1:size()):zero()
local lambda = 0.5

while true do
   local im2 = getframe()
   win2 = image.display{win=win2, image=im2}
   local out, outErr, outDbg = opticalflow(im1, im2, K)
   
   conf:mul(lambda)
   local newconf = outErr--out:ne(0):float()
   local divconf = conf + newconf
   divconf:add(divconf:eq(0):float())
   depth:cmul(conf):add(out:cmul(newconf)):cdiv(divconf)
   conf:add(newconf)

   win = image.display{win=win, image=depth:cmul(conf:gt(0.5):float()), min=0, max=1}
   win3 = image.display{win=win3, image=outDbg}
   im1 = im2
   i_img = i_img + 1
end