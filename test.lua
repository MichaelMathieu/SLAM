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
local xtarget, ytarget

while true do
   local timer = torch.Timer()
   local im2 = getframe()
   win2 = image.display{win=win2, image=im2}
   local out, outErr, outDbg, xtargetnew, ytargetnew = opticalflow(im1, im2, K)
   if xtargetnew then
      xtarget = xtargetnew
      ytarget = ytargetnew
   end

   local targetdir = 0
   if xtarget < 120 then
      targetdir = -0.2
   elseif xtarget > 200 then
      targetdir = 0.2
   end
   
   local todisp = torch.Tensor(3, out:size(1), out:size(2)):zero();
   todisp[1] = out:clone():cmul(out:gt(1e-5):float())
   todisp[1] = torch.ones(out:size()):cdiv(todisp[1]):cmul(out:ne(0):float());
   todisp[1]:mul(25)
   todisp[1]:add((-todisp[1]+1):cmul(todisp[1]:gt(1):float()))
   todisp[1]:cmul(outErr:gt(0.5):float())
   todisp[2] = -todisp[1]+1

   --win = image.display{win=win, image=depth:cmul(conf:gt(0.5):float()), min=0, max=1}
   win = image.display{win=win, image=todisp, min=0, max=1}
   win3 = image.display{win=win3, image=outDbg}
   win4 = image.display{win=win4, image=outErr, legend="conf"}
   im1 = im2
   i_img = i_img + 1
end