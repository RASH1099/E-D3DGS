import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, cam_no=None, iter=None, train_coarse=False, \
    num_down_emb_c=5, num_down_emb_f=5):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建一个张量，计算2D屏幕坐标 形状与pc.get_xyz相同
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz 
    # if cam_type != "PanopticSports":
    # 获取 相机的水平视角的半正切值 和 垂直视角的半正切值 用来计算投影矩阵
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # 高斯体素光栅化设置（传到可微光栅化渲染器里面）
    raster_settings = GaussianRasterizationSettings(
        image_height=torch.tensor(viewpoint_camera.image_height).cuda(), # 图像高度
        image_width=torch.tensor(viewpoint_camera.image_width).cuda(), # 图像宽度
        tanfovx=torch.tensor(tanfovx).cuda(), # 
        tanfovy=torch.tensor(tanfovy).cuda(),
        bg=bg_color.cuda(), # 背景颜色
        scale_modifier=torch.tensor(scaling_modifier).cuda(),# 缩放因子
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),# 视图矩阵（世界到相机）
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),# 投影矩阵（世界到像素）
        sh_degree=torch.tensor(pc.active_sh_degree).cuda(),# 已激活的球谐函数阶数
        campos=viewpoint_camera.camera_center.cuda(),# 相机位置
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False
    )
    # 把时间数字变成张量，然后放在和高斯坐标一样的设备上，这个时间复制 N 行 1 列，变成 (N,1)，让每个点都有一个对应的时间输入
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
  
    # else:
    #     raster_settings = viewpoint_camera['camera']
    #     time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        
    # 光栅化实例
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    # 2D屏幕坐标（高斯体素投影到图像平面上的坐标）
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 是否提前计算3D协方差矩阵
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:# false不执行
        # 预计算3D协方差矩阵
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation

    means3D_final, scales_final, rotations_final, opacity_final, shs_final, extras = pc._deformation(means3D, scales, 
        rotations, opacity, time, cam_no, pc, None, shs, iter=iter, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_f)

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    # 是否使用预计算颜色（使用球谐函数系数去计算rgb）
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    depth = None
    # 光栅化渲染图片
    outputs = rasterizer(
        means3D = means3D_final,# 3D 坐标
        means2D = means2D,# 2D 屏幕坐标
        shs = shs_final,# 球谐函数系数
        colors_precomp = colors_precomp,# 预计算颜色值
        opacities = opacity,# 不透明度
        scales = scales_final,# 尺度缩放因子
        rotations = rotations_final, # 旋转矩阵
        cov3D_precomp = cov3D_precomp)# 预计算三维协方差矩阵
    if len(outputs) == 2:
        rendered_image, radii = outputs
    elif len(outputs) == 3:
        rendered_image, radii, depth = outputs
    else:
        assert False, "only (depth-)diff-gaussian-rasterization supported!"
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.


    return {"render": rendered_image, # 渲染后的图像
            "viewspace_points": screenspace_points,#  高斯在屏幕空间的坐标
            "visibility_filter" : radii > 0,
            "radii": radii,# 每个高斯体素在图像平面上的半径
            "depth":depth, # 深度图
            "sh_coefs_final": shs_final,# 高斯球谐函数系数
            "extras":extras,}