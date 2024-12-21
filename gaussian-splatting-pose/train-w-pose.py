#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.camera_utils import update_pose
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

from scipy.spatial.transform import Rotation as R
import numpy as np
from scene.colmap_loader import qvec2rotmat, rotmat2qvec

def save_camera_poses(scene, output_path):
    cameras = scene.getTrainCameras()

    # 按图像名称中的时间戳排序
    cameras.sort(key=lambda cam: float(cam.image_name[:-4]) if cam.image_name.endswith(".png") else float(cam.image_name))

    with open(output_path, "w") as f:
        for cam in cameras:
            # 获取相机名称并删除 .png 后缀
            image_name = cam.image_name[:-4] if cam.image_name.endswith(".png") else cam.image_name
            
                        # 确保 R 和 T 是 NumPy 数组
            if isinstance(cam.R, np.ndarray):
                R_wc = cam.R
            else:
                R_wc = cam.R.cpu().numpy() if cam.R.is_cuda else cam.R.numpy()
            
            if isinstance(cam.T, np.ndarray):
                T_wc = cam.T
            else:
                T_wc = cam.T.cpu().numpy() if cam.T.is_cuda else cam.T.numpy()

            quaternion = rotmat2qvec(R_wc)
            
            # save camera poses to a txt file
            f.write(f"{image_name} {T_wc[0]:.8f} {T_wc[1]:.8f} {T_wc[2]:.8f} {-quaternion[1]:.8f} {-quaternion[2]:.8f} {-quaternion[3]:.8f} {quaternion[0]:.8f}\n")

def downsample_image(image, num_levels=2):
    """
    使用高斯金字塔方法将图像的分辨率降低。
    image: 输入图像 (torch tensor, 形状为 [C, H, W])
    num_levels: 高斯金字塔的层数，这里我们只关心降采样两次
    """
    for _ in range(num_levels):
        # 使用 bilinear 插值进行下采样
        image = F.interpolate(image.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
    return image

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    save_camera_poses(scene,output_path=dataset.source_path+"/camera_poses_init.txt")
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_all = scene.getTrainCameras().copy()
    
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # 初始化实时绘图
    plt.ion()
    fig, ax = plt.subplots()
    loss_values = []
    global_iterations = []
    iteration_pose = 0
    global_iteration = 0  # 全局计数器

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
            
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # if pose_optimizer is not None:
        if viewpoint_cam.cam_trans_delta is not None:
            pose_optimizer = viewpoint_cam.training_setup()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if global_iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        (
            image,
            viewspace_point_tensor,
            visibility_filter,
            radii,
            depth,
            opacity,
            n_touched,
        ) = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["depth"],
            render_pkg["opacity"],
            render_pkg["n_touched"],
        )
        
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
    
        # 将图像分辨率降低
        # gt_image_downsampled = downsample_image(gt_image, num_levels=2)
        # image_downsampled = downsample_image(image, num_levels=2)
        
        gt_image_downsampled = gt_image
        image_downsampled = image        
        Ll1 = l1_loss(image_downsampled, gt_image_downsampled)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image_downsampled.unsqueeze(0), gt_image_downsampled.unsqueeze(0))
        else:
            ssim_value = ssim(image_downsampled, gt_image_downsampled)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()        
            
        with torch.no_grad():
            global_iterations.append(global_iteration)
            loss_values.append(loss.item())
            global_iteration += 1  # 增加全局计数
            # Densification
            if global_iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if global_iteration > opt.densify_from_iter and global_iteration % opt.densification_interval == 0:
                    size_threshold = 20 if global_iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if global_iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and global_iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    # iteration_pose = 0
                    
            # Optimizer step
            if global_iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

        if iteration < 2999 and viewpoint_cam.cam_trans_delta is not None:
            iteration_pose += 1
            # 遍历所有相机
            total_loss = 0
            # 选择附近的相机进行局部BA优化
            for i in range(5):
                if not (0 <= rand_idx + i < len(viewpoint_all)):
                    continue
                if not (0 <= rand_idx - i < len(viewpoint_all)):
                    continue
                
                viewpoint_cam= viewpoint_all[rand_idx+i]
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
                (image,_,_,_,_,_,_) = (render_pkg["render"],render_pkg["viewspace_points"],render_pkg["visibility_filter"],render_pkg["radii"],render_pkg["depth"],render_pkg["opacity"],render_pkg["n_touched"])
                
                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask.cuda()
                    image *= alpha_mask

                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                if FUSED_SSIM_AVAILABLE:
                    ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                else:
                    ssim_value = ssim(image, gt_image)
                    
                viewpoint_cam1= viewpoint_all[rand_idx-i]
                render_pkg1 = render(viewpoint_cam1, gaussians, pipe, bg)
                (image1,_,_,_,_,_,_)  = (render_pkg1["render"],render_pkg1["viewspace_points"],render_pkg1["visibility_filter"],render_pkg1["radii"],render_pkg1["depth"],render_pkg1["opacity"],render_pkg1["n_touched"])
                
                if viewpoint_cam1.alpha_mask is not None:
                    alpha_mask = viewpoint_cam1.alpha_mask.cuda()
                    image1 *= alpha_mask

                # Loss
                gt_image1 = viewpoint_cam1.original_image.cuda()
                Ll11 = l1_loss(image1, gt_image1)
                if FUSED_SSIM_AVAILABLE:
                    ssim_value1 = fused_ssim(image1.unsqueeze(0), gt_image1.unsqueeze(0))
                else:
                    ssim_value1 = ssim(image1, gt_image1)
                    
                total_loss = total_loss+(1.0 - opt.lambda_dssim) * Ll11 + opt.lambda_dssim * (1.0 - ssim_value1)+\
                (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
                    
            total_loss.backward()
            with torch.no_grad():
                pose_optimizer.step()
                # 更新窗口中相机的姿态
                viewpoint_cam= viewpoint_all[rand_idx]
                update_pose(viewpoint_cam)
                viewpoint_cam.camera_optimizer.zero_grad(set_to_none = True)

            # if converged:
            #     # clear the pose_optimizer never to be used again
            #     viewpoint_cam.clean()
                    
                                   
        if (iteration in checkpoint_iterations):
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if global_iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()    
                    
        # Log and save
        iter_end.record()  # 记录结束时间
        torch.cuda.synchronize()  # 确保所有CUDA操作完成
        training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset.train_test_exp)
        if (iteration in saving_iterations):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)        
            
    plt.figure()
    plt.plot(global_iterations, loss_values, label="Loss")
    plt.title("Loss Curve")
    plt.xlabel("Global Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve_pose.png")
    plt.close()
    
    # save all camera names and poses to a txt file, like image_name, tx ty tz qx qy qz qw
    save_camera_poses(scene,output_path=dataset.source_path+"/camera_poses_final.txt")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

import lpips
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 初始化 LPIPS 模型为全局静态变量，避免重复加载
    global lpips_loss_fn
    if 'lpips_loss_fn' not in globals():
        lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()
        lpips_loss_fn.eval()  # 确保模型处于推理模式，不参与梯度计算

    # 控制 LPIPS 计算频率
    if iteration % 50 != 0 and iteration not in testing_iterations:  # 每 50 次计算一次
        return

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                metrics = {'l1': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0}
                batch_images, batch_gt_images = [], []

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    # 如果开启了曝光训练/测试模式
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]

                    # 收集图像以便批处理
                    batch_images.append(image)
                    batch_gt_images.append(gt_image)

                    # 渲染的图像与 GT 图像日志
                    if tb_writer and idx < 5:
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)

                # 批量计算 LPIPS，减少逐帧调用的开销
                batch_images = torch.stack(batch_images, dim=0)
                batch_gt_images = torch.stack(batch_gt_images, dim=0)
                with torch.no_grad():
                    metrics['l1'] = l1_loss(batch_images, batch_gt_images).mean().item()
                    metrics['psnr'] = psnr(batch_images, batch_gt_images).mean().item()
                    metrics['ssim'] = ssim(batch_images, batch_gt_images).mean().item()
                    metrics['lpips'] = lpips_loss_fn(batch_images, batch_gt_images).mean().item()

                # 打印并记录日志
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {metrics['l1']}, PSNR {metrics['psnr']:.4f}, SSIM {metrics['ssim']:.4f}, LPIPS {metrics['lpips']:.4f}")
                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", metrics['l1'], iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", metrics['psnr'], iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - ssim", metrics['ssim'], iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - lpips", metrics['lpips'], iteration)

        # 可选：添加透明度和点数量的直方图
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000,30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000,30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
