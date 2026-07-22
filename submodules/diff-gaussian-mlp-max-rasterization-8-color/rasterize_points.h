/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansMLPCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &cano_means3D,
	const torch::Tensor &colors,
	const torch::Tensor &opacity,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const torch::Tensor &weight1,
	const torch::Tensor &bias1,
	const torch::Tensor &weight2,
	const torch::Tensor &bias2,
	const torch::Tensor &weight2_color,  // NOUVEAU
    const torch::Tensor &bias2_color,
	const float scale_modifier,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansMLPBackwardCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &radii,
	const torch::Tensor &colors,
	const torch::Tensor &scales,
	const torch::Tensor &opacities,
	const torch::Tensor &rotations,
	const torch::Tensor &weight1,
	const torch::Tensor &bias1,
	const torch::Tensor &weight2,
	const torch::Tensor &bias2,
	const torch::Tensor &weight2_color,  // NOUVEAU
    const torch::Tensor &bias2_color,
	const float scale_modifier,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor &dL_dout_color,
	const torch::Tensor &dL_dout_invdepth,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	const torch::Tensor &geomBuffer,
	const int R,
	const torch::Tensor &binningBuffer,
	const torch::Tensor &imageBuffer,
	const bool antialiasing,
	const bool debug);

torch::Tensor markVisible(
	torch::Tensor &means3D,
	torch::Tensor &viewmatrix,
	torch::Tensor &projmatrix);
