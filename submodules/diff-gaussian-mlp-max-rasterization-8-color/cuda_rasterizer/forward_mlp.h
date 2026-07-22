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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "utils.h"

namespace FORWARD_MLP
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
					const float *means3D,
					const float *cano_means3D,
					const glm::vec3 *scales,
					const float scale_modifier,
					const glm::vec4 *rotations,
					const float *weight1,
					const float *bias1,
					const float *weight2,
					const float *bias2,
					const float *weight2_color,
					const float *bias2_color,
					const float *opacities,
					const float *shs,
					bool *clamped,
					const float *cov3D_precomp,
					const float *colors_precomp,
					const float *viewmatrix,
					const float *projmatrix,
					const glm::vec3 *cam_pos,
					const int W, int H,
					const float focal_x, float focal_y,
					const float tan_fovx, float tan_fovy,
					int *radii,
					float2 *means2D,
					float *depths,
					float *cov3Ds,
					float3x3 *inv_mat_modelview,
					float3 *points_xyz_view,
					float3 *cano_points_xyz_view,
					float *inv_max_radius,
					int *max_dim,
					float8x3 *out_weight1,
					float8 *out_bias1,
					float8 *out_weight2,
					float *out_bias2,
					float8x3 *out_weight2_color,
					float3 *out_bias2_color,
					float *rgb,
					float4 *conic_opacity,
					const dim3 grid,
					uint32_t *tiles_touched,
					bool prefiltered,
					bool antialiasing);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2 *ranges,
		const uint32_t *point_list,
		int W, int H,
		float fx, float fy,
		const float3 *points_xyz_view,
		const float3 *cano_points_xyz_view,
		const float *colors,
		const float8x3 *weight1,
		const float8 *bias1,
		const float8 *weight2,
		const float *bias2,
		const float8x3 *weight2_color,
		const float3 *bias2_color,
		const float3x3 *inv_mat_modelview,
		const float *viewmatrix,
		const float *inv_max_radius,
		const float4 *conic_opacity,
		float *final_T,
		uint32_t *n_contrib,
		const float *bg_color,
		float *out_color,
		float *depths,
		float *depth);
}

#endif
