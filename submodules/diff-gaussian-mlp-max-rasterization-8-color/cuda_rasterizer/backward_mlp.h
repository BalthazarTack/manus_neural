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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "utils.h"

namespace BACKWARD_MLP
{
	// void render(
	// 	const dim3 grid, dim3 block,
	// 	const uint2 *ranges,
	// 	const uint32_t *point_list,
	// 	int W, int H,
	// 	const float *bg_color,
	// 	const float2 *means2D,
	// 	const float4 *conic_opacity,
	// 	const float *colors,
	// 	const float *depths,
	// 	const float *final_Ts,
	// 	const uint32_t *n_contrib,
	// 	const float *dL_dpixels,
	// 	const float *dL_invdepths,
	// 	float3 *dL_dmean2D,
	// 	float4 *dL_dconic2D,
	// 	float *dL_dopacity,
	// 	float *dL_dcolors,
	// 	float *dL_dinvdepths);

	void renderMLP(
		const dim3 grid,
		dim3 block,
		const uint2 *ranges,
		const uint32_t *point_list,
		int W, int H,
		float fx, float fy,
		const float *bg_color,
		const float *viewmatrix,
		const float3 *means3D_view,
		const float3 *cano_means3D_view,
		const float *inv_max_radius,
		const float8x3 *weight1,
		const float8 *bias1,
		const float8 *weight2,
		const float *bias2, 
		const float8x3 *weight2_color,    
        const float3 *bias2_color,
		const float3x3 *inv_mat_modelview,
		const float *depths,
		const float *final_Ts,
		const uint32_t *n_contrib,
		const float *dL_dpixels,
		const float *dL_invdepths,
		float3 *dL_dmeans3D_view,
		float3 *dL_dcano_means3D_view,
		float *dL_dinv_max_radius,
		float8x3 *dL_dweight1,
		float8 *dL_dbias1,
		float8 *dL_dweight2,
		float *dL_dbias2,
		float8x3 *dL_dweight2_color,
		float3 *dL_dbias2_color,
		float3x3 *dL_dinv_mat_modelview,
		float *dL_dinvdepths);

	void preprocess(
		int P, int D, int M,
		const float3 *means_world,
		const int *radii,
		const float *shs,
		const bool *clamped,
		const float *opacities,
		const glm::vec3 *scales,
		const glm::vec4 *rotations,
		const float scale_modifier,
		const float *view,
		const float *proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3 *campos,
		const float3 *dL_dpoints_xyz_view,
		const float *dL_dinv_max_radius,
		const int *max_dim,
		const float *dL_dcolor,
		const float3x3 *dL_dinv_mat_modelview,
		const float *dL_dinvdepth,
		float *dL_dopacity,
		glm::vec3 *dL_dmeans_world,
		float *dL_dsh,
		glm::vec3 *dL_dscale,
		glm::vec4 *dL_drot,
		bool antialiasing);
}

#endif
