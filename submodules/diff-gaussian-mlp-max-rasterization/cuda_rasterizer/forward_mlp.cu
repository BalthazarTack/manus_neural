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

#include "forward_mlp.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "mlp_kernels.cu"

namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 *means, glm::vec3 campos, const float *shs, bool *clamped)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
					 SH_C2[0] * xy * sh[4] +
					 SH_C2[1] * yz * sh[5] +
					 SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
					 SH_C2[3] * xz * sh[7] +
					 SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
						 SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
						 SH_C3[1] * xy * z * sh[10] +
						 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
						 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
						 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
						 SH_C3[5] * z * (xx - yy) * sh[14] +
						 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3 &mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float *cov3D, const float *viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float *cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// printf("computeCov3D, scale_x = %.15f, scale_y = %.15f, scale_z = %.15f\n", scale.x, scale.y, scale.z);

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot; // / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	// glm::mat3 R is implicitly tranpose of the actual R
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),	// 1st column
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),	// 2nd column
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)); // 3rd column

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// S_model^-1 * R_model^-1 * R_view^-1
__device__ float3x3 computeInvMatModelView(const glm::vec3 scale, float mod, const glm::vec4 rot, const float *viewmatrix)
{
	/*
		// Create inverse scaling matrix
		glm::mat3 inv_scale_model = glm::mat3(1.0f);
		inv_scale_model[0][0] = 1 / (mod * scale.x);
		inv_scale_model[1][1] = 1 / (mod * scale.y);
		inv_scale_model[2][2] = 1 / (mod * scale.z);

		// Normalize quaternion to get valid rotation
		glm::vec4 q = rot; // / glm::length(rot);
		float r = q.x;
		float x = q.y;
		float y = q.z;
		float z = q.w;

		// Compute rotation matrix from quaternion
		glm::mat3 rot_model = glm::mat3(
			1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),	// 1st row
			2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),	// 2nd row
			2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)); // 3rd row

		glm::mat3 inv_rot_model = glm::transpose(rot_model);

		glm::mat3 rot_view = glm::mat3(
			viewmatrix[0], viewmatrix[4], viewmatrix[8],
			viewmatrix[1], viewmatrix[5], viewmatrix[9],
			viewmatrix[2], viewmatrix[6], viewmatrix[10]);

		glm::mat3 inv_rot_view = glm::transpose(rot_view);

		glm::mat3 inv_mat_modelview = inv_scale_model * inv_rot_model * inv_rot_view;

		// printf("==========================\n");
		// printf("computeInvMatModelView\n");
		// printf("<---------------------------->\n");
		// printf("inv_scale_model:\n %f %f %f\n %f %f %f\n %f %f %f \n",
		// 	   inv_scale_model[0][0], inv_scale_model[0][1], inv_scale_model[0][2],
		// 	   inv_scale_model[1][0], inv_scale_model[1][1], inv_scale_model[1][2],
		// 	   inv_scale_model[2][0], inv_scale_model[2][1], inv_scale_model[2][2]);

		// printf("inv_rot_model:\n %f %f %f\n %f %f %f\n %f %f %f \n",
		// 	   inv_rot_model[0][0], inv_rot_model[0][1], inv_rot_model[0][2],
		// 	   inv_rot_model[1][0], inv_rot_model[1][1], inv_rot_model[1][2],
		// 	   inv_rot_model[2][0], inv_rot_model[2][1], inv_rot_model[2][2]);

		// printf("inv_rot_view:\n %f %f %f\n %f %f %f\n %f %f %f \n",
		// 	   inv_rot_view[0][0], inv_rot_view[0][1], inv_rot_view[0][2],
		// 	   inv_rot_view[1][0], inv_rot_view[1][1], inv_rot_view[1][2],
		// 	   inv_rot_view[2][0], inv_rot_view[2][1], inv_rot_view[2][2]);

		// printf("inv_mat_modelview:\n %f %f %f\n %f %f %f\n %f %f %f \n",
		// 	   inv_mat_modelview[0][0], inv_mat_modelview[0][1], inv_mat_modelview[0][2],
		// 	   inv_mat_modelview[1][0], inv_mat_modelview[1][1], inv_mat_modelview[1][2],
		// 	   inv_mat_modelview[2][0], inv_mat_modelview[2][1], inv_mat_modelview[2][2]);

		// printf("==========================\n");

		return make_float3x3(
			inv_mat_modelview[0][0], inv_mat_modelview[0][1], inv_mat_modelview[0][2],
			inv_mat_modelview[1][0], inv_mat_modelview[1][1], inv_mat_modelview[1][2],
			inv_mat_modelview[2][0], inv_mat_modelview[2][1], inv_mat_modelview[2][2]);
	*/

	// Create inverse scaling matrix
	glm::mat3 inv_scale_model = glm::mat3(1.0f);
	inv_scale_model[0][0] = 1 / (mod * scale.x);
	inv_scale_model[1][1] = 1 / (mod * scale.y);
	inv_scale_model[2][2] = 1 / (mod * scale.z);

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot; // / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute inverse rotation matrix from quaternion
	glm::mat3 inv_rot_model = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),	// 1st column
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),	// 2nd column
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)); // 3rd column

	// viewmatrix is a 1D vector of 16 elements, stored in column order

	// a11, a12, a13, a14,
	// a21, a22, a23, a24,
	// a31, a32, a33, a34,
	// 0, 0, 0, 1,

	// viewmatrix: a11, a21, a31, 0, a12, a22, a32, 0, a13, a23, a33, 0, a14, a24, a34, 1

	// glm::mat3 rot_view = glm::mat3(
	// 	viewmatrix[0], viewmatrix[1], viewmatrix[2],
	// 	viewmatrix[4], viewmatrix[5], viewmatrix[6],
	// 	viewmatrix[8], viewmatrix[9], viewmatrix[10]);

	glm::mat3 inv_rot_view = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 inv_mat_modelview = inv_scale_model * inv_rot_model * inv_rot_view;

	return make_float3x3(
		inv_mat_modelview[0][0], inv_mat_modelview[1][0], inv_mat_modelview[2][0],
		inv_mat_modelview[0][1], inv_mat_modelview[1][1], inv_mat_modelview[2][1],
		inv_mat_modelview[0][2], inv_mat_modelview[1][2], inv_mat_modelview[2][2]);
}

// Normalize a float3
__device__ float3 normalize(const float3 &v)
{
	float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	if (len > 0.0f)
		return make_float3(v.x / len, v.y / len, v.z / len);
	else
		return make_float3(0.0f, 0.0f, 0.0f); // Or handle zero-length as you wish
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void preprocessCUDA(int P, int D, int M,
							   const float *orig_points,
							   const float *cano_orig_points,
							   const glm::vec3 *scales,
							   const float scale_modifier,
							   const glm::vec4 *rotations,
							   const float *weight1,
							   const float *bias1,
							   const float *weight2,
							   const float *bias2,
							   const float *opacities,
							   const float *shs,
							   bool *clamped,
							   const float *cov3D_precomp,
							   const float *colors_precomp,
							   const float *viewmatrix,
							   const float *projmatrix,
							   const glm::vec3 *cam_pos,
							   const int W, int H,
							   const float tan_fovx, float tan_fovy,
							   const float focal_x, float focal_y,
							   int *radii,
							   float2 *points_xy_image,
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
							   float *rgb,
							   float4 *conic_opacity,
							   const dim3 grid,
							   uint32_t *tiles_touched,
							   bool prefiltered,
							   bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters.
	const float *cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if (antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles.
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3 *)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Compute canonical view-space position for MLP
	if (cano_orig_points != nullptr)
	{
		float3 p_orig_cano = {cano_orig_points[3 * idx], cano_orig_points[3 * idx + 1], cano_orig_points[3 * idx + 2]};
		cano_points_xyz_view[idx] = transformPoint4x3(p_orig_cano, viewmatrix);
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];

	conic_opacity[idx] = {conic.x, conic.y, conic.z, opacity * h_convolution_scaling};

	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

	// + ---------------------- +
	// |      MLP Data Prep     |
	// + ---------------------- +

	float3x3 g_inv_mat_modelview = computeInvMatModelView(scales[idx], scale_modifier, rotations[idx], viewmatrix);

	float max_scale = scales[idx].x;
	int dim = 0;

	if (scales[idx].y > max_scale)
	{
		max_scale = scales[idx].y;
		dim = 1;
	}
	if (scales[idx].z > max_scale)
	{
		max_scale = scales[idx].z;
		dim = 2;
	}

	inv_max_radius[idx] = 1 / max_scale;
	max_dim[idx] = dim;

	inv_mat_modelview[idx] = g_inv_mat_modelview;
	// points_xyz_view[idx] = transformPoint4x3(p_orig[idx], viewmatrix);
	points_xyz_view[idx] = p_view;

	float8x3 g_weight1;
	float8 g_bias1;
	float8 g_weight2;
	float g_bias2;

	for (int i = 0; i < 8; i++)
	{
		g_weight1.rows[i] = make_float3(
			weight1[idx * 24 + i * 3 + 0],
			weight1[idx * 24 + i * 3 + 1],
			weight1[idx * 24 + i * 3 + 2]);
		g_bias1.data[i] = bias1[idx * 8 + i];
		g_weight2.data[i] = weight2[idx * 8 + i];
	}
	g_bias2 = bias2[idx];

	out_weight1[idx] = g_weight1;
	out_bias1[idx] = g_bias1;
	out_weight2[idx] = g_weight2;
	out_bias2[idx] = g_bias2;
}

/**
 * @brief Main rasterization method for MLPs. Similar to the previous one
 * @param CHANNELS Number of channels in the output color (e.g., 3 for RGB).
 * @details This method processes a tile of pixels, fetching Gaussian data
 * collectively and rasterizing it using a multi-layer perceptron (MLP).
 * It computes the contribution of each Gaussian to the pixel color and
 * depth, applying the MLP weights and biases to transform the Gaussian
 * features into a final color output.
 *
 * @param ranges Pointer to the ranges of Gaussian IDs to process.
 * @param point_list Pointer to the list of Gaussian IDs.
 * @param W Width of the image.
 * @param H Height of the image.
 * @param fx Focal length in the x direction.
 * @param fy Focal length in the y direction.
 * @param points_xyz_view Pointer to the 3D positions of the Gaussians in view space.
 * @param features Pointer to the RGB colors of the Gaussians.
 * @param weight1 Pointer to the first layer MLP weights.
 * @param bias1 Pointer to the first layer MLP biases.
 * @param weight2 Pointer to the second layer MLP weights.
 * @param bias2 Pointer to the second layer MLP biases.
 * @param inv_mat_modelview Pointer to the inverse matrices to transform from view space to model space, using only rotation and scale. inv_mat_modelview = S_model^(-1) * R_model^T * R_view^T
 * @param final_T Pointer to the final transparency values for each pixel.
 * @param n_contrib Pointer to the number of contributing Gaussians for each pixel.
 * @param bg_color Pointer to the background color.
 * @param out_color Pointer to the output color buffer.
 * @param depths Pointer to the depth values of the Gaussians.
 * @param invdepth Pointer to the inverse depth values for each pixel.
 */
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderMLPCUDA(
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int W, int H,
		float fx, float fy,
		const float3 *__restrict__ points_xyz_view,
		const float3 *__restrict__ cano_points_xyz_view,
		const float *__restrict__ features,
		const float8x3 *__restrict__ weight1,
		const float8 *__restrict__ bias1,
		const float8 *__restrict__ weight2,
		const float *__restrict__ bias2,
		const float3x3 *__restrict__ inv_mat_modelview,
		const float *__restrict__ viewmatrix,
		const float *__restrict__ inv_max_radius,
		float *__restrict__ final_T,
		uint32_t *__restrict__ n_contrib,
		const float *__restrict__ bg_color,
		float *__restrict__ out_color,
		const float *__restrict__ depths,
		float *__restrict__ invdepth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = {(float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Compute direction vector from camera to pixel.
	float3 pixDir = {(pixf.x - (float)W / 2.0f + 0.5f) / fx, (pixf.y - (float)H / 2.0f + 0.5f) / fy, 1.0f};
	// float3 pixDir = {(340.0f - (float)W / 2.0f + 0.5f) / fx, (558.0f - (float)H / 2.0f + 0.5f) / fy, 1.0f};
	// float3 pixDir = {(318.0f - (float)W / 2.0f + 0.5f) / fx, (552.0f - (float)H / 2.0f + 0.5f) / fy, 1.0f};
	// float3 pixDir = {(354.0f - (float)W / 2.0f + 0.5f) / fx, (604.0f - (float)H / 2.0f + 0.5f) / fy, 1.0f};
	// float3 pixDir = {(357.0f - (float)W / 2.0f + 0.5f) / fx, (603.0f - (float)H / 2.0f + 0.5f) / fy, 1.0f};
	// float3 pixDir = {(150.0f - (float)W / 2.0f + 0.5f) / fx, (150.0f - (float)H / 2.0f + 0.5f) / fy, 1.0f};

	pixDir = normalize(pixDir);

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_xyz[BLOCK_SIZE];
	__shared__ float3 collected_cano_xyz[BLOCK_SIZE];
	__shared__ float3x3 collected_inv_mat_modelview[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = {0};

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xyz[block.thread_rank()] = points_xyz_view[coll_id];
			collected_cano_xyz[block.thread_rank()] = cano_points_xyz_view[coll_id];
			collected_inv_mat_modelview[block.thread_rank()] = inv_mat_modelview[coll_id];
		}

		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Compute the 3D position of the Gaussian in view space.
			float3 g_xyz_vs = collected_xyz[j];
			float3x3 g_inv_mat_modelview = collected_inv_mat_modelview[j];

			float integral = 0;
			float8x3 g_weight1 = weight1[collected_id[j]];
			float8 g_bias1 = bias1[collected_id[j]];
			float8 g_weight2 = weight2[collected_id[j]];
			float g_bias2 = bias2[collected_id[j]];
			float g_inv_max_radius = inv_max_radius[collected_id[j]];

			bool hit = forward_mlp_siren(
				g_xyz_vs,
				collected_cano_xyz[j],
				pixDir,
				g_inv_mat_modelview,
				g_inv_max_radius,
				viewmatrix,
				g_weight1,
				g_bias1,
				g_weight2,
				g_bias2,
				integral);

			if (!hit)
				continue;

			float alpha = min(0.99f, 1 - exp(-integral));
			// printf("g_xyz_vs: %f %f %f \n", g_xyz_vs.x, g_xyz_vs.y, g_xyz_vs.z);

			// Alpha blending
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if (invdepth)
				expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
			invdepth[pix_id] = expected_invdepth; // 1. / (expected_depth + T * 1e3);
	}
}

void FORWARD_MLP::render(
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
	const float3x3 *inv_mat_modelview,
	const float *viewmatrix,
	const float *inv_max_radius,
	const float4 *conic_opacity,
	float *final_T,
	uint32_t *n_contrib,
	const float *bg_color,
	float *out_color,
	float *depths,
	float *depth)
{
	renderMLPCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		W, H,
		fx, fy,
		points_xyz_view,
		cano_points_xyz_view,
		colors,
		weight1,
		bias1,
		weight2,
		bias2,
		inv_mat_modelview,
		viewmatrix,
		inv_max_radius,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths,
		depth);
}

void FORWARD_MLP::preprocess(int P, int D, int M,
							 const float *means3D,
							 const float *cano_means3D,
							 const glm::vec3 *scales,
							 const float scale_modifier,
							 const glm::vec4 *rotations,
							 const float *weight1,
							 const float *bias1,
							 const float *weight2,
							 const float *bias2,
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
							 float *rgb,
							 float4 *conic_opacity,
							 const dim3 grid,
							 uint32_t *tiles_touched,
							 bool prefiltered,
							 bool antialiasing)
{
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M,
		means3D,
		cano_means3D,
		scales,
		scale_modifier,
		rotations,
		weight1,
		bias1,
		weight2,
		bias2,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		inv_mat_modelview,
		points_xyz_view,
		cano_points_xyz_view,
		inv_max_radius,
		max_dim,
		out_weight1,
		out_bias1,
		out_weight2,
		out_bias2,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing);
}
