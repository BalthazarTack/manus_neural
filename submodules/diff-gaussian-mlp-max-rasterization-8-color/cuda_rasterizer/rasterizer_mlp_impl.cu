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

#include "rasterizer_mlp_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward_mlp.h"
#include "backward_mlp.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
							 const float *orig_points,
							 const float *viewmatrix,
							 const float *projmatrix,
							 bool *present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2 *points_xy,
	const float *depths,
	const uint32_t *offsets,
	uint64_t *gaussian_keys_unsorted,
	uint32_t *gaussian_values_unsorted,
	int *radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth.
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t *)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t *point_list_keys, uint2 *ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizerMLP::RasterizerMLP::markVisible(
	int P,
	float *means3D,
	float *viewmatrix,
	float *projmatrix,
	bool *present)
{
	checkFrustum<<<(P + 255) / 256, 256>>>(
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizerMLP::GeometryMLPState CudaRasterizerMLP::GeometryMLPState::fromChunk(char *&chunk, size_t P)
{
	GeometryMLPState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);

	// New fields for MLP (add at the end)
	obtain(chunk, geom.means3D_view, P, 128);
	obtain(chunk, geom.cano_means3D_view, P, 128);
	obtain(chunk, geom.inv_mat_modelview, P, 128);
	obtain(chunk, geom.inv_max_radius, P, 128);
	obtain(chunk, geom.max_dim, P, 128);

	obtain(chunk, geom.weight1, P, 128);
	obtain(chunk, geom.bias1, P, 128);
	obtain(chunk, geom.weight2, P, 128);
	obtain(chunk, geom.bias2, P, 128);
	obtain(chunk, geom.weight2_color, P, 128);
	obtain(chunk, geom.bias2_color, P, 128);

	// obtain(chunk, geom.inv_mat_modelview, P, alignof(float3x3));
	// obtain(chunk, geom.weight1, P, alignof(float8x3));
	// obtain(chunk, geom.bias1, P, alignof(float8));
	// obtain(chunk, geom.weight2, P, alignof(float8));
	// obtain(chunk, geom.bias2, P, alignof(float));

	return geom;
}

CudaRasterizerMLP::ImageState CudaRasterizerMLP::ImageState::fromChunk(char *&chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizerMLP::BinningState CudaRasterizerMLP::BinningState::fromChunk(char *&chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizerMLP::RasterizerMLP::forward(
	std::function<char *(size_t)> geometryBuffer,
	std::function<char *(size_t)> binningBuffer,
	std::function<char *(size_t)> imageBuffer,
	const int P, int D, int M,
	const float *background,
	const int width, int height,
	const float *means3D,
	const float *cano_means3D,
	const float *shs,
	const float *colors_precomp,
	const float *opacities,
	const float *scales,
	const float scale_modifier,
	const float *rotations,
	const float *weight1,
	const float *bias1,
	const float *weight2,
	const float *bias2,
	const float *weight2_color,
	const float *bias2_color,
	const float *cov3D_precomp,
	const float *viewmatrix,
	const float *projmatrix,
	const float *cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float *out_color,
	float *depth,
	bool antialiasing,
	int *radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryMLPState>(P);
	char *chunkptr = geometryBuffer(chunk_size);
	GeometryMLPState geomState = GeometryMLPState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char *img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD_MLP::preprocess(
				   P, D, M,
				   means3D,
				   cano_means3D,
				   (glm::vec3 *)scales,
				   scale_modifier,
				   (glm::vec4 *)rotations,
				   weight1,
				   bias1,
				   weight2,
				   bias2,
				   weight2_color,
				   bias2_color,
				   opacities,
				   shs,
				   geomState.clamped,
				   cov3D_precomp,
				   colors_precomp,
				   viewmatrix,
				   projmatrix,
				   (glm::vec3 *)cam_pos,
				   width, height,
				   focal_x, focal_y,
				   tan_fovx, tan_fovy,
				   radii,
				   geomState.means2D,
				   geomState.depths,
				   geomState.cov3D,
				   geomState.inv_mat_modelview,
				   geomState.means3D_view,
				   geomState.cano_means3D_view,
				   geomState.inv_max_radius,
				   geomState.max_dim,
				   geomState.weight1,
				   geomState.bias1,
				   geomState.weight2,
				   geomState.bias2,
				   geomState.weight2_color,
				   geomState.bias2_color,
				   geomState.rgb,
				   geomState.conic_opacity,
				   tile_grid,
				   geomState.tiles_touched,
				   prefiltered,
				   antialiasing),
			   debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char *binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys<<<(P + 255) / 256, 256>>>(
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
		CHECK_CUDA(, debug)

			int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
				   binningState.list_sorting_space,
				   binningState.sorting_size,
				   binningState.point_list_keys_unsorted, binningState.point_list_keys,
				   binningState.point_list_unsorted, binningState.point_list,
				   num_rendered, 0, 32 + bit),
			   debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float *feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD_MLP::render(
				   tile_grid, block,
				   imgState.ranges,
				   binningState.point_list,
				   width, height,
				   focal_x, focal_y,
				   geomState.means3D_view,
				   geomState.cano_means3D_view,
				   feature_ptr,
				   geomState.weight1,
				   geomState.bias1,
				   geomState.weight2,
				   geomState.bias2,
				   geomState.weight2_color,
				   geomState.bias2_color,
				   geomState.inv_mat_modelview,
				   viewmatrix,
				   geomState.inv_max_radius,
				   geomState.conic_opacity,
				   imgState.accum_alpha,
				   imgState.n_contrib,
				   background,
				   out_color,
				   geomState.depths,
				   depth),
			   debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizerMLP::RasterizerMLP::backward(
	const int P, int D, int M, int R,
	const float *background,
	const int width, int height,
	const float *means3D_world,
	const float *shs,
	const float *colors_precomp,
	const float *opacities,
	const float *scales,
	const float scale_modifier,
	const float *rotations,
	const float *weight1,
	const float *bias1,
	const float *weight2,
	const float *bias2,
	const float *weight2_color,
	const float *bias2_color,
	const float *cov3D_precomp,
	const float *viewmatrix,
	const float *projmatrix,
	const float *campos,
	const float tan_fovx, float tan_fovy,
	const int *radii,
	char *geom_buffer,
	char *binning_buffer,
	char *img_buffer,
	const float *dL_dpix,
	const float *dL_invdepths,
	float *dL_dmean2D,
	float *dL_dconic,
	float *dL_dopacity,
	float *dL_dcolor,
	float *dL_dinvdepth,
	float *dL_dmean3D_world,
	float *dL_dinv_max_radius,
	float *dL_dcov3D,
	float *dL_dsh,
	float *dL_dscale,
	float *dL_drot,
	float *dL_dweight1,
	float *dL_dbias1,
	float *dL_dweight2,
	float *dL_dbias2,
	float *dL_dweight2_color,
	float *dL_dbias2_color,
	float *dL_dinv_mat_modelview,
	float *dL_dmeans3D_view,
	float *dL_dcano_means3D_view,
	bool antialiasing,
	bool debug)
{

	GeometryMLPState geomState = GeometryMLPState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float *color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD_MLP::renderMLP(
				   tile_grid,
				   block,
				   imgState.ranges,
				   binningState.point_list,
				   width, height,
				   focal_x, focal_y,
				   background,
				   viewmatrix,
				   geomState.means3D_view,
				   geomState.cano_means3D_view,
				   geomState.inv_max_radius,
				   geomState.weight1,
				   geomState.bias1,
				   geomState.weight2,
				   geomState.bias2,
				   geomState.weight2_color,
				   geomState.bias2_color,
				   geomState.inv_mat_modelview,
				   geomState.depths,
				   imgState.accum_alpha,
				   imgState.n_contrib,
				   dL_dpix,
				   dL_invdepths,
				   (float3 *)dL_dmeans3D_view,
				   (float3 *)dL_dcano_means3D_view,
				   dL_dinv_max_radius,
				   (float8x3 *)dL_dweight1,
				   (float8 *)dL_dbias1,
				   (float8 *)dL_dweight2,
				   dL_dbias2,
				   (float8x3 *)dL_dweight2_color,  // NOUVEAU
               	   (float3 *)dL_dbias2_color,
				   (float3x3 *)dL_dinv_mat_modelview,
				   dL_dinvdepth),
			   debug);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float *cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD_MLP::preprocess(P, D, M,
										(float3 *)means3D_world,
										radii,
										shs,
										geomState.clamped,
										opacities,
										(glm::vec3 *)scales,
										(glm::vec4 *)rotations,
										scale_modifier,
										viewmatrix,
										projmatrix,
										focal_x, focal_y,
										tan_fovx, tan_fovy,
										(glm::vec3 *)campos,
										(float3 *)dL_dmeans3D_view,
										dL_dinv_max_radius,
										geomState.max_dim,
										dL_dcolor,
										(float3x3 *)dL_dinv_mat_modelview,
										dL_dinvdepth,
										dL_dopacity,
										(glm::vec3 *)dL_dmean3D_world,
										dL_dsh,
										(glm::vec3 *)dL_dscale,
										(glm::vec4 *)dL_drot,
										antialiasing),
			   debug);
}
