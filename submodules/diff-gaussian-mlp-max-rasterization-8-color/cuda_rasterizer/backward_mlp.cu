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

#include "backward_mlp.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "mlp_kernels.cu"
#include <stdio.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 *means, glm::vec3 campos, const float *shs, const bool *clamped, const glm::vec3 *dL_dcolor, glm::vec3 *dL_dmeans, glm::vec3 *dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3 *dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy +
						   SH_C3[1] * sh[10] * yz +
						   SH_C3[2] * sh[11] * -2.f * xy +
						   SH_C3[3] * sh[12] * -3.f * 2.f * xz +
						   SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
						   SH_C3[5] * sh[14] * 2.f * xz +
						   SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (SH_C3[0] * sh[9] * 3.f * (xx - yy) +
						   SH_C3[1] * sh[10] * xz +
						   SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
						   SH_C3[3] * sh[12] * -3.f * 2.f * yz +
						   SH_C3[4] * sh[13] * -2.f * xy +
						   SH_C3[5] * sh[14] * -2.f * yz +
						   SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (SH_C3[1] * sh[10] * xy +
						   SH_C3[2] * sh[11] * 4.f * 2.f * yz +
						   SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
						   SH_C3[4] * sh[13] * 4.f * 2.f * xz +
						   SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z}, float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
								 const float3 *means,
								 const int *radii,
								 const float *cov3Ds,
								 const float h_x, float h_y,
								 const float tan_fovx, float tan_fovy,
								 const float *view_matrix,
								 const float *opacities,
								 const float *dL_dconics,
								 float *dL_dopacity,
								 const float *dL_dinvdepth,
								 float3 *dL_dmeans,
								 float *dL_dcov,
								 bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float *cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = {dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3]};
	float3 t = transformPoint4x3(mean, view_matrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
							0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
							0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float c_xx = cov2D[0][0];
	float c_xy = cov2D[0][1];
	float c_yy = cov2D[1][1];

	constexpr float h_var = 0.3f;
	float d_inside_root = 0.f;
	if (antialiasing)
	{
		const float det_cov = c_xx * c_yy - c_xy * c_xy;
		c_xx += h_var;
		c_yy += h_var;
		const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
		const float h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability
		const float dL_dopacity_v = dL_dopacity[idx];
		const float d_h_convolution_scaling = dL_dopacity_v * opacities[idx];
		dL_dopacity[idx] = dL_dopacity_v * h_convolution_scaling;
		d_inside_root = (det_cov / det_cov_plus_h_cov) <= 0.000025f ? 0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);
	}
	else
	{
		c_xx += h_var;
		c_yy += h_var;
	}

	float dL_dc_xx = 0;
	float dL_dc_xy = 0;
	float dL_dc_yy = 0;
	if (antialiasing)
	{
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdx
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdz
		const float x = c_xx;
		const float y = c_yy;
		const float z = c_xy;
		const float w = h_var;
		const float denom_f = d_inside_root / sq(w * w + w * (x + y) + x * y - z * z);
		const float dL_dx = w * (w * y + y * y + z * z) * denom_f;
		const float dL_dy = w * (w * x + x * x + z * z) * denom_f;
		const float dL_dz = -2.f * w * z * (w + x + y) * denom_f;
		dL_dc_xx = dL_dx;
		dL_dc_yy = dL_dy;
		dL_dc_xy = dL_dz;
	}

	float denom = c_xx * c_yy - c_xy * c_xy;

	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a

		dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x + 2 * c_xy * c_yy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.z);
		dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z + 2 * c_xx * c_xy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.x);
		dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x - (denom + 2 * c_xy * c_xy) * dL_dconic.y + c_xx * c_xy * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_dc_xx + T[0][0] * T[1][0] * dL_dc_xy + T[1][0] * T[1][0] * dL_dc_yy);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_dc_xx + T[0][1] * T[1][1] * dL_dc_xy + T[1][1] * T[1][1] * dL_dc_yy);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_dc_xx + T[0][2] * T[1][2] * dL_dc_xy + T[1][2] * T[1][2] * dL_dc_yy);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_dc_xx + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][1] * dL_dc_yy;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_dc_xx + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][2] * dL_dc_yy;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_dc_xx + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_dc_xy + 2 * T[1][1] * T[1][2] * dL_dc_yy;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xx +
					(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xx +
					(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xx +
					(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_xy;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_yy +
					(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_yy +
					(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_yy +
					(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xy;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;
	// Account for inverse depth gradients
	if (dL_dinvdepth)
		dL_dtz -= dL_dinvdepth[idx] / (t.z * t.z);

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({dL_dtx, dL_dty, dL_dtz}, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a
// 3D covariance matrix for each Gaussian.
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float *dL_dcov3Ds, glm::vec3 *dL_dscales, glm::vec4 *dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot; // / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float *dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3 *dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4 *dL_drot = (float4 *)(dL_drots + idx);
	*dL_drot = float4{dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w}; // dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// S_model^-1 * R_model^-1 * R_view^-1
__device__ void backward_computeInvMatModelView(
	const glm::vec3 scale,
	float mod,
	const glm::vec4 rot,
	const float *viewmatrix,
	const float3x3 &dL_dinv_mat_modelview,
	glm::vec3 &dL_dscales,
	glm::vec4 &dL_drots)
{

	/*
		// printf("scale: %f %f %f\n", scale.x, scale.y, scale.z);
		// printf("rot: %f %f %f %f\n", rot.x, rot.y, rot.z, rot.w);
		// printf("viewmatrix:\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f \n",
		// 	   viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
		// 	   viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
		// 	   viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
		// 	   viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]);

		// printf("dL_dinv_mat_modelview:\n %f %f %f\n %f %f %f\n %f %f %f \n",
		// 	   dL_dinv_mat_modelview.rows[0].x, dL_dinv_mat_modelview.rows[0].y, dL_dinv_mat_modelview.rows[0].z,
		// 	   dL_dinv_mat_modelview.rows[1].x, dL_dinv_mat_modelview.rows[1].y, dL_dinv_mat_modelview.rows[1].z,
		// 	   dL_dinv_mat_modelview.rows[2].x, dL_dinv_mat_modelview.rows[2].y, dL_dinv_mat_modelview.rows[2].z);

		// Recompute inv_scale_model (A)
		glm::mat3 inv_scale_model = glm::mat3(1.0f);
		inv_scale_model[0][0] = 1.0f / (mod * scale.x);
		inv_scale_model[1][1] = 1.0f / (mod * scale.y);
		inv_scale_model[2][2] = 1.0f / (mod * scale.z);

		// Recompute rot_model from the quaternion
		glm::vec4 q = rot;
		float r = q.x;
		float x = q.y;
		float y = q.z;
		float z = q.w;

		glm::mat3 rot_model = glm::mat3(
			1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
			2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
			2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
		glm::mat3 inv_rot_model = glm::transpose(rot_model); // B

		// Recompute inv_rot_view (C) from viewmatrix
		glm::mat3 rot_view = glm::mat3(
			viewmatrix[0], viewmatrix[4], viewmatrix[8],
			viewmatrix[1], viewmatrix[5], viewmatrix[9],
			viewmatrix[2], viewmatrix[6], viewmatrix[10]);
		glm::mat3 inv_rot_view = glm::transpose(rot_view); // C = transpose(rot_view)

		// Compute D = B * C
		glm::mat3 D = inv_rot_model * inv_rot_view;

		// Convert input gradient to glm::mat3
		// glm::mat3 dL_dM = glm::mat3(
		// 	dL_dinv_mat_modelview->data[0], dL_dinv_mat_modelview->data[1], dL_dinv_mat_modelview->data[2],
		// 	dL_dinv_mat_modelview->data[3], dL_dinv_mat_modelview->data[4], dL_dinv_mat_modelview->data[5],
		// 	dL_dinv_mat_modelview->data[6], dL_dinv_mat_modelview->data[7], dL_dinv_mat_modelview->data[8]);

		glm::mat3 dL_dM = glm::mat3(
			dL_dinv_mat_modelview.rows[0].x, dL_dinv_mat_modelview.rows[0].y, dL_dinv_mat_modelview.rows[0].z,
			dL_dinv_mat_modelview.rows[1].x, dL_dinv_mat_modelview.rows[1].y, dL_dinv_mat_modelview.rows[1].z,
			dL_dinv_mat_modelview.rows[2].x, dL_dinv_mat_modelview.rows[2].y, dL_dinv_mat_modelview.rows[2].z);

		// Compute dL_dA = dL_dM * D^T
		glm::mat3 dL_dA = dL_dM * glm::transpose(D);
		// glm::mat3 dL_dA = glm::transpose(D) * dL_dM;

		glm::mat3 D_transpose = glm::transpose(D);
		printf("D_transpose:\n %f %f %f\n %f %f %f\n %f %f %f \n",
			   D_transpose[0][0], D_transpose[0][1], D_transpose[0][2],
			   D_transpose[1][0], D_transpose[1][1], D_transpose[1][2],
			   D_transpose[2][0], D_transpose[2][1], D_transpose[2][2]);

		printf("dL_dM:\n %f %f %f\n %f %f %f\n %f %f %f \n",
			   dL_dM[0][0], dL_dM[0][1], dL_dM[0][2],
			   dL_dM[1][0], dL_dM[1][1], dL_dM[1][2],
			   dL_dM[2][0], dL_dM[2][1], dL_dM[2][2]);

		printf("dL_dA:\n %f %f %f\n %f %f %f\n %f %f %f \n",
			   dL_dA[0][0], dL_dA[0][1], dL_dA[0][2],
			   dL_dA[1][0], dL_dA[1][1], dL_dA[1][2],
			   dL_dA[2][0], dL_dA[2][1], dL_dA[2][2]);

		float tmp1 = dL_dM[0][0] * D_transpose[0][0] + dL_dM[0][1] * D_transpose[1][0] + dL_dM[0][2] * D_transpose[2][0];
		float tmp2 = dL_dM[1][0] * D_transpose[0][1] + dL_dM[1][1] * D_transpose[1][1] + dL_dM[1][2] * D_transpose[2][1];
		float tmp3 = dL_dM[2][0] * D_transpose[0][2] + dL_dM[2][1] * D_transpose[1][2] + dL_dM[2][2] * D_transpose[2][2];

		printf("tmp: %f %f %f\n", tmp1, tmp2, tmp3);

		// Compute gradient for scales
		float denom_x = mod * scale.x * scale.x;
		dL_dscales.x += dL_dA[0][0] * (-1.0f / denom_x);

		float denom_y = mod * scale.y * scale.y;
		dL_dscales.y += dL_dA[1][1] * (-1.0f / denom_y);

		float denom_z = mod * scale.z * scale.z;
		dL_dscales.z += dL_dA[2][2] * (-1.0f / denom_z);

		// Compute dL_dD = A^T * dL_dM (A is diagonal, so A^T = A)
		glm::mat3 dL_dD = inv_scale_model * dL_dM;

		// Compute dL_dB = dL_dD * C^T = dL_dD * rot_view (since C^T = rot_view)
		glm::mat3 dL_dB = dL_dD * rot_view;

		// Compute dL_dR = (dL_dB)^T (gradient w.r.t. original rotation matrix)
		glm::mat3 dL_dR = glm::transpose(dL_dB);

		// Extract components of dL_dR for gradient calculation
		float d00 = dL_dR[0][0], d01 = dL_dR[0][1], d02 = dL_dR[0][2];
		float d10 = dL_dR[1][0], d11 = dL_dR[1][1], d12 = dL_dR[1][2];
		float d20 = dL_dR[2][0], d21 = dL_dR[2][1], d22 = dL_dR[2][2];

		// Compute gradients for quaternion components
		float dr = 2.0f * (x * (d21 - d12) + y * (d02 - d20) + z * (d10 - d01));
		float dx = 2.0f * (y * (d01 + d10) + z * (d02 + d20) + r * (d21 - d12) - 2.0f * x * (d11 + d22));
		float dy = 2.0f * (x * (d01 + d10) + z * (d12 + d21) + r * (d02 - d20) - 2.0f * y * (d00 + d22));
		float dz = 2.0f * (x * (d02 + d20) + y * (d12 + d21) + r * (d10 - d01) - 2.0f * z * (d00 + d11));

		glm::vec4 dL_drot_val = glm::vec4(dr, dx, dy, dz);
		// atomicAdd(&dL_drots->x, dL_drot_val.x);
		// atomicAdd(&dL_drots->y, dL_drot_val.y);
		// atomicAdd(&dL_drots->z, dL_drot_val.z);
		// atomicAdd(&dL_drots->w, dL_drot_val.w);
		dL_drots.x += dL_drot_val.x;
		dL_drots.y += dL_drot_val.y;
		dL_drots.z += dL_drot_val.z;
		dL_drots.w += dL_drot_val.w;
	*/

	// Recompute inv_scale_model (A)
	glm::mat3 inv_scale_model = glm::mat3(1.0f);
	inv_scale_model[0][0] = 1.0f / (mod * scale.x);
	inv_scale_model[1][1] = 1.0f / (mod * scale.y);
	inv_scale_model[2][2] = 1.0f / (mod * scale.z);

	// Recompute rot_model from the quaternion
	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 inv_rot_model = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)); // B
	glm::mat3 rot_model = glm::transpose(inv_rot_model);

	// Recompute inv_rot_view (C) from viewmatrix
	glm::mat3 inv_rot_view = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]); // C = transpose(rot_view)

	glm::mat3 rot_view = glm::transpose(inv_rot_view);

	// Compute D = B * C
	glm::mat3 D = inv_rot_model * inv_rot_view;

	glm::mat3 dL_dM = glm::mat3(
		dL_dinv_mat_modelview.rows[0].x, dL_dinv_mat_modelview.rows[1].x, dL_dinv_mat_modelview.rows[2].x,
		dL_dinv_mat_modelview.rows[0].y, dL_dinv_mat_modelview.rows[1].y, dL_dinv_mat_modelview.rows[2].y,
		dL_dinv_mat_modelview.rows[0].z, dL_dinv_mat_modelview.rows[1].z, dL_dinv_mat_modelview.rows[2].z);

	// Compute dL_dA = dL_dM * D^T
	glm::mat3 dL_dA = dL_dM * glm::transpose(D);
	// glm::mat3 dL_dA = glm::transpose(D) * dL_dM;

	// Compute gradient for scales
	float denom_x = mod * scale.x * scale.x;
	dL_dscales.x += dL_dA[0][0] * (-1.0f / denom_x);

	float denom_y = mod * scale.y * scale.y;
	dL_dscales.y += dL_dA[1][1] * (-1.0f / denom_y);

	float denom_z = mod * scale.z * scale.z;
	dL_dscales.z += dL_dA[2][2] * (-1.0f / denom_z);

	// Compute dL_dD = A^T * dL_dM (A is diagonal, so A^T = A)
	glm::mat3 dL_dD = inv_scale_model * dL_dM;

	// Compute dL_dB = dL_dD * C^T = dL_dD * rot_view (since C^T = rot_view)
	glm::mat3 dL_dB = dL_dD * rot_view;

	// Compute dL_dR = (dL_dB)^T (gradient w.r.t. original rotation matrix)
	glm::mat3 dL_dR = glm::transpose(dL_dB);

	// Extract components of dL_dR for gradient calculation
	float d00 = dL_dR[0][0], d01 = dL_dR[1][0], d02 = dL_dR[2][0];
	float d10 = dL_dR[0][1], d11 = dL_dR[1][1], d12 = dL_dR[2][1];
	float d20 = dL_dR[0][2], d21 = dL_dR[1][2], d22 = dL_dR[2][2];

	// Compute gradients for quaternion components
	float dr = 2.0f * (x * (d21 - d12) + y * (d02 - d20) + z * (d10 - d01));
	float dx = 2.0f * (y * (d01 + d10) + z * (d02 + d20) + r * (d21 - d12) - 2.0f * x * (d11 + d22));
	float dy = 2.0f * (x * (d01 + d10) + z * (d12 + d21) + r * (d02 - d20) - 2.0f * y * (d00 + d22));
	float dz = 2.0f * (x * (d02 + d20) + y * (d12 + d21) + r * (d10 - d01) - 2.0f * z * (d00 + d11));

	glm::vec4 dL_drot_val = glm::vec4(dr, dx, dy, dz);

	dL_drots.x += dL_drot_val.x;
	dL_drots.y += dL_drot_val.y;
	dL_drots.z += dL_drot_val.z;
	dL_drots.w += dL_drot_val.w;
}

__device__ void backward_transformView(
	const float3 &dL_dxyz_view,
	const float *view_matrix,
	float3 &dL_dxyz_world)
{
	float dL_dx = dL_dxyz_view.x * view_matrix[0] + dL_dxyz_view.y * view_matrix[1] + dL_dxyz_view.z * view_matrix[2];
	float dL_dy = dL_dxyz_view.x * view_matrix[4] + dL_dxyz_view.y * view_matrix[5] + dL_dxyz_view.z * view_matrix[6];
	float dL_dz = dL_dxyz_view.x * view_matrix[8] + dL_dxyz_view.y * view_matrix[9] + dL_dxyz_view.z * view_matrix[10];

	// *dL_dxyz_world = make_float3(dL_dx, dL_dy, dL_dz);

	dL_dxyz_world.x = dL_dx;
	dL_dxyz_world.y = dL_dy;
	dL_dxyz_world.z = dL_dz;
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template <int C>
__global__ void preprocessMLPCUDA(
	int P, int D, int M,
	const float3 *means_world,
	const int *radii,
	const float *shs,
	const bool *clamped,
	const glm::vec3 *scales,
	const glm::vec4 *rotations,
	const float scale_modifier,
	const float *proj,
	const float *view_matrix,
	const glm::vec3 *campos,
	const float3 *dL_dpoints_xyz_view,
	const float *dL_dinv_max_radius,
	const int *max_dim,
	const float *dL_dcolor,
	const float3x3 *dL_dinv_mat_modelview,
	glm::vec3 *dL_dmeans_world,
	float *dL_dsh,
	glm::vec3 *dL_dscale,
	glm::vec4 *dL_drot,
	float *dL_dopacity)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means_world[idx];
	// printf("dL_dpoints_xyz_view[%d]: %f %f %f\n", idx, dL_dpoints_xyz_view[idx].x, dL_dpoints_xyz_view[idx].y, dL_dpoints_xyz_view[idx].z);
	// printf("dL_dinv_mat_modelview[%d].rows: %f %f %f\n", idx, dL_dinv_mat_modelview[idx].rows[0].x, dL_dinv_mat_modelview[idx].rows[0].y, dL_dinv_mat_modelview[idx].rows[0].z);
	// printf("dL_dinv_mat_modelview[%d].rows: %f %f %f\n", idx, dL_dinv_mat_modelview[idx].rows[1].x, dL_dinv_mat_modelview[idx].rows[1].y, dL_dinv_mat_modelview[idx].rows[1].z);
	// printf("dL_dinv_mat_modelview[%d].rows: %f %f %f\n", idx, dL_dinv_mat_modelview[idx].rows[2].x, dL_dinv_mat_modelview[idx].rows[2].y, dL_dinv_mat_modelview[idx].rows[2].z);

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 3D means in view space
	// from rendering procedure
	float3 dL_dxyz_world_from_view;
	backward_transformView(dL_dpoints_xyz_view[idx], view_matrix, dL_dxyz_world_from_view);

	dL_dmeans_world[idx].x += dL_dxyz_world_from_view.x;
	dL_dmeans_world[idx].y += dL_dxyz_world_from_view.y;
	dL_dmeans_world[idx].z += dL_dxyz_world_from_view.z;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3 *)means_world, *campos, shs, clamped, (glm::vec3 *)dL_dcolor, (glm::vec3 *)dL_dmeans_world, (glm::vec3 *)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
	{
		// computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
		backward_computeInvMatModelView(
			scales[idx],
			scale_modifier,
			rotations[idx],
			view_matrix,
			dL_dinv_mat_modelview[idx],
			dL_dscale[idx],
			dL_drot[idx]);

		int dim = max_dim[idx];

		// if (dim == 0)
		// {
		// 	dL_dscale[idx].x += dL_dinv_max_radius[idx] * (-1 / (scales[idx].x * scales[idx].x));
		// }
		// else if (dim == 1)
		// {
		// 	dL_dscale[idx].y += dL_dinv_max_radius[idx] * (-1 / (scales[idx].y * scales[idx].y));
		// }
		// else
		// {
		// 	dL_dscale[idx].z += dL_dinv_max_radius[idx] * (-1 / (scales[idx].z * scales[idx].z));
		// }

		// Faster
		dL_dscale[idx][dim] += dL_dinv_max_radius[idx] * (-1.0f / (scales[idx][dim] * scales[idx][dim]));
	}
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderMLPCUDA(
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int W, int H,
		float fx, float fy,
		const float *__restrict__ bg_color,
		const float *__restrict__ viewmatrix,
		const float3 *__restrict__ points_xyz_view,
		const float3 *__restrict__ cano_points_xyz_view,
		const float *__restrict__ inv_max_radius,
		const float8x3 *__restrict__ weight1,
		const float8 *__restrict__ bias1,
		const float8 *__restrict__ weight2,
		const float *__restrict__ bias2,
		const float8x3 *__restrict__ weight2_color,  // ⭐ NOUVEAU
        const float3 *__restrict__ bias2_color,
		const float3x3 *__restrict__ inv_mat_modelview,
		const float *__restrict__ depths,
		const float *__restrict__ final_Ts,
		const uint32_t *__restrict__ n_contrib,
		const float *__restrict__ dL_dpixels,
		const float *__restrict__ dL_invdepths,
		float3 *__restrict__ dL_dpoints_xyz_view,
		float3 *__restrict__ dL_dcano_points_xyz_view,
		float *__restrict__ dL_dinv_max_radius,
		float8x3 *__restrict__ dL_dweight1,
		float8 *__restrict__ dL_dbias1,
		float8 *__restrict__ dL_dweight2,
		float *__restrict__ dL_dbias2,
		float8x3 *__restrict__ dL_dweight2_color,  // NOUVEAU
        float3 *__restrict__ dL_dbias2_color,
		float3x3 *__restrict__ dL_dinv_mat_modelview,
		float *__restrict__ dL_dinvdepths)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;

	// Compute direction vector from camera to pixel.
	float3 pixDir = {(pixf.x - (float)W / 2.0f + 0.5f) / fx, (pixf.y - (float)H / 2.0f + 0.5f) / fy, 1.0f};
	// float3 pixDir = {(340.0f - (float)W / 2.0f + 0.5f) / fx, (558.0f - (float)H / 2.0f + 0.5f) / fy, 1.0f};
	// float3 pixDir = {(318.0f - (float)W / 2.0f + 0.5f) / fx, (552.0f - (float)H / 2.0f + 0.5f) / fy, 1.0f};
	// float3 pixDir = {(354.0f - (float)W / 2.0f + 0.5f) / fx, (604.0f - (float)H / 2.0f + 0.5f) / fy, 1.0f};
	// float3 pixDir = {(357.0f - (float)W / 2.0f + 0.5f) / fx, (603.0f - (float)H / 2.0f + 0.5f) / fy, 1.0f};
	// float3 pixDir = {(150.0f - (float)W / 2.0f + 0.5f) / fx, (150.0f - (float)H / 2.0f + 0.5f) / fy, 1.0f};

	pixDir = pixDir / sqrtf(dot(pixDir, pixDir));

	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_xyz[BLOCK_SIZE];
	__shared__ float3 collected_cano_xyz[BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float3x3 collected_inv_mat_modelview[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors.
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float3 accum_rec = make_float3(0.f, 0.f, 0.f);
	float dL_dpixel[C];
	float dL_invdepth;
	float accum_invdepth_rec = 0;
	if (inside)
	{
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		if (dL_invdepths)
			dL_invdepth = dL_invdepths[pix_id];
	}

	float last_alpha = 0;
	float3 last_color = make_float3(0.f, 0.f, 0.f);
	float last_invdepth = 0;

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xyz[block.thread_rank()] = points_xyz_view[coll_id];
			collected_cano_xyz[block.thread_rank()] = cano_points_xyz_view[coll_id];
			collected_inv_mat_modelview[block.thread_rank()] = inv_mat_modelview[coll_id];


			if (dL_invdepths)
				collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			float3 g_xyz_vs = collected_xyz[j];
			float3x3 g_inv_mat_modelview = collected_inv_mat_modelview[j];

			float integral = 0;
			float8x3 g_weight1 = weight1[collected_id[j]];
			float8 g_bias1 = bias1[collected_id[j]];
			float8 g_weight2 = weight2[collected_id[j]];
			float g_bias2 = bias2[collected_id[j]];
			float8x3 g_weight2_color = weight2_color[collected_id[j]]; 
			float3 g_bias2_color = bias2_color[collected_id[j]];      
			float3 color_integral = make_float3(0.f, 0.f, 0.f); 
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
				g_weight2_color,
				g_bias2_color,
				integral, 
				color_integral);

			if (!hit)
				continue;

			const float G = exp(-integral);
			const float alpha = min(0.99f, 1 - G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];

			accum_rec.x = last_alpha * last_color.x + (1.f - last_alpha) * accum_rec.x;
			accum_rec.y = last_alpha * last_color.y + (1.f - last_alpha) * accum_rec.y;
			accum_rec.z = last_alpha * last_color.z + (1.f - last_alpha) * accum_rec.z;
			last_color = color_integral;

			dL_dalpha += (color_integral.x - accum_rec.x) * dL_dpixel[0];
			dL_dalpha += (color_integral.y - accum_rec.y) * dL_dpixel[1];
			dL_dalpha += (color_integral.z - accum_rec.z) * dL_dpixel[2];

			float3 dL_dcolor_integral = make_float3(
				dchannel_dcolor * dL_dpixel[0],
				dchannel_dcolor * dL_dpixel[1],
				dchannel_dcolor * dL_dpixel[2]
			);


			// Propagate gradients from inverse depth to alphaas and
			// per Gaussian inverse depths
			if (dL_dinvdepths)
			{
				const float invd = 1.f / collected_depths[j];
				accum_invdepth_rec = last_alpha * last_invdepth + (1.f - last_alpha) * accum_invdepth_rec;
				last_invdepth = invd;
				dL_dalpha += (invd - accum_invdepth_rec) * dL_invdepth;
				atomicAdd(&(dL_dinvdepths[global_id]), dchannel_dcolor * dL_invdepth);
			}

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Helpful reusable temporary variables
			const float dL_dintegral = G * dL_dalpha;

			float3 dL_dg_xyz_vs;
			float3 dL_dg_cano_xyz_vs;
			float3x3 dL_dg_inv_mat_modelview;
			float8x3 dL_dg_weight1;
			float8 dL_dg_bias1;
			float8 dL_dg_weight2;
			float dL_dg_bias2;
			float dL_dg_inv_max_radius;
			float8x3 dL_dg_weight2_color;  // NOUVEAU
			float3 dL_dg_bias2_color;     // NOUVEAU

			bool success = backward_mlp_siren(
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
				g_weight2_color,      // NOUVEAU
				g_bias2_color,         // NOUVEAU
				integral,
				color_integral,       // NOUVEAU (déjà calculé dans le forward)
				dL_dintegral,
				dL_dcolor_integral,   // NOUVEAU
				dL_dg_xyz_vs,
				dL_dg_cano_xyz_vs,
				dL_dg_inv_mat_modelview,
				dL_dg_inv_max_radius,
				dL_dg_weight1,
				dL_dg_bias1,
				dL_dg_weight2,
				dL_dg_bias2,
				dL_dg_weight2_color,  // NOUVEAU
				dL_dg_bias2_color);   // NOUVEAU

			if (!success)
				continue;

			// Update gradients
			atomicAdd(&dL_dpoints_xyz_view[global_id].x, dL_dg_xyz_vs.x);
			atomicAdd(&dL_dpoints_xyz_view[global_id].y, dL_dg_xyz_vs.y);
			atomicAdd(&dL_dpoints_xyz_view[global_id].z, dL_dg_xyz_vs.z);

			atomicAdd(&dL_dcano_points_xyz_view[global_id].x, dL_dg_cano_xyz_vs.x);
			atomicAdd(&dL_dcano_points_xyz_view[global_id].y, dL_dg_cano_xyz_vs.y);
			atomicAdd(&dL_dcano_points_xyz_view[global_id].z, dL_dg_cano_xyz_vs.z);

			// Accumuler les gradients pour weight2_color et bias2_color
			#pragma unroll
			for (int i = 0; i < 8; i++) {
				atomicAdd(&dL_dweight2_color[global_id].rows[i].x, dL_dg_weight2_color.rows[i].x);
				atomicAdd(&dL_dweight2_color[global_id].rows[i].y, dL_dg_weight2_color.rows[i].y);
				atomicAdd(&dL_dweight2_color[global_id].rows[i].z, dL_dg_weight2_color.rows[i].z);
			}
			atomicAdd(&dL_dbias2_color[global_id].x, dL_dg_bias2_color.x);
			atomicAdd(&dL_dbias2_color[global_id].y, dL_dg_bias2_color.y);
			atomicAdd(&dL_dbias2_color[global_id].z, dL_dg_bias2_color.z);

#pragma unroll
			for (int i = 0; i < 3; i++)
			{
				atomicAdd(&dL_dinv_mat_modelview[global_id].rows[i].x, dL_dg_inv_mat_modelview.rows[i].x);
				atomicAdd(&dL_dinv_mat_modelview[global_id].rows[i].y, dL_dg_inv_mat_modelview.rows[i].y);
				atomicAdd(&dL_dinv_mat_modelview[global_id].rows[i].z, dL_dg_inv_mat_modelview.rows[i].z);
			}
#pragma unroll
			for (int i = 0; i < 8; i++)
			{
				atomicAdd(&dL_dweight1[global_id].rows[i].x, dL_dg_weight1.rows[i].x);
				atomicAdd(&dL_dweight1[global_id].rows[i].y, dL_dg_weight1.rows[i].y);
				atomicAdd(&dL_dweight1[global_id].rows[i].z, dL_dg_weight1.rows[i].z);

				atomicAdd(&dL_dbias1[global_id].data[i], dL_dg_bias1.data[i]);
				atomicAdd(&dL_dweight2[global_id].data[i], dL_dg_weight2.data[i]);
			}
			atomicAdd(&dL_dbias2[global_id], dL_dg_bias2);
			atomicAdd(&dL_dinv_max_radius[global_id], dL_dg_inv_max_radius);
		}
	}
}

void BACKWARD_MLP::preprocess(
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
	bool antialiasing)
{

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessMLPCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M,
		(float3 *)means_world,
		radii,
		shs,
		clamped,
		(glm::vec3 *)scales,
		(glm::vec4 *)rotations,
		scale_modifier,
		proj,
		view,
		campos,
		dL_dpoints_xyz_view,
		dL_dinv_max_radius,
		max_dim,
		dL_dcolor,
		dL_dinv_mat_modelview,
		(glm::vec3 *)dL_dmeans_world,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dopacity);
}

void BACKWARD_MLP::renderMLP(
	const dim3 grid,
	const dim3 block,
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
	const float8x3 *weight2_color,  // NOUVEAU
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
	float8x3 *dL_dweight2_color,  // NOUVEAU
    float3 *dL_dbias2_color,
	float3x3 *dL_dinv_mat_modelview,
	float *dL_dinvdepths)
{
	renderMLPCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		W, H,
		fx, fy,
		bg_color,
		viewmatrix,
		means3D_view,
		cano_means3D_view,
		inv_max_radius,
		weight1,
		bias1,
		weight2,
		bias2,
		weight2_color,
		bias2_color,
		inv_mat_modelview,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_invdepths,
		dL_dmeans3D_view,
		dL_dcano_means3D_view,
		dL_dinv_max_radius,
		dL_dweight1,
		dL_dbias1,
		dL_dweight2,
		dL_dbias2,
		dL_dweight2_color,  // NOUVEAU
        dL_dbias2_color,
		dL_dinv_mat_modelview,
		dL_dinvdepths);
}
